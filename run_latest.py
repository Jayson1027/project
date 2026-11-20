import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from core.agent1 import MacroAgent
from core.clawer import FinancialNewsCrawler
from core.fed_retriever import FedRetriever
from core.market_data import fetch_market_snapshot
from main import NEWS_SOURCES
from scripts.query_expansion import QueryExpansionEngine
from scripts.state_tracker import StateTracker

COLLECTION_NAME = "project"
OUTPUT_PATH = Path("data/latest_rag.json")
MACRO_PATH = Path("data/latest_macro.json")
NEWS_WINDOW = timedelta(days=7)


def recent_news(articles: list[dict], window: timedelta = NEWS_WINDOW) -> list[dict]:
    """只保留發布時間在 window 內的新聞；沒有時區的時間視為 UTC，無法解析時間的捨棄。"""
    cutoff = datetime.now(timezone.utc) - window
    recent = []
    for a in articles:
        dt = a.get("published_dt")
        if dt is None:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt >= cutoff:
            recent.append(a)
    return recent


def format_news(articles: list[dict]) -> str:
    return "\n\n".join(
        f"[{a['source']}｜{a['published_at']}] {a['title']}\n{a['content'][:3000]}"
        for a in articles
    )


def main():
    load_dotenv()
    gemini_key = os.environ["GEMINI_API_KEY"]
    qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
        timeout=60,
    )

    crawler = FinancialNewsCrawler(sources=NEWS_SOURCES)
    fetched = asyncio.run(crawler.fetch_latest())
    articles = recent_news(fetched)
    if not articles:
        raise RuntimeError(f"抓到 {len(fetched)} 篇新聞，但沒有一篇在最近 {NEWS_WINDOW.days} 天內")
    print(f"抓到 {len(fetched)} 篇新聞，最近 {NEWS_WINDOW.days} 天內 {len(articles)} 篇")
    news_text = format_news(articles)

    market = fetch_market_snapshot()
    print(f"市場數據：{market}")

    state = StateTracker(qdrant, COLLECTION_NAME, gemini_key).get_state()

    plan = QueryExpansionEngine(gemini_key).generate_plan(
        runtime_news=news_text,
        fed_state=state["fed_policy_phase"],
        market_focus=state["key_focus"],
    )

    retriever = FedRetriever(qdrant, COLLECTION_NAME)
    results = retriever.retrieve(plan)
    context = retriever.build_context(results)
    print(context)

    macro = MacroAgent(gemini_key).analyze(
        runtime_news=news_text,
        fed_state=state,
        rag_results=results,
        market_data=market,
    )
    print(json.dumps(macro, ensure_ascii=False, indent=2))

    generated_at = datetime.now().isoformat(timespec="seconds")
    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "news": [
                    {"source": a["source"], "published_at": a["published_at"], "title": a["title"]}
                    for a in articles
                ],
                "fed_state": state,
                "market_data": market,
                "retrieval": results,
                "context": context,
                "agent1": macro,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"結果已存到 {OUTPUT_PATH}")

    MACRO_PATH.write_text(
        json.dumps({"generated_at": generated_at, **macro}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"大環境判斷已存到 {MACRO_PATH}（供 Agent 2 使用）")


if __name__ == "__main__":
    main()
