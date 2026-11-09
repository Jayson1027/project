import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from core.clawer import FinancialNewsCrawler
from core.fed_retriever import FedRetriever
from main import NEWS_SOURCES
from scripts.query_expansion import QueryExpansionEngine
from scripts.state_tracker import StateTracker

COLLECTION_NAME = "project"
OUTPUT_PATH = Path("data/latest_rag.json")


def format_news(articles: list[dict]) -> str:
    return "\n\n".join(
        f"[{a['source']}] {a['title']}\n{a['content'][:3000]}" for a in articles
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
    articles = asyncio.run(crawler.fetch_latest())
    if not articles:
        raise RuntimeError("沒有抓到任何新聞")
    print(f"抓到 {len(articles)} 篇新聞")

    state = StateTracker(qdrant, COLLECTION_NAME, gemini_key).get_state()

    plan = QueryExpansionEngine(gemini_key).generate_plan(
        runtime_news=format_news(articles),
        fed_state=state["fed_policy_phase"],
        market_focus=state["key_focus"],
    )

    retriever = FedRetriever(qdrant, COLLECTION_NAME)
    results = retriever.retrieve(plan)
    context = retriever.build_context(results)
    print(context)

    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "news": [
                    {"source": a["source"], "published_at": a["published_at"], "title": a["title"]}
                    for a in articles
                ],
                "fed_state": state,
                "retrieval": results,
                "context": context,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"結果已存到 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
