import json
from enum import Enum

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

EPS = 1e-6


class MarketStance(str, Enum):
    BULLISH = "Bullish"
    NEUTRAL = "Neutral"
    BEARISH = "Bearish"


class Position(str, Enum):
    OVERWEIGHT = "Overweight"
    NEUTRAL = "Neutral"
    UNDERWEIGHT = "Underweight"


class HistoricalAnalog(BaseModel):
    doc_id: str = Field(description="引用的歷史文獻編號，例如 D1，必須是提供的文獻之一")
    lesson: str = Field(description="這份歷史文獻對當前情境的啟示 (繁體中文)")


class AssetConstraint(BaseModel):
    ticker: str = Field(description="標的代號，必須是指定的可投資標的之一")
    stance: Position = Field(description="對此標的的配置態度")
    min_weight: float = Field(description="此標的在投資組合中的最低權重 (0~1)")
    max_weight: float = Field(description="此標的在投資組合中的最高權重 (0~1)")
    rationale: str = Field(description="從新聞、Fed 狀態與歷史文獻推論到此約束的理由 (繁體中文)")


class Agent1Output(BaseModel):
    market_stance: MarketStance = Field(description="對整體風險性資產的態度")
    summary: str = Field(description="綜合新聞、Fed 狀態與歷史類比的結論 (繁體中文)")
    historical_analogs: list[HistoricalAnalog] = Field(description="支持結論的歷史文獻引用")
    constraints: list[AssetConstraint] = Field(description="每個可投資標的各一筆權重約束")
    risk_flags: list[str] = Field(description="可能推翻此判斷的主要風險 (繁體中文)")
    confidence: float = Field(description="對此判斷的信心 (0~1)")


class MacroAgent:
    """Step 4 - Agent 1：綜合即時新聞、Fed 狀態與歷史文獻，產出給 optimizer 用的權重約束。"""

    def __init__(self, api_key: str, universe: list[str]):
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
            api_version="v1beta",
            retry_options=types.HttpRetryOptions(attempts=5, initial_delay=2.0, max_delay=30.0, http_status_codes=[503]),
        ))
        self.model_name = 'gemini-2.5-flash'
        self.universe = universe

    @staticmethod
    def _number_documents(rag_results: list[dict]) -> tuple[dict[str, dict], str]:
        """把檢索結果去重並編號 (D1, D2...)，回傳 {編號: 文獻} 與給 LLM 看的文字。"""
        docs: dict[str, dict] = {}
        seen: dict[tuple, str] = {}
        query_lines = []
        for r in rag_results:
            ids = []
            for doc in r["documents"]:
                key = (doc["date"], doc["type"], doc["content"])
                if key not in seen:
                    seen[key] = f"D{len(seen) + 1}"
                    docs[seen[key]] = doc
                if seen[key] not in ids:
                    ids.append(seen[key])
            query_lines.append(
                f"- 查詢「{r['search_query']}」（{r['start_date']} ~ {r['end_date']}）\n"
                f"  選擇此時期的理由：{r.get('historical_context', '')}\n"
                f"  找到的文獻：{', '.join(ids) if ids else '（無）'}"
            )
        doc_lines = [
            f"[{doc_id}｜{doc['date']}｜{doc['type']}]\n{doc['content']}"
            for doc_id, doc in docs.items()
        ]
        return docs, "\n".join(query_lines) + "\n\n" + "\n\n".join(doc_lines)

    def _validate(self, output: Agent1Output, docs: dict[str, dict]) -> list[str]:
        errors = []
        tickers = [c.ticker for c in output.constraints]
        if sorted(tickers) != sorted(self.universe):
            errors.append(f"constraints 的 ticker 必須剛好是 {self.universe} 各一筆，目前是 {tickers}")
        for c in output.constraints:
            if not (0 <= c.min_weight <= c.max_weight <= 1):
                errors.append(f"{c.ticker} 的權重不合法：須滿足 0 <= min_weight <= max_weight <= 1")
        min_sum = sum(c.min_weight for c in output.constraints)
        max_sum = sum(c.max_weight for c in output.constraints)
        if min_sum > 1 + EPS:
            errors.append(f"所有 min_weight 加總為 {min_sum:.2f}，不可超過 1")
        if max_sum < 1 - EPS:
            errors.append(f"所有 max_weight 加總為 {max_sum:.2f}，不可低於 1")
        for a in output.historical_analogs:
            if a.doc_id not in docs:
                errors.append(f"historical_analogs 引用了不存在的文獻編號 {a.doc_id}")
        if docs and not output.historical_analogs:
            errors.append("historical_analogs 至少要引用一份提供的歷史文獻")
        if not (0 <= output.confidence <= 1):
            errors.append("confidence 必須介於 0 與 1 之間")
        return errors

    def analyze(self, runtime_news: str, fed_state: dict, rag_results: list[dict]) -> dict:
        print("\n[Step 4 - Agent 1] 正在綜合新聞、Fed 狀態與歷史文獻，產出權重約束")
        docs, history_text = self._number_documents(rag_results)

        prompt = f"""
        你是一位專業的總經策略分析師 (Agent 1)，負責為下游的投資組合最佳化器設定權重約束。
        請綜合【當前 Fed 狀態】、【即時新聞】與【歷史文獻】，判斷市場態度並為每個可投資標的設定權重上下限。

        【嚴格要求】：
        1. 可投資標的只有：{', '.join(self.universe)}。constraints 必須每個標的剛好一筆。
        2. 權重以 0~1 的小數表示，須滿足 0 <= min_weight <= max_weight <= 1，
           且所有 min_weight 加總 <= 1、所有 max_weight 加總 >= 1。
        3. historical_analogs 只能引用下方出現的文獻編號 (如 D1)，不可引用其他文獻或自行編造。
        4. 歷史文獻是 Fed 的聲明與會議紀錄，不會直接談到個別標的；
           請在每個 rationale 中寫清楚從文獻與新聞推論到該標的的邏輯。
        5. 新聞附有發布時間，請以較新的新聞為主。
        6. 文字欄位請使用繁體中文。

        【當前 Fed 狀態】: {fed_state['fed_policy_phase']}
        【市場核心關注】: {fed_state['key_focus']}

        【即時新聞】:
        {runtime_news}

        【歷史文獻】:
        {history_text}
        """

        feedback = ""
        for attempt in range(2):
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt + feedback,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=Agent1Output,
                    temperature=0.2,
                ),
            )
            output = Agent1Output.model_validate_json(response.text)
            errors = self._validate(output, docs)
            if not errors:
                break
            print(f"   ↳ 第 {attempt + 1} 次輸出未通過檢查: {errors}")
            feedback = (
                "\n【你上一次的輸出有以下錯誤，請修正後重新輸出】:\n"
                + "\n".join(f"- {e}" for e in errors)
            )
        else:
            raise ValueError(f"Agent 1 的輸出兩次都未通過檢查: {errors}")

        result = output.model_dump(mode="json")
        for analog in result["historical_analogs"]:
            doc = docs[analog["doc_id"]]
            analog["date"] = doc["date"]
            analog["type"] = doc["type"]
        return result


if __name__ == "__main__":
    # 用已存檔的 Step 1-3 結果單獨測試 Step 4：python -m core.agent1
    import os
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv()
    saved = json.loads(Path("data/latest_rag.json").read_text(encoding="utf-8"))
    news = "\n".join(
        f"[{n['source']}｜{n['published_at']}] {n['title']}" for n in saved["news"]
    )

    agent = MacroAgent(os.environ["GEMINI_API_KEY"], ["SPY", "QQQ", "TLT", "GLD"])
    decision = agent.analyze(
        runtime_news=news,
        fed_state=saved["fed_state"],
        rag_results=saved["retrieval"],
    )
    print("\n=== Step 4: Agent 1 產出的 JSON 約束 ===")
    print(json.dumps(decision, indent=2, ensure_ascii=False))
