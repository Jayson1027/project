import json
from enum import Enum

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from core.market_data import format_market_data


class MarketCondition(str, Enum):
    BULLISH = "Bullish"
    NEUTRAL = "Neutral"
    BEARISH = "Bearish"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RateTrend(str, Enum):
    UPWARD = "UPWARD"
    STABLE = "STABLE"
    DOWNWARD = "DOWNWARD"


class MacroSignals(BaseModel):
    inflation_risk: RiskLevel = Field(description="通膨風險高低")
    interest_rate_trend: RateTrend = Field(description="利率走向：升息 UPWARD、維持 STABLE、降息 DOWNWARD")
    # vix_spike 不交給模型判斷，由 core/market_data.py 依 VIX 實際數值計算後補進輸出


class SentimentAnalysis(BaseModel):
    overall_risk_score: int = Field(description="整體市場風險分數，0 = 極低風險，100 = 極高風險", ge=0, le=100)
    tech_sector_sentiment: float = Field(description="科技類股情緒，-1 = 極度悲觀，1 = 極度樂觀", ge=-1, le=1)
    bond_sector_sentiment: float = Field(description="長天期公債情緒，-1 = 極度悲觀，1 = 極度樂觀", ge=-1, le=1)


class TacticalGuidance(BaseModel):
    suggested_safe_haven_bias: float = Field(description="建議避險資產（公債、黃金）至少佔的比例 (0~1)", ge=0, le=1)
    max_growth_asset_exposure: float = Field(description="建議成長型資產（科技股）最多佔的比例 (0~1)", ge=0, le=1)


class HistoricalAnalog(BaseModel):
    doc_id: str = Field(description="引用的歷史文獻編號，例如 D1，必須是提供的文獻之一")
    lesson: str = Field(description="這份歷史文獻對當前情境的啟示 (繁體中文)")


class Agent1Output(BaseModel):
    market_condition: MarketCondition = Field(description="整體市場體制")
    macro_signals: MacroSignals
    sentiment_analysis: SentimentAnalysis
    tactical_guidance: TacticalGuidance
    historical_analogs: list[HistoricalAnalog] = Field(description="真正可作為當前情境類比的歷史文獻引用；都不相符時為空陣列")
    reasoning: str = Field(description="從新聞、Fed 狀態與歷史文獻推論出上述判斷的過程 (繁體中文)")


class MacroAgent:
    """Step 4 - Agent 1：綜合即時新聞、Fed 狀態與歷史文獻，產出大環境判斷，交給 Agent 2 轉成權重上下限。"""

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
            api_version="v1beta",
            retry_options=types.HttpRetryOptions(attempts=5, initial_delay=2.0, max_delay=30.0, http_status_codes=[503]),
        ))
        self.model_name = 'gemini-2.5-flash'

    @staticmethod
    def _number_documents(rag_results: list[dict]) -> tuple[dict[str, dict], str]:
        """把檢索結果去重並編號 (D1, D2...)，回傳 {編號: 文獻} 與給 LLM 看的文字。

        只放文件本身（日期、類型、原文），不放 Step 2 的查詢與選擇理由，
        避免 Agent 1 沿用 Step 2 的推測。
        """
        docs: dict[str, dict] = {}
        seen: set[tuple] = set()
        for r in rag_results:
            for doc in r["documents"]:
                key = (doc["date"], doc["type"], doc["content"])
                if key not in seen:
                    seen.add(key)
                    docs[f"D{len(docs) + 1}"] = doc
        doc_lines = [
            f"[{doc_id}｜{doc['date']}｜{doc['type']}]\n{doc['content']}"
            for doc_id, doc in docs.items()
        ]
        return docs, "\n\n".join(doc_lines)

    @staticmethod
    def _validate(output: Agent1Output, docs: dict[str, dict]) -> list[str]:
        errors = []
        for a in output.historical_analogs:
            if a.doc_id not in docs:
                errors.append(f"historical_analogs 引用了不存在的文獻編號 {a.doc_id}")
        return errors

    def analyze(self, runtime_news: str, fed_state: dict, rag_results: list[dict], market_data: dict) -> dict:
        print("\n[Step 4 - Agent 1] 正在綜合新聞、Fed 狀態與歷史文獻，產出大環境判斷")
        docs, history_text = self._number_documents(rag_results)

        prompt = f"""
        你是一位專業的總經策略分析師 (Agent 1)，負責判斷當前的市場大環境。
        你的輸出會交給下游的 Agent 2，由它轉換成各資產的權重上下限，所以你不需要決定個別資產的權重。
        請綜合【當前 Fed 狀態】、【市場數據】、【即時新聞】與【歷史文獻】，填寫市場體制、總經訊號、情緒分數與戰術指引。

        【嚴格要求】：
        1. overall_risk_score 為 0~100 的整數；兩個 sector_sentiment 介於 -1 與 1 之間。
        2. suggested_safe_haven_bias 與 max_growth_asset_exposure 為 0~1 的比例。
        3. interest_rate_trend 要對照【市場數據】中十年期殖利率的實際變化判斷。
        4. historical_analogs 只能引用下方出現的文獻編號 (如 D1)，不可引用其他文獻或自行編造。
           只引用經濟情境真正與現在相似、可作為類比的文獻；主題相近但情境不同的不要引用。
           若所有文獻都不相符，historical_analogs 留空陣列，並在 reasoning 說明原因。
        5. reasoning 要寫清楚每個判斷依據哪些新聞、哪項市場數據與哪份文獻。
        6. 新聞附有發布時間，請以較新的新聞為主。
        7. 文字欄位請使用繁體中文。

        【當前 Fed 狀態】: {fed_state['fed_policy_phase']}
        【市場核心關注】: {fed_state['key_focus']}

        【市場數據】:
        {format_market_data(market_data)}

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
            try:
                output = Agent1Output.model_validate_json(response.text)
                errors = self._validate(output, docs)
            except ValidationError as e:
                errors = [f"{'.'.join(map(str, err['loc']))}：{err['msg']}" for err in e.errors()]
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
        result["macro_signals"]["vix_spike"] = market_data["vix_spike"]
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

    load_dotenv(".env")
    saved = json.loads(Path("data/latest_rag.json").read_text(encoding="utf-8"))
    news = "\n".join(
        f"[{n['source']}｜{n['published_at']}] {n['title']}" for n in saved["news"]
    )

    from core.market_data import fetch_market_snapshot

    market = saved.get("market_data") or fetch_market_snapshot()
    agent = MacroAgent(os.environ["GEMINI_API_KEY"])
    macro = agent.analyze(
        runtime_news=news,
        fed_state=saved["fed_state"],
        rag_results=saved["retrieval"],
        market_data=market,
    )
    print("\n=== Step 4: Agent 1 產出的大環境 JSON ===")
    print(json.dumps(macro, indent=2, ensure_ascii=False))
