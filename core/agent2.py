"""Step 5 - Agent 2：把 Agent 1 的大環境判斷轉成 4 檔 ETF 的權重上下限，交給 Markowitz 當 bounds。"""
import json
from enum import Enum

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

SAFE_HAVENS = ("TLT", "GLD")
MIN_WIDTH = 0.10  # 每檔上下限至少相差 10%，保留空間給 Markowitz
EPS = 1e-9  # 浮點數誤差容許值，例如 0.3 - 0.2 = 0.0999...


class Asset(str, Enum):
    SPY = "SPY"
    QQQ = "QQQ"
    TLT = "TLT"
    GLD = "GLD"


class AssetBound(BaseModel):
    asset: Asset
    lower: float = Field(description="權重下限 (0~1)", ge=0, le=1)
    upper: float = Field(description="權重上限 (0~1)", ge=0, le=1)
    reason: str = Field(description="這檔資產上下限的依據 (繁體中文)")


class Agent2Output(BaseModel):
    bounds: list[AssetBound] = Field(description="SPY、QQQ、TLT、GLD 各一筆")
    safe_haven_min: float = Field(description="避險資產 TLT + GLD 合計的最低比例 (0~1)", ge=0, le=1)
    reasoning: str = Field(description="整體配置邏輯 (繁體中文)")


class ConstraintAgent:
    """Step 5 - Agent 2：依 Agent 1 的大環境判斷，產出各資產權重上下限。"""

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
            api_version="v1beta",
            retry_options=types.HttpRetryOptions(attempts=5, initial_delay=2.0, max_delay=30.0, http_status_codes=[503]),
        ))
        self.model_name = "gemini-2.5-flash"

    @staticmethod
    def _validate(output: Agent2Output) -> list[str]:
        errors = []
        b = {x.asset.value: x for x in output.bounds}
        missing = [a.value for a in Asset if a.value not in b]
        if missing or len(output.bounds) != len(Asset):
            return [f"bounds 必須剛好包含 SPY、QQQ、TLT、GLD 各一筆，缺少 {missing}"]
        for a, x in b.items():
            if x.upper - x.lower < MIN_WIDTH - EPS:
                errors.append(f"{a} 上限 {x.upper} 與下限 {x.lower} 相差不足 {MIN_WIDTH}")
        if sum(x.lower for x in b.values()) > 1 + EPS:
            errors.append("4 檔下限加總超過 1")
        if sum(x.upper for x in b.values()) < 1 - EPS:
            errors.append("4 檔上限加總不足 1")
        if output.safe_haven_min > sum(b[a].upper for a in SAFE_HAVENS) + EPS:
            errors.append("safe_haven_min 超過 TLT 與 GLD 上限的加總")
        if output.safe_haven_min + sum(b[a].lower for a in b if a not in SAFE_HAVENS) > 1 + EPS:
            errors.append("safe_haven_min 加上 SPY、QQQ 下限超過 1")
        return errors

    def build(self, macro: dict) -> dict:
        print("\n[Step 5 - Agent 2] 正在依大環境判斷產出權重上下限")
        prompt = f"""
        你是一位量化投資組合經理 (Agent 2)。上游的 Agent 1 已判斷當前市場大環境，
        請據此為 SPY（美股大盤）、QQQ（科技股）、TLT（長天期美債）、GLD（黃金）設定權重上下限。
        你的上下限會交給 Markowitz 最佳化求解最高夏普值的權重，所以你只決定範圍，不決定最終權重。

        【設定原則】：
        1. 市場偏空、風險高或 VIX 急升時：壓低 SPY、QQQ 的上限，提高 TLT、GLD 與 safe_haven_min 的下限。
        2. 市場偏多、風險低時：放寬股票上限，可提高強勢資產的下限。
        3. 參考各資產情緒、通膨、利率與衰退風險，調整對應資產的範圍。
        4. confidence 為 LOW 時，上下限保持接近中性、不要極端；MEDIUM 時可偏向一方，但不要極端；HIGH 時才可明顯偏向一方。

        【嚴格要求】：
        1. bounds 必須剛好包含 SPY、QQQ、TLT、GLD 各一筆，數值為 0~1 的比例。
        2. 每檔 upper − lower 至少 {MIN_WIDTH}，保留空間給 Markowitz。
        3. 4 檔 lower 加總 ≤ 1，4 檔 upper 加總 ≥ 1。
        4. safe_haven_min 不可超過 TLT 與 GLD 的 upper 加總，且加上 SPY、QQQ 的 lower 不可超過 1。
        5. 每檔的 reason 要指出依據 Agent 1 的哪些欄位。文字請使用繁體中文。
        6. 若影響同一資產的訊號互相矛盾（例如通膨風險高對 TLT 不利，但避險需求對 TLT 有利），
           reason 必須同時列出有利與不利的訊號，並說明如何取捨。
        7. reason 引用的數值與描述必須與 Agent 1 的輸出一致，例如情緒分數為正數時不可描述為偏低。

        【Agent 1 大環境判斷】:
        {json.dumps(macro, ensure_ascii=False, indent=2)}
        """

        feedback = ""
        for attempt in range(2):
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt + feedback,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=Agent2Output,
                    temperature=0.0,
                ),
            )
            try:
                output = Agent2Output.model_validate_json(response.text)
                errors = self._validate(output)
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
            raise ValueError(f"Agent 2 的輸出兩次都未通過檢查: {errors}")

        return {
            "bounds": {x.asset.value: [x.lower, x.upper] for x in output.bounds},
            "safe_haven_min": output.safe_haven_min,
            "reasons": {x.asset.value: x.reason for x in output.bounds},
            "reasoning": output.reasoning,
        }


if __name__ == "__main__":
    # 用已存檔的 Agent 1 結果單獨測試：python -m core.agent2
    import os
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv(".env")
    macro = json.loads(Path("data/latest_macro.json").read_text(encoding="utf-8"))
    result = ConstraintAgent(os.environ["GEMINI_API_KEY"]).build(macro)
    print(json.dumps(result, indent=2, ensure_ascii=False))
