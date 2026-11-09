import os
import json
from qdrant_client import QdrantClient
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from qdrant_client.models import Filter, FieldCondition, DatetimeRange, OrderBy, Direction

class FedStateOutput(BaseModel):
    fed_policy_phase: str = Field(description="根據文本判斷當前的聯準會政策階段與利率狀態 (請用繁體中文總結)，例如：降息循環 (4.75%-5.00%) 或 升息平臺期")
    key_focus: str = Field(description="總結聯準會目前最關注的經濟指標或市場焦點 (請用繁體中文總結)，例如：關注通膨與就業市場的平衡")

class StateTracker:
    def __init__(self, qdrant_client, collection_name: str, gemini_api_key: str):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.client = genai.Client(api_key=gemini_api_key, http_options=types.HttpOptions(
            api_version="v1beta",
            retry_options=types.HttpRetryOptions(attempts=5, initial_delay=2.0, max_delay=30.0, http_status_codes=[503]),
        ))
        self.model_name = 'gemini-2.5-flash'
        self._current_macro_state = self._fetch_and_analyze_latest_state()

    def _fetch_and_analyze_latest_state(self) -> dict:
        print("🔄 State Tracker: 正在從 Qdrant 尋找最新的 FOMC 紀錄")
        
        response, _ = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            limit=1,
            order_by=OrderBy(key="date", direction=Direction.DESC),
            with_payload=["date"],
            with_vectors=False
        )

        if not response:
            raise RuntimeError("Qdrant 沒有資料，無法取得最新 FOMC 紀錄")

        latest_date = response[0].payload["date"]

        records = []
        offset = None
        while True:
            batch, offset = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="date", range=DatetimeRange(gte=latest_date, lte=latest_date))]
                ),
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            records.extend(batch)
            if offset is None:
                break

        # 3. 將這些區塊合併成一篇完整文章
        latest_content = "\n".join([r.payload.get("content", "") for r in records]).strip()

        if not latest_content:
            raise RuntimeError(f"{latest_date} 的 FOMC 紀錄沒有內文")
        print(f"成功拼接 {len(records)} 個區塊作為最新 FOMC 紀錄 ({latest_date})。")

        print("\n=== 預覽即將送給 Gemini 的 FOMC 文本 ===")
        print(latest_content[:500])
        print("==================================================\n")

        print("正在呼叫 Gemini 2.5 Flash 進行狀態解析")

        prompt = f"""請作為專業的總經分析師，閱讀以下聯準會最新的會議紀錄片段，總結當前的貨幣政策階段與核心關注點。
        
        【嚴格要求】：
        1. 請務必根據以下 <document> 的真實內容進行總結。
        2. 絕對不要照抄預設的範例字眼，必須根據文本內容給出客製化的精確描述。
        
        <document>{latest_content}</document>
        """

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FedStateOutput,
                temperature=0.3
            )
        )
        
        parsed_result = response.parsed
        
        return {
            "fed_policy_phase": parsed_result.fed_policy_phase, 
            "key_focus": parsed_result.key_focus
        }

    def get_current_fed_state(self) -> str:
        state = self._current_macro_state
        return f"當前 Fed 狀態：{state['fed_policy_phase']}\n市場核心關注：{state['key_focus']}"

    def get_state(self) -> dict:
        return self._current_macro_state


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL", "https://f8874759-e225-42e1-85f2-dc759e4e9b1e.ca-central-1-0.aws.cloud.qdrant.io")
    
    # 修正點：絕對不要將真實金鑰寫在這裡，請確保它存在你的 .env 檔案中
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") 
    if not QDRANT_API_KEY:
        print("找不到 QDRANT_API_KEY，請確認 .env 檔案設定！")
        
    COLLECTION_NAME = "project"

    print("\n=== 開始測試 Step 1: State Tracker ===")
    
    test_qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    tracker = StateTracker(
        qdrant_client=test_qdrant_client, 
        collection_name=COLLECTION_NAME,    
        gemini_api_key=GEMINI_API_KEY
    )
    
    print("\n最終輸出的全域狀態：")
    print(tracker.get_current_fed_state())
    print("=======================================\n")