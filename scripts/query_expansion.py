import os
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# 定義第一階段的輸出結構：只負責生 Query
class CoreQueries(BaseModel):
    queries: list[str] = Field(
        description="3個綜合當前 Fed 狀態、市場關注點與新聞所生成的英文語意搜尋關鍵字", 
        max_length=3, 
        min_length=3
    )

# 定義第二階段的輸出結構：負責配對時間與整合
class TemporalSubQuery(BaseModel):
    search_query: str = Field(description="從第一階段繼承來的英文搜尋關鍵字")
    start_date: str = Field(description="搜尋起始日期 (YYYY-MM-DD)")
    end_date: str = Field(description="搜尋結束日期 (YYYY-MM-DD)")
    historical_context: str = Field(description="簡述為何選擇此歷史時間段")

class QueryExpansionPlan(BaseModel):
    sub_queries: list[TemporalSubQuery] = Field(description="包含 3 個帶有時間範圍的子查詢")

class QueryExpansionEngine:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = 'gemini-2.5-flash'
        
    def generate_plan(self, runtime_news: str, fed_state: str, market_focus: str) -> list[dict]:
        print("\n[Step 2 - Phase 1] 正在綜合 Fed 狀態、市場關注與突發新聞，生成 3 個核心 Queries")
        
        prompt_phase1 = f"""
        你是一個專業的量化投資檢索引擎。
        請綜合以下【當前 Fed 狀態】、【市場核心關注】與【即時突發新聞】，思考歷史上可能發生過類似情況的宏觀經濟與產業環境。
        請生成 3 個不同切入點的「英文檢索字串 (search queries)」，用於後續在歷史文獻庫中進行語意搜索。
        
        【當前 Fed 狀態】: {fed_state}
        【市場核心關注】: {market_focus}
        【即時突發新聞】: {runtime_news}
        """
        
        response_p1 = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt_phase1,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CoreQueries,
                temperature=0.2
            )
        )
        
        core_queries = json.loads(response_p1.text)["queries"]
        for i, q in enumerate(core_queries):
            print(f"   ↳ Query {i+1}: {q}")
            
        print("\n[Step 2 - Phase 2] 正在為這 3 個 Queries 尋找對應的歷史 DatetimeRange")
        
        prompt_phase2 = f"""
        你剛才基於當前的總經與新聞狀態，生成了以下 3 個英文檢索字串：
        {json.dumps(core_queries, indent=2)}
        
        現在，請為這 3 個檢索字串，分別指定最合適的「歷史時間範圍 (DatetimeRange)」。
        
        【系統嚴格限制】：
        1. 我們的歷史文獻庫僅包含 2004-01-01 至 2024-12-31 的資料。
        2. start_date 與 end_date 必須嚴格落在此區間內。
        3. start_date 與 end_date 間隔須為剛好一年
        3. 請為每個時間段提供簡短的 historical_context 理由。
        """

        response_p2 = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt_phase2,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=QueryExpansionPlan,
                temperature=0.3
            )
        )
        
        plan_data = json.loads(response_p2.text)
        return plan_data["sub_queries"]

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    fed_state = "預期升息階段 (市場高度預期九月會議將升息25個基點，且年底前可能進一步緊縮)"
    market_focus = "關注持續性通膨、經濟韌性及地緣政治對能源價格的影響"
    runtime_news = "台積電 (2330) 法說會利多：CAPEX 上調至 380 億美元，加速 2nm 製程擴建"
    
    engine = QueryExpansionEngine(api_key=os.getenv("GEMINI_API_KEY"))
    
    # 將兩個總經變數分別傳入
    plan = engine.generate_plan(
        runtime_news=runtime_news, 
        fed_state=fed_state, 
        market_focus=market_focus
    )
    
    print("\n=== Step 2: 最終產出的檢索計畫 (Queries + DatetimeRange) ===")
    print(json.dumps(plan, indent=2, ensure_ascii=False))