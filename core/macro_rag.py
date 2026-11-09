from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

class MacroRAGLayer:
    def __init__(self, db_path: str = "./qdrant_db"):
        # 1. 初始化 Qdrant (會自動在本地建立資料夾儲存向量)
        self.client = QdrantClient(path=db_path)
        self.collection_name = "macro_black_swans"
        
        # 2. 設定 Dense 模型 (語意檢索)：使用 BAAI 的輕量強大模型
        self.client.set_model("BAAI/bge-small-en-v1.5")
        
        # 3. 設定 Sparse 模型 (BM25 關鍵字檢索)：捕捉專有名詞如 "QQQ", "CPI"
        self.client.set_sparse_model("Qdrant/bm25")
        
        # 4. 若集合不存在則建立
        if not self.client.collection_exists(self.collection_name):
            print(f"建立新的 Qdrant 集合: {self.collection_name} (支援 Hybrid Search)")
            # 建立支援多向量的集合會由 Qdrant 依據上述載入的模型自動處理
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.client.get_fastembed_vector_params(),
                sparse_vectors_config=self.client.get_fastembed_sparse_vector_params()
            )
    
    def ingest_documents(self, documents: list[str], metadatas: list[dict]):
        """
        將歷史報告文本切塊後匯入資料庫
        """
        print(f"正在向量化並匯入 {len(documents)} 筆歷史文獻...")
        
        # Qdrant 客戶端會自動在背景使用 fastembed 計算 Dense 與 BM25 向量
        self.client.add(
            collection_name=self.collection_name,
            documents=documents,
            metadata=metadatas,
            parallel=0  # 自動使用所有 CPU 核心加速
        )
        print("✅ 歷史文獻匯入完成！")

    def retrieve_hybrid_context(self, current_news: str, top_k: int = 3) -> str:
        """
        啟動 Hybrid Search (BM25 + Dense)，並透過 RRF 演算法撈出相似歷史經驗
        """
        # Qdrant 的 query_text 只要搭配載入好的雙模型，就會自動執行 Hybrid Search
        search_results = self.client.query(
            collection_name=self.collection_name,
            query_text=current_news,
            limit=top_k,
            # 指定使用 RRF (Reciprocal Rank Fusion) 進行排名融合
            # 這能確保「語意相似」和「關鍵字命中」的文獻都被排到最前面
            query_filter=None 
        )
        
        # 將撈出的歷史文獻組合成 Agent 的背景脈絡
        context_parts = []
        for idx, result in enumerate(search_results):
            # result.document 是文本內容，result.metadata 可存放如日期、事件名稱等
            date = result.metadata.get('date', 'Unknown Date')
            event = result.metadata.get('event_type', 'Historical Event')
            
            context_parts.append(f"[歷史事件 {idx+1} | 日期: {date} | 類型: {event}]\n{result.document}")
            
        combined_context = "\n\n".join(context_parts)
        return combined_context






    import os

if __name__ == "__main__":
    rag_layer = MacroRAGLayer()
    
    # 若資料庫是空的，才進行匯入 (避免每次執行都重複匯入)
    if rag_layer.client.count(rag_layer.collection_name).count == 0:
        print("資料庫為空，開始從資料夾讀取歷史資料...")
        
        data_folder = "./data/historical/"
        historical_docs = []
        historical_meta = []
        
        # 掃描資料夾內的所有 .txt 檔案
        if os.path.exists(data_folder):
            for filename in os.listdir(data_folder):
                if filename.endswith(".txt"):
                    filepath = os.path.join(data_folder, filename)
                    
                    # 讀取檔案內容
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    historical_docs.append(content)
                    
                    # 將檔名作為 Metadata 存起來，方便日後追溯
                    historical_meta.append({
                        "source_file": filename,
                        "event_type": "Historical Analysis"
                    })
            
            # 呼叫我們寫好的匯入函式，正式寫入資料庫
            if historical_docs:
                rag_layer.ingest_documents(historical_docs, historical_meta)
            else:
                print("找不到任何 .txt 檔案！")
        else:
            print(f"找不到資料夾 {data_folder}，請先建立並放入資料。")
