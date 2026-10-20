# scripts/ingest_from_csv_url.py
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance  # 補上 VectorParams, Distance
from fastembed import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter

QDRANT_URL = "https://f8874759-e225-42e1-85f2-dc759e4e9b1e.ca-central-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6YzhmMmY2NGUtNmM0YS00NTRlLWI3ZWEtOWVjYTMwZDAzMDBlIn0.i4h0pWnutzV-5K1dLr7ya5IKjRLeILrj5cBJaL-c-3Y"
COLLECTION_NAME = "project"

# 開源社群維護的 2000 年至今 FOMC 全文資料庫 URL
CSV_URL = "https://raw.githubusercontent.com/vtasca/fed-statement-scraping/master/communications.csv"

def main():
    print("1. 從 GitHub Raw 網址讀取 20 年 FOMC 歷史紀錄 (純記憶體讀取)...")
    try:
        df = pd.read_csv(CSV_URL)
        print(f"成功載入資料！共 {len(df)} 筆 FOMC 發布紀錄。")
    except Exception as e:
        print(f"讀取失敗: {e}")
        return

    print("2. 連線至 Qdrant Cloud 並初始化 FastEmbed 模型...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60.0)
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)

    #自動檢查並建立 Collection
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"建立全新的 Qdrant Collection: '{COLLECTION_NAME}' (維度: 384, 距離演算法: Cosine)...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,                
                distance=Distance.COSINE 
            )
        )

    points = []
    point_id = 0

    print("3. 開始進行切塊與向量生成 (篩選 2004 年至今資料)...")
    for idx, row in df.iterrows():
        if idx % 20 == 0:
            print(f"正在處理第 {idx+1}/{len(df)} 筆 FOMC 資料...")
            
        date = str(row.get("Date", ""))
        comm_type = str(row.get("Type", ""))
        text = str(row.get("Text", ""))

        # 只保留 2004 年之後的紀錄且排除空資料
        if date < "2004-01-01" or not text or len(text) < 50:
            continue

        chunks = text_splitter.split_text(text)
        
        # 若切出來無文字則跳過
        if not chunks:
            continue

        embeddings = list(embedder.embed(chunks))

        for chunk, vector in zip(chunks, embeddings):
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload={
                        "date": date,
                        "type": comm_type,
                        "content": chunk
                    }
                )
            )
            point_id += 1

    print(f"4. 正將 {len(points)} 個歷史語意區塊分批上傳至 Qdrant Cloud...")
    batch_size = 50
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=COLLECTION_NAME, points=points[i:i+batch_size])
        print(f"進度: {min(i+batch_size, len(points))}/{len(points)}")

    print("✅ 20 年聯準會 FOMC 官方聲明與會議紀錄已完全上傳至 Qdrant Cloud！")

if __name__ == "__main__":
    main()