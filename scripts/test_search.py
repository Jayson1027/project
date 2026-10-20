# scripts/test_search.py
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, DatetimeRange, PayloadSchemaType
from fastembed import TextEmbedding

QDRANT_URL = "https://f8874759-e225-42e1-85f2-dc759e4e9b1e.ca-central-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6YzhmMmY2NGUtNmM0YS00NTRlLWI3ZWEtOWVjYTMwZDAzMDBlIn0.i4h0pWnutzV-5K1dLr7ya5IKjRLeILrj5cBJaL-c-3Y"
COLLECTION_NAME = "project"

def main():
    print("1. 連線至 Qdrant Cloud 並載入 FastEmbed 模型...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # 🛠️ 為 date 欄位建立 datetime 索引 (只需建立一次，Qdrant 會自動維護)
    print("2. 檢查並為 'date' 欄位建立 Payload 索引...")
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="date",
            field_schema=PayloadSchemaType.DATETIME,
        )
    except Exception as e:
        # 如果索引已存在會拋出異常，直接忽略即可
        pass

    query_text = "How about the inflation in 2022 "
    print(f"\n🔍 測試查詢問題: '{query_text}'")

    # 1. 將查詢問題轉為 384 維向量
    query_vector = list(embedder.embed([query_text]))[0].tolist()

    # 2. 向 Qdrant Cloud 發起搜尋，並加上 2022 年日期過濾條件
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="date",
                    range=DatetimeRange(
                        gte="2022-01-01",
                        lte="2022-12-31"
                    )
                )
            ]
        ),
        limit=3
    )

    results = response.points

    # 3. 印出檢索結果
    print("\n" + "="*20 + " 🎯 語意檢索結果 (Top 3 - 僅限 2022 年) " + "="*20)
    if not results:
        print("⚠️ 未找到符合條件的歷史紀錄。")
        return

    for idx, hit in enumerate(results, 1):
        score = hit.score
        payload = hit.payload or {}
        date = payload.get("date", "N/A")
        comm_type = payload.get("type", "N/A")
        content = payload.get("content", "N/A")

        print(f"\n【結果 {idx}】 語意相似度分數: {score:.4f}")
        print(f"📅 官方發布日期: {date} | 📑 文件類型: {comm_type}")
        print(f"💬 歷史文本內容:\n{content}")
        print("-" * 60)

if __name__ == "__main__":
    main()