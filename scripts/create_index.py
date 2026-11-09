import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

print("正在為 Qdrant 的 date 欄位建立索引")

# 為 project 集合中的 date 欄位建立 datetime 索引
client.create_payload_index(
    collection_name="project",
    field_name="date",
    field_schema="datetime"
)

print("索引建立完成！")