import os
from dotenv import load_dotenv
from google import genai

# 讀取 .env 中的金鑰
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("🎯 你的 API Key 支援以下模型 (可生成內容)：")
# 列出所有可用的模型
for model in client.models.list():
    if "generateContent" in model.supported_actions:
        print(model.name)