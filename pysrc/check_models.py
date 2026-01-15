import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

# 確保環境變數有設定
if "GOOGLE_API_KEY" not in os.environ:
    print("❌ 請先設定 GOOGLE_API_KEY 環境變數")
    exit(1)

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

print("🔍 正在查詢您的 API Key 可用模型列表...\n")

try:
    # 列出所有模型
    # config={'page_size': 100} 可以確保列出足夠多
    for m in client.models.list():
        name = m.name.replace("models/", "")
        
        # 我們只關心 Flash 系列，因為它們通常比較便宜/額度高
        if "flash" in name:
            print(f"👉 {name}")
            # print(f"   (版本: {m.version}, 支援: {m.supported_generation_methods})")

    print("\n💡 建議選擇含有 '-001', '-002' 或 '8b' 結尾的舊版 Flash，通常免費額度較高。")

except Exception as e:
    print(f"❌ 查詢失敗: {e}")