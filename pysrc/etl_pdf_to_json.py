import os
import json
import time
import base64
from pydantic import BaseModel, Field
from typing import List, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# --- 1. 設定區 (Config) ---
if "GOOGLE_API_KEY" not in os.environ:
    print("❌ 錯誤: 請設定 GOOGLE_API_KEY 環境變數")
    exit(1)

# 初始化新版 Client
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

INPUT_DIR = "./data/raw_pdfs"
OUTPUT_DIR = "./data/processed_json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. 定義資料結構 (Pydantic Schema) ---
# 新版 SDK 支援直接傳入 Pydantic Class，這樣 Gemini 就絕對不會吐錯格式！
# 同義詞是多出來的, 在跑DOCX時還沒有, 下次重跑PDF時再補
class SynonymEntry(BaseModel):
    slang: str = Field(description="客戶常說的口語 (如: 死掉, 殘廢, 存錢)")
    formal: str = Field(description="對應的保單專業術語 (如: 身故給付, 完全失能)")
class FaqItem(BaseModel):
    q: str = Field(description="使用者可能問的問題")
    a: str = Field(description="根據文件的簡短回答")

class BasicInfo(BaseModel):
    product_name: str = Field(description="完整商品名稱")
    product_code: str = Field(description="備查文號/核准文號 (例如: 114.01.01臺壽字第...號)")
    company: str = Field(description="保險公司名稱")
    currency: List[str] = Field(description="幣別列表 (例如: ['TWD', 'USD'])")
    product_type: str = Field(description="商品類型描述 (例如: 變額萬能壽險, 傳統型美元養老險)")
    payment_period: str = Field(description="繳費年期/方式")

class Conditions(BaseModel):
    age_range: str = Field(description="投保年齡限制")
    premium_limit: str = Field(description="保費門檻限制")
    fees_and_discounts: str = Field(description="相關費用率或高保費折扣說明")

class Coverage(BaseModel):
    death_benefit: str = Field(description="身故/喪葬給付計算邏輯")
    maturity_benefit: str = Field(description="滿期/祝壽金給付邏輯")
    other_benefits: List[str] = Field(description="其他給付項目 (如完全失能, 意外給付)")

class Investment(BaseModel):
    is_investment_linked: bool = Field(description="是否為投資型保單")
    features: List[str] = Field(description="投資特色 (如: ['月撥回', '全權委託'])")
    risks: List[str] = Field(description="風險揭露")

class RagData(BaseModel):
    keywords: List[str] = Field(description="RAG 檢索用的關鍵字與同義詞")
    synonym_mapping: List[SynonymEntry] = Field(description="口語與專業術語對照表")
    target_audience: str = Field(description="適合客群描述")
    faq: List[FaqItem] = Field(description="5-8 組常見問答")

class PolicyData(BaseModel):
    basic_info: BasicInfo
    conditions: Conditions
    coverage: Coverage
    investment: Investment
    rag_data: RagData

# --- 3. 核心處理函式 ---

def process_single_pdf(pdf_path, filename):
    print(f"   📤 讀取 PDF: {filename}...")
    
    # 新版 SDK 支援直接讀取本地檔案並 encode，不需要先 upload 再 delete (針對小檔案更快)
    # 但為了穩定性，針對大 PDF，我們還是用 File API
    
    try:
        # A. 上傳檔案 (File API)
        with open(pdf_path, "rb") as f:
            file_content = client.files.upload(
                file=f, 
                config=dict(
                    display_name=filename,
                    mime_type='application/pdf'
                )
            )
        
        # 等待處理
        while file_content.state == "PROCESSING":
            time.sleep(1)
            file_content = client.files.get(name=file_content.name)

        if file_content.state == "FAILED":
            raise ValueError("PDF 上傳失敗")
            
        print("   🤖 Gemini 分析提取中 (Using Pydantic Schema)...")
        
        # B. 生成內容 (使用 Structured Output)
        """
        👉 gemini-2.5-flash
👉 gemini-2.0-flash-exp
👉 gemini-2.0-flash
👉 gemini-2.0-flash-001
👉 gemini-2.0-flash-exp-image-generation
👉 gemini-2.0-flash-lite-001
👉 gemini-2.0-flash-lite
👉 gemini-2.0-flash-lite-preview-02-05
👉 gemini-2.0-flash-lite-preview
👉 gemini-2.5-flash-preview-tts
👉 gemini-flash-latest
👉 gemini-flash-lite-latest
👉 gemini-2.5-flash-lite
👉 gemini-2.5-flash-image-preview
👉 gemini-2.5-flash-image
👉 gemini-2.5-flash-preview-09-2025
👉 gemini-2.5-flash-lite-preview-09-2025
👉 gemini-3-flash-preview
👉 gemini-2.5-flash-native-audio-latest
👉 gemini-2.5-flash-native-audio-preview-09-2025
👉 gemini-2.5-flash-native-audio-preview-12-2025
        """
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", # 或 gemini-2.0-flash 如果你有權限
            contents=[
                file_content,
                "你是一位資深的保險精算師。請從這份保單中精確提取資料。請注意 product_code (文號) 的準確性。"
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PolicyData, # ★ 直接傳入 Pydantic Class
                temperature=0.1
            )
        )
        
        # C. 解析結果
        # SDK 會自動回傳符合 Schema 的物件，我們轉成 Dict 方便存 JSON
        # 注意: response.parsed 屬性在新版 SDK 會自動對應 Schema
        if response.parsed:
             # Pydantic model dump
            data = response.parsed.model_dump()
        else:
            # Fallback (很少發生)
            data = json.loads(response.text)

        # D. 加上原始檔名
        data["source_filename"] = filename
        
        # E. 清理雲端檔案
        client.files.delete(name=file_content.name)
        
        return data

    except Exception as e:
        print(f"❌ 處理失敗: {e}")
        return None

# --- 4. 主程式 ---
if __name__ == "__main__":
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
    total = len(files)
    
    print(f"🚀 開始處理 {total} 個檔案 (使用 google-genai SDK + Pydantic)")
    
    for i, filename in enumerate(files):
        json_name = os.path.splitext(filename)[0] + ".json"
        save_path = os.path.join(OUTPUT_DIR, json_name)
        
        if os.path.exists(save_path):
            print(f"⏩ [{i+1}/{total}] 跳過已存在: {filename}")
            continue

        try:
            print(f"\n🔄 [{i+1}/{total}] 處理: {filename}")
            start_time = time.time()
            
            result = process_single_pdf(os.path.join(INPUT_DIR, filename), filename)
            
            if result:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                duration = time.time() - start_time
                print(f"✅ 完成: {json_name} ({duration:.1f}s)")
            
            print("   💤 冷卻 5 秒...")
            time.sleep(5)

        except Exception as e:
            print(f"❌ 嚴重錯誤: {filename} - {e}")
            time.sleep(5)
        
    