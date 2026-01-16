import os
import json
import time
from docx import Document
from pydantic import BaseModel, Field
from typing import List
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# --- 設定 ---
if "GOOGLE_API_KEY" not in os.environ:
    print("❌ 錯誤: 請設定 GOOGLE_API_KEY")
    exit(1)

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

INPUT_DIR = "./data/raw_docx"        # DOCX 來源
OUTPUT_DIR = "./data/processed_json" # 輸出到跟 PDF JSON 一樣的地方
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 定義資料結構 (包含同義詞 Mapping) ---

class SynonymEntry(BaseModel):
    slang: str = Field(description="客戶常說的口語 (如: 死掉, 殘廢, 存錢)")
    formal: str = Field(description="對應的保單專業術語 (如: 身故給付, 完全失能)")

class FaqItem(BaseModel):
    q: str = Field(description="使用者可能問的問題")
    a: str = Field(description="簡短回答")

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
    keywords: List[str] = Field(description="檢索關鍵字列表")
    # 🔥 這是 DOCX 版特有的強項：同義詞對照表
    synonym_mapping: List[SynonymEntry] = Field(description="口語與專業術語對照表")
    target_audience: str = Field(description="適合客群")
    faq: List[FaqItem] = Field(description="常見問答")

class PolicyData(BaseModel):
    basic_info: BasicInfo
    conditions: Conditions
    coverage: Coverage
    investment: Investment
    rag_data: RagData

# --- 核心函式 ---

def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        return "\n".join(full_text)
    except Exception as e:
        print(f"❌ 讀取 DOCX 失敗: {e}")
        return None

def process_single_docx(file_path, filename):
    print(f"   📄 讀取 DOCX: {filename}...")
    text_content = extract_text_from_docx(file_path)
    
    if not text_content: return None

    try:
        print("   🤖 Gemini 分析中...")
        response = client.models.generate_content(
            model="gemini-3-flash-preview", 
            contents=[
                f"你是一位保險專家。請分析這份文件 (檔名: {filename}) 並提取 RAG 所需資料，特別是『客戶口語 vs 專業術語』的對照。",
                text_content[:30000] 
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PolicyData,
                temperature=0.1
            )
        )

        if response.parsed:
            data = response.parsed.model_dump()
            data["source_filename"] = filename
            return data
        return None

    except Exception as e:
        print(f"❌ Gemini 處理失敗: {e}")
        return None

# --- 主程式 ---
if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        print(f"⚠️ 目錄不存在: {INPUT_DIR} (請建立並放入 .docx 檔)")
        exit(0)

    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".docx")]
    print(f"🚀 開始處理 {len(files)} 個 DOCX 檔案")
    
    for filename in files:
        json_name = os.path.splitext(filename)[0] + ".json"
        save_path = os.path.join(OUTPUT_DIR, json_name)
        
        if os.path.exists(save_path):
            print(f"⏩ 跳過已存在: {filename}")
            continue

        print(f"\n🔄 處理: {filename}")
        result = process_single_docx(os.path.join(INPUT_DIR, filename), filename)
        
        if result:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✅ 完成")
        
        time.sleep(1)