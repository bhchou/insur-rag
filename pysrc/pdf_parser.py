import pdfplumber
import json
import sys
import pytesseract
import requests
import re
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 🔧 設定區 (請依據您的實際環境修改)
# ==========================================
#
VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT")  # 例如: http://192.168.1.100:8000
MODEL_NAME = os.getenv("MODEL_NAME")              # 例如: meta-llama/Llama-3-8b-instruct
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

def clean_json_string(text):
    """
    清理 LLM 回傳的字串，移除 Markdown 標記
    """
    cleaned = re.sub(r"```json\s*", "", text)
    cleaned = re.sub(r"```\s*", "", cleaned)
    return cleaned.strip()

def extract_metadata_via_llm(text_content):
    """
    呼叫 VLLM API 進行 Metadata 提取
    """
    # 1. URL 智慧處理 (自動補全 /v1)
    base_url = VLLM_ENDPOINT.rstrip('/')
    if '/v1' in base_url:
        api_url = f"{base_url}/chat/completions"
    else:
        api_url = f"{base_url}/v1/chat/completions"

    # 2. Header 設定 (避免 401 錯誤)
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "PDF-Parser"
    }
    # 只有當 Token 有效且不是 none 時才加入 Authorization
    if BEARER_TOKEN and str(BEARER_TOKEN).lower() not in ['none', '', 'null']:
        headers["Authorization"] = f"Bearer {BEARER_TOKEN}"

        
    #url = f"{VLLM_ENDPOINT}/v1/chat/completions"
    
    # 設置請求標頭 (參考您提供的程式碼)
    #headers = {"Content-Type": "application/json"}
    #if BEARER_TOKEN:
    #    headers["Authorization"] = f"Bearer {BEARER_TOKEN}"

    ###
    # 定義提示詞 (System Prompt + User Context)
    system_instruction = """
    你是一個專業的保險文件分析師。請分析使用者提供的 OCR 文字，提取以下 JSON 欄位。
    如果找不到對應資訊，請填 null 或空陣列 []。
    
    必須提取的欄位:
    1. product_name (字串): 產品全名
    2. product_code (字串): 文號或商品代碼
    3. insurance_type (字串陣列): 例如 ["終身壽險", "美元保單", "利率變動型"]
    4. target_audience (字串): 適合的對象描述
    5. benefits (字串陣列): 主要給付項目
    6. currency (字串): 幣別 (如 USD, TWD)

    請直接回傳 JSON 物件，不要包含任何解釋或 Markdown 格式。
    """

    # 為了避免超過 Token 上限，我們只取前 3000 個字 (通常 metadata 都在前幾頁)
    truncated_text = text_content[:3000]
    final_prompt = f"{system_instruction}\n\n=== 待分析文本 ===\n{truncated_text}"

    # 4. 建構 Payload
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": final_prompt}
        ],
        "temperature": 0.1, # 低溫以確保輸出格式穩定
        #"max_tokens": 1024,
        "stream": False
    }

    try:
        print(f"[Python] Calling LLM: {api_url}...", file=sys.stderr)
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            #content = resp_json["choices"][0]["message"]["content"]
            
            # 清理並解析 JSON
            #cleaned_content = clean_json_string(content)
            #print(f"[Python] LLM Response: {cleaned_content}...", file=sys.stderr) # debug log
            
            #return json.loads(cleaned_content)
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                return clean_json_string(content)
            else:
                print(f"[Error] Empty choices in response", file=sys.stderr)
                return "{}"
        else:
            print(f"[Error] VLLM API Error: {response.status_code} - {response.text}", file=sys.stderr)
            return {}

    except Exception as e:
        print(f"[Error] Connection failed: {str(e)}", file=sys.stderr)
        return {}

def parse_pdf(file_path):
    """
    (這是FOR PYO)
    讀取 PDF，簡單過濾頁首頁尾，回傳結構化資料。
    """
    result = {
        "file_path": file_path,
        "pages": [],
        "debug_info": [],
        "metadata": {}
    }

    full_text = ""
    
    try:
        with pdfplumber.open(file_path) as pdf:
            if len(pdf.pages) == 0:
                return json.dumps({"error": "PDF has 0 pages."})

            for i, page in enumerate(pdf.pages):
                # 先提取文字看看
                
                text = page.extract_text()
                method = "text_layer"

                # 如果文字太少 (例如 DM 轉外框)，則啟動 OCR
                if not text or len(text.strip()) < 10:
                    try:
                        # 將頁面轉為圖片 (解析度 300 dpi 以提升辨識率)
                        im = page.to_image(resolution=300).original
                        # 使用 Tesseract 辨識繁體中文 (chi_tra) + 英文 (eng)
                        text = pytesseract.image_to_string(im, lang='chi_tra+eng')
                        method = "ocr_fallback"
                    except Exception as e:
                        result["debug_info"].append(f"Page {i+1} OCR failed: {str(e)}")
                
                if text and len(text.strip()) > 0:
                    #簡單清洗
                    clean_text = text.strip()

                    result["pages"].append({
                        "page_number": i + 1,
                        "content": clean_text,
                        "method": method # 標記是用什麼方法讀到的
                    })
                    full_text += clean_text + "\n"

                else:
                    # 讀不到文字，記錄原因
                    result["debug_info"].append(f"Page {i+1}: No text layer found. (Likely scanned image or encrypted)")
        
        # --- 呼叫 LLM 進行 Metadata 提取 ---
        if len(full_text) > 0:
            # 確保有設定 Endpoint 才呼叫
            if "YOUR_VLLM_IP" not in VLLM_ENDPOINT: 
                result["metadata"] = extract_metadata_via_llm(full_text)
            else:
                result["debug_info"].append("Skipped LLM call: VLLM_ENDPOINT not configured.")
                
    except Exception as e:
        return json.dumps({"error": str(e)})

    return json.dumps(result, ensure_ascii=False)

def extract_text_from_pdf(file_path):
    """
    讀取 PDF 並轉為純文字 (含 OCR Fallback)
    """
    full_text = ""
    print(f"[Python] Processing PDF: {file_path}", file=sys.stderr)
    
    try:
        with pdfplumber.open(file_path) as pdf:
            if len(pdf.pages) == 0:
                return ""

            for i, page in enumerate(pdf.pages):
                # 嘗試直接提取文字
                text = page.extract_text()
                
                # 如果文字太少，啟動 OCR
                if not text or len(text.strip()) < 10:
                    try:
                        print(f"[Python] Page {i+1} using OCR...", file=sys.stderr)
                        # 解析度 300 dpi 以提升辨識率
                        im = page.to_image(resolution=300).original
                        text = pytesseract.image_to_string(im, lang='chi_tra+eng')
                    except Exception as e:
                        print(f"[Python] Page {i+1} OCR failed: {e}", file=sys.stderr)
                
                if text:
                    full_text += text.strip() + "\n"
                    
    except Exception as e:
        print(f"[Error] PDF Read Failed: {e}", file=sys.stderr)
        return ""
        
    return full_text

def main():
    # 檢查參數
    if len(sys.argv) < 2:
        # 錯誤訊息也輸出成 JSON 格式，方便 Rust 判讀
        print(json.dumps({"error": "No file path provided"}))
        return

    pdf_path = sys.argv[1]
    
    # 1. 提取全文 (Full Text)
    raw_text = extract_text_from_pdf(pdf_path)
    
    if not raw_text:
        # 如果讀不到字，回傳空的結構避免 Rust 解析失敗
        final_output = {
            "metadata": {
                "product_name": "Unknown",
                "product_code": None,
                "insurance_type": [],
                "benefits": [],
                "currency": "Unknown",
                "target_audience": None
            },
            "full_text": ""
        }
        print(json.dumps(final_output, ensure_ascii=False))
        return

    # 2. 呼叫 LLM 提取 Metadata
    metadata_json_str = extract_metadata_via_llm(raw_text)
    
    # 嘗試解析 LLM 回傳的 JSON
    try:
        metadata_obj = json.loads(metadata_json_str)
    except:
        print(f"[Python] JSON Parse Failed, Raw: {metadata_json_str}", file=sys.stderr)
        # Fallback 結構
        metadata_obj = {
            "product_name": "Unknown", 
            "product_code": None,
            "insurance_type": [],
            "benefits": [],
            "currency": "Unknown",
            "target_audience": None
        }

    # 3. 組裝最終結構
    final_output = {
        "metadata": metadata_obj,
        "full_text": raw_text
    }

    # 4. 輸出到 Stdout (Rust 讀取目標)
    print(json.dumps(final_output, ensure_ascii=False))

if __name__ == "__main__":
    main()

