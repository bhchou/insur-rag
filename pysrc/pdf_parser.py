import pdfplumber
import json
import sys
import pytesseract
import requests
import re
from PIL import Image
import os

# ==========================================
# 🔧 設定區 (請依據您的實際環境修改)
# ==========================================
#
VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT")  # 例如: http://192.168.1.100:8000
MODEL_NAME = os.getenv("MODEL_NAME")              # 例如: meta-llama/Llama-3-8b-instruct
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

def clean_json_string(text):
    """
    清理 LLM 回傳的字串，移除 Markdown 標記 (```json ... ```)
    """
    # 移除 ```json 或 ```
    cleaned = re.sub(r"```json\s*", "", text)
    cleaned = re.sub(r"```\s*", "", cleaned)
    return cleaned.strip()

def extract_metadata_via_llmYY(text_content):
    # 1. 讀取設定 (確保這裡讀得到最新的 .env)
    vllm_endpoint = os.getenv('VLLM_ENDPOINT', 'http://localhost:11434')
    bearer_token = os.getenv('BEARER_TOKEN')
    model_name = os.getenv('MODEL_NAME', 'qwen2.5:7b')

    # 2. 處理網址 (跟之前一樣的修復邏輯)
    base_url = vllm_endpoint.rstrip('/')
    if '/v1' in base_url:
        api_url = f"{base_url}/chat/completions"
    else:
        api_url = f"{base_url}/v1/chat/completions"

    # 3. 設定 Header (解決 401 的關鍵)
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "PDF-Parser"
    }
    # 只有當 Token 存在且不是 'none' 時才加 Authorization
    if bearer_token and str(bearer_token).lower() not in ['none', '', 'null']:
        headers["Authorization"] = f"Bearer {bearer_token}"
        # print(f"DEBUG: Using Token {bearer_token[:3]}...") # 除錯用

    # 4. 定義 Prompt (這裡示範將 System Prompt 融入 User Prompt)
    # 請將您原本寫在 System Role 的指令貼在 system_instruction 變數裡
    system_instruction = """
    你是一個專業的文檔分析助手。
    請分析以下文本，並提取關鍵 Metadata (如: 日期、保險公司、險種名稱)。
    請直接輸出 JSON 格式，不要包含 Markdown 標記。
    """
    
    final_prompt = f"{system_instruction}\n\n=== 待分析文本 ===\n{text_content}"

    # 5. 建構 Payload (強制使用單一 User Role，相容性最高)
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user", 
                "content": final_prompt
            }
        ],
        "temperature": 0.1,
        "stream": False
    }

    try:
        # print(f"DEBUG: Posting to {api_url}") # 除錯用
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result:
                content = result['choices'][0]['message']['content']
                # 這裡可以加一些簡單的 JSON 清理邏輯 (去掉 ```json ...)
                clean_content = content.replace('```json', '').replace('```', '').strip()
                return clean_content
            
        elif response.status_code == 401:
            print(f"❌ [PDF_PARSER] 401 權限錯誤! 請檢查 .env 的 BEARER_TOKEN")
            print(f"   連線目標: {api_url}")
            return None
            
        else:
            print(f"❌ [PDF_PARSER] API Error {response.status_code}: {response.text}")
            return None

    except Exception as e:
        print(f"❌ [PDF_PARSER] 連線例外: {e}")
        return None

def extract_metadata_via_llm(raw_text):
    """
    呼叫 VLLM API 進行 Metadata 提取
    """
    url = f"{VLLM_ENDPOINT}/v1/chat/completions"
    
    # 設置請求標頭 (參考您提供的程式碼)
    headers = {"Content-Type": "application/json"}
    if BEARER_TOKEN:
        headers["Authorization"] = f"Bearer {BEARER_TOKEN}"

    # 定義提示詞 (System Prompt + User Context)
    system_prompt = """
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
    truncated_text = raw_text[:3000]

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"這是保險文件的 OCR 內容:\n\n{truncated_text}"}
        ],
        "temperature": 0.1, # 低溫以確保輸出格式穩定
        "max_tokens": 1024
    }

    try:
        print(f"[Python] Calling LLM: {url}...", file=sys.stderr)
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            resp_json = response.json()
            content = resp_json["choices"][0]["message"]["content"]
            
            # 清理並解析 JSON
            cleaned_content = clean_json_string(content)
            print(f"[Python] LLM Response: {cleaned_content}...", file=sys.stderr) # debug log
            
            return json.loads(cleaned_content)
        else:
            print(f"[Error] VLLM API Error: {response.status_code} - {response.text}", file=sys.stderr)
            return {}

    except Exception as e:
        print(f"[Error] Connection failed: {str(e)}", file=sys.stderr)
        return {}

def parse_pdf(file_path):
    """
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
