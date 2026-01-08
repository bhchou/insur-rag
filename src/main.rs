use dotenvy::{dotenv, from_path}; // 1. 引入套件
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env; // 引入標準環境庫
use std::fs;
use std::path::Path;

fn main() -> PyResult<()> {
    // --- 除錯用 Start ---
    let cwd = std::env::current_dir().unwrap();
    println!("Cargo 執行當下的路徑 (CWD): {:?}", cwd);
    println!(".env 檔案是否存在於此路徑: {}", std::path::Path::new(".env").exists());
    // --- 除錯用 End ---
    // // 程式一啟動就載入 .env
    // 這行會找當前目錄下的 .env 檔，並將其內容注入系統環境變數
    // dotenv().expect(".env file not found");
    //from_path(Path::new(".env")).expect("找不到 .env 檔案，請確認它就在 cargo run 執行的目錄下");
    match dotenv() {
        Ok(path) => println!("成功載入 .env: {:?}", path),
        Err(e) => {
            eprintln!("CRITICAL ERROR: .env 載入失敗！");
            eprintln!("錯誤原因: {:?}", e); // 這行會告訴我們真相
            std::process::exit(1);
        }
    }
    // 在 Rust 端檢查一下有沒有讀到，方便除錯
    let endpoint = env::var("VLLM_ENDPOINT").unwrap_or("未設定".to_string());
    println!("正在連接 LLM Endpoint: {}", endpoint);

    // 當 Rust 啟動 Python 時，因為上面已經執行過 dotenv()，
    // 所以 Python 裡的 os.environ["VLLM_ENDPOINT"] 也會自動有值！
    // 您不需要在 Python 裡再裝 python-dotenv。


    // 1. 設定要讀取的 PDF 路徑
    let pdf_path = "./data/sample.pdf";
    if !Path::new(pdf_path).exists() {
        println!("找不到檔案: {}, 請確認 data 目錄下有 PDF", pdf_path);
        return Ok(());
    }

    // 2. 讀取 Python script 內容
    let py_app = fs::read_to_string("pysrc/pdf_parser.py")
        .expect("無法讀取 python script");

    // 3. 啟動 Python 解譯器
    Python::with_gil(|py| {
        // 載入我們的 Python 模組
        // 這裡將 python 程式碼作為一個 module 載入，名稱取為 "parser_mod"
        let module = PyModule::from_code(py, &py_app, "pdf_parser.py", "parser_mod")?;

        // 取得 parse_pdf 函式
        let parse_func = module.getattr("parse_pdf")?;

        println!("正在使用 Python 解析 PDF: {}", pdf_path);

        // 呼叫函式，傳入參數 (Tuple 形式)
        let args = PyTuple::new(py, &[pdf_path]);
        let result: String = parse_func.call1(args)?.extract()?;

        // 4. 在 Rust 端處理結果 (JSON)
        let parsed_json: Value = serde_json::from_str(&result).unwrap();

        if let Some(error) = parsed_json.get("error") {
            println!("解析失敗: {}", error);
        } 
        else {
            // 檢查有沒有 debug_info (警告訊息)
            if let Some(debug_infos) = parsed_json["debug_info"].as_array() {
                for info in debug_infos {
                    println!("[警告] {}", info.as_str().unwrap_or(""));
                }
            }

            // 顯示頁面內容
            if let Some(pages) = parsed_json["pages"].as_array() {
                let page_count = pages.len();
                println!("解析成功！共讀取 {} 頁。", page_count);
                
                if page_count > 0 {
                    let first_page = &pages[0];
                    let method = first_page["method"].as_str().unwrap_or("unknown");
                    let content = first_page["content"].as_str().unwrap_or("");
                    
                    println!("--- 第 1 頁預覽 (使用方法: {}) ---", method);
                    // 只印出前 150 個字避免洗版
                    let preview_len = std::cmp::min(content.chars().count(), 150);
                    let preview: String = content.chars().take(preview_len).collect();
                    println!("{}...", preview);
                }
            }
        }

        Ok(())
    })
}
