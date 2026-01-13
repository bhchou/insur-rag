mod models;

use futures::TryStreamExt;
use dotenvy::{dotenv, from_path}; // 1. 引入套件
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env; // 引入標準環境庫
use std::fs;
use std::path::Path;

use std::process::Command;
use std::sync::Arc;
use std::error::Error;

use models::{InsuranceMetadata, ParsedDocument};

// LanceDB 與 Arrow 相關引入
use lancedb::{connect, Table, query::{ExecutableQuery, QueryBase}};
use arrow_schema::{Schema, Field, DataType};
use arrow_array::{RecordBatch, RecordBatchIterator, StringArray, builder::Float32Builder, builder::FixedSizeListBuilder, Array};
//use lancedb::arrow::arrow_schema::{Schema, Field, DataType};
//use lancedb::arrow::array::{RecordBatch, StringArray, Float32Builder, FixedSizeListBuilder, Array};
//use lancedb::arrow::arrow_array::{self, RecordBatch, StringArray, Float32Builder, FixedSizeListBuilder, Array};
// Embedding
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};

use serde_json::json;

// --- 1. Python Bridge (與 Python 溝通) ---
fn run_python_parser(pdf_path: &str) -> Result<ParsedDocument, Box<dyn Error>> {
    println!("🦀 Rust: 呼叫 Python 解析器處理 {}...", pdf_path);

    // 這裡假設您的 python script 會吐出包含 metadata 和 full_text 的 JSON
    // 如果目前的 Python 只有吐 Metadata，您可以暫時 Mock full_text，或是修改 Python 
    let output = Command::new("python3")
        .arg("pysrc/pdf_parser.py") // 請確認檔名
        .arg(pdf_path)
        .output()?;
    
    // ★★★ 新增這段：無論成功與否，都把 Python 的 Log 印出來 ★★★
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.is_empty() {
        println!("🐍 Python Debug Log:\n{}", stderr);
    }
    // ★★★ 結束新增 ★★★

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Python 執行失敗: {}", stderr).into());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // 從 stdout 中抓取 JSON 字串 (過濾掉 Log)
    let json_str = find_json_part(&stdout).ok_or("找不到有效的 JSON")?;

    println!("🦀 Rust: 收到 JSON，正在轉換為結構體...");
    
    // 這裡反序列化成您在 models.rs 定義的結構
    // 注意：如果您的 Python 目前只回傳 Metadata，這裡要稍微改一下
    // 假設 Python 回傳的是完整的 ParsedDocument (含 metadata 和 text)
    let parsed_doc: ParsedDocument = serde_json::from_str(json_str)?;

    Ok(parsed_doc)
}

// 輔助函式：抓出 JSON區塊
fn find_json_part(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if start <= end {
        Some(&text[start..=end])
    } else {
        None
    }
}

// --- 4. 向量搜尋 (Retrieval) ---
async fn search_document(
    db: &lancedb::Connection, 
    model: &mut TextEmbedding, 
    query_text: &str
) -> Result<(), Box<dyn Error>> {
    println!("\n🔍 正在搜尋: \"{}\"", query_text);

    // 1. 將查詢語句轉為向量
    // 注意：model.embed 接受 Vec<String>，所以要把單一查詢包起來
    let query_embedding = model.embed(vec![query_text.to_string()], None)?;
    let query_vector = query_embedding[0].clone(); // 拿第一筆(也是唯一一筆)

    // 2. 開啟 Table
    let table = db.open_table("insurance_docs").execute().await?;

    // 3. 執行向量搜尋 (Vector Search)
    // 搜尋最相似的 3 筆資料
    let results = table
        .query()
        .nearest_to(query_vector)? // 傳入 query 向量
        .limit(3)
        .execute()
        .await?;

    // 4. 解析並顯示結果
    // results 是一個 Stream of RecordBatch，我們把它蒐集起來
    use futures::TryStreamExt;
    let batches: Vec<RecordBatch> = results.try_collect().await?;

    println!("--------------------------------------------------");
    for batch in batches {
        let text_col = batch.column_by_name("text").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        // LanceDB 搜尋結果會自動多一個 "_distance" 欄位，代表相似度距離 (越小越相似)
        // 注意：如果您的 LanceDB 版本較舊，可能沒有回傳 distance，這邊先做個防呆
        // let dist_col = batch.column_by_name("_distance"); 

        for i in 0..batch.num_rows() {
            let content = text_col.value(i);
            // 這裡可以做字串截斷，避免印出太多
            let display_content: String = content.chars().take(100).collect();
            
            println!("📄 [結果 {}]: {}...", i + 1, display_content);
            println!("--------------------------------------------------");
        }
    }

    Ok(())
}

// --- 2. Semantic Chunking (核心邏輯：注入 Metadata) ---
fn semantic_chunking(doc: &ParsedDocument) -> Vec<String> {
    let mut chunks = Vec::new();
    let metadata = &doc.metadata;
    
    // 這裡使用簡單的句點切分，實務上可換成更聰明的 TextSplitter
    let raw_sentences: Vec<&str> = doc.full_text.split("。").collect();

    for sentence in raw_sentences {
        let clean_text = sentence.trim();
        if clean_text.is_empty() { continue; }
        
        // ★★★ 關鍵：將商品名稱與文號「焊死」在每一段文字前 ★★★
        // 這樣 Embedding 之後，這段向量就永遠帶有這些屬性
        let enriched_chunk = format!(
            "商品: {} | 文號: {} | 對象: {} | 內容: {}", // 加入對象
            metadata.product_name, 
            metadata.product_code.clone().unwrap_or_default(), // 處理 Option
            metadata.target_audience.clone().unwrap_or("不限".to_string()), // 處理 Option
            clean_text
        );
        chunks.push(enriched_chunk);
    }
    
    // 特殊處理：把 Benefit 也變成獨立的 Chunk
    for benefit in &metadata.benefits {
        let benefit_chunk = format!(
            "商品: {} | 給付項目: {}", 
            metadata.product_name, 
            benefit
        );
        chunks.push(benefit_chunk);
    }

    chunks
}

// --- 5. 生成回答 (Generation) ---
async fn ask_llm(context: &str, query: &str) -> Result<(), Box<dyn Error>> {
    println!("\n🤖 正在詢問 LLM (這可能需要幾秒鐘)...");

    // 1. 準備 Prompt
    // 這是最經典的 RAG Prompt 模板
    let system_prompt = "你是一個專業的保險顧問。請根據以下提供的『參考資料』回答使用者的問題。如果資料中沒有答案，請直接說『資料不足，無法回答』，不要捏造事實。";
    let user_prompt = format!(
        "參考資料：\n{}\n\n使用者問題：{}", 
        context, query
    );

    // 2. 準備 HTTP Client
    // let client = reqwest::Client::new();
    let client = reqwest::Client::builder()
        .no_proxy() // ★ 關鍵：告訴它不要管 http_proxy/HTTP_PROXY
        .build()?;  // 注意這裡會回傳 Result，所以要加 ?
    
    // 1. 先讀取原始的環境變數 (例如 "http://172.17.116.182:13407")
    let vllm_endpoint = std::env::var("VLLM_ENDPOINT")
        .unwrap_or("http://localhost:11434".to_string());
    let model_name = std::env::var("MODEL_NAME")
        .unwrap_or("gemma2:27b".to_string());

    // 2. 執行 Python 那段邏輯：去尾 + 判斷路徑
    let base_url = vllm_endpoint.trim_end_matches('/'); // 對應 .rstrip('/')
    
    let api_url = if base_url.contains("/v1") {
        format!("{}/chat/completions", base_url)
    } else {
        format!("{}/v1/chat/completions", base_url)
    };

    println!("🔗 連線 Endpoint: {}", api_url);
    // 3. 發送請求 (OpenAI Compatible API 格式)
    let body = json!({
        "model": model_name, // ★請確認您的 Model 名稱 (如 llama3, mistral)
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1, // RAG 建議低溫，減少幻覺
        "stream": false
    });

    let token = std::env::var("BEARER_TOKEN").unwrap_or_default();
    let mut request_builder = client.post(&api_url)
        .header("Content-Type", "application/json")
        .header("User-Agent", "INSUR-RAG");
    let token_check = token.trim().to_lowercase();
    let invalid_values = ["", "none", "null"];
    if !invalid_values.contains(&token_check.as_str()) {
        // 只有有效時才加入 Authorization
        // println!("🔐 Token 有效，已加入 Header"); // Debug 用，可拿掉
        request_builder = request_builder.header("Authorization", format!("Bearer {}", token));
    }

    let res = request_builder
        .json(&body)
        .send() // 這裡才真正發送
        .await?;

/*    let res = client.post(&api_url)
        .header("Content-Type", "application/json")
        // 如果需要 API Key (例如 OpenAI/DeepSeek)，可在此加 .bearer_auth("sk-...")
        .json(&body)
        .send()
        .await?; */

    // 4. 解析回應
    if res.status().is_success() {
        let response_json: Value = res.json().await?;
        // 抓取 choices[0].message.content
        if let Some(content) = response_json["choices"][0]["message"]["content"].as_str() {
            println!("\n💬 LLM 回答：\n==================================\n{}\n==================================", content);
        } else {
            println!("⚠️ LLM 回應格式無法解析: {:?}", response_json);
        }
    } else {
        println!("❌ LLM 請求失敗: Status {}", res.status());
        println!("Response: {}", res.text().await?);
    }

    Ok(())
}

// --- 3. Main Workflow ---
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenvy::dotenv().ok(); // 載入環境變數

    // A. 準備資料庫 (Local File)
    let uri = "data/lancedb_store";
    let db = connect(uri).execute().await?;
    println!("💾 連線至 LanceDB: {}", uri);

    // B. 準備 Embedding 模型 (BGE-M3 或 Base)
    let mut model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::BGEBaseENV15)
            .with_show_download_progress(true)
    )?;

    // C. 執行 ETL 流程
    let pdf_path = "./data/sample.pdf";
    
    // 1. Python 解析
    // 註：如果您的 models.rs 還沒定義 ParsedDocument，請先只讀 Metadata，FullText 暫時用 fake data 測試
    let doc = run_python_parser(pdf_path)?; 
    println!("✅ 解析完成: {}", doc.metadata.product_name);

    // 2. 智能切分
    let text_chunks = semantic_chunking(&doc);
    println!("🔪 切分成 {} 個語意區塊", text_chunks.len());

    if text_chunks.is_empty() {
        println!("⚠️ 沒有內容可存，結束程序。");
        return Ok(());
    }

    // 3. 向量化 (Batch Embedding)
    println!("🧠 開始向量化...");
    let embeddings = model.embed(text_chunks.clone(), None)?;

    // 4. 準備 Arrow Data (這是 LanceDB 要求的格式)
    
    let total_rows = text_chunks.len();
    let dim = 768; // BGE-Base 維度
                   //
    // 4.1 定義 Schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("text", DataType::Utf8, false),
        Field::new("vector", DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32
        ), true),
        Field::new("product_name", DataType::Utf8, false),
    ]));

    // 4.2 構建欄位數據
    //let total_rows = text_chunks.len();
    let text_array = StringArray::from(text_chunks.clone());
    let product_array = StringArray::from(vec![doc.metadata.product_name.clone(); total_rows]);
    
    // 4.3 處理向量數據 (扁平化 -> FixedSizeList)
    //let flat_vectors: Vec<f32> = embeddings.iter().flat_map(|v| v.clone()).collect();
    //let vector_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
    //    total_rows,
    //    768
    //);
    //
    // 3. 建立向量欄位 (使用 Builder，這是 Arrow 53 最穩的寫法)
    let mut list_builder = FixedSizeListBuilder::new(
        Float32Builder::with_capacity(total_rows * dim),
        dim as i32
    );

    for vector in &embeddings {
        // vector 是 Vec<f32>，直接 append slice
        list_builder.values().append_slice(vector);
        list_builder.append(true);
    }
    let vector_array = list_builder.finish();

    // 4.4 組合 Batch
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(text_array),
            Arc::new(vector_array),
            Arc::new(product_array),
        ],
    )?;

    // 4.5 Claude 說要這樣做
    let batches = RecordBatchIterator::new(
        vec![Ok(batch)],
        schema.clone(),
    );

    // 5. 寫入資料庫
    let table_name = "insurance_docs";
    /*
    let table_exists = db.table_names().execute().await?.contains(&table_name.to_string());

    if table_exists {
        let table = db.open_table(table_name).execute().await?;
        table.add(Box::new(std::iter::once(Ok(batch)))).execute().await?;
        println!("➕ 成功追加資料到現有 Table");
    } else {
        db.create_table(table_name, Box::new(std::iter::once(Ok(batch)))).execute().await?;
        println!("✨ 成功建立新 Table 並寫入資料");
    }*/
    let table_names = db.table_names().execute().await?;
    
    if table_names.contains(&table_name.to_string()) {
        let table = db.open_table(table_name).execute().await?;
        // table.add(Box::new(std::iter::once(Ok(batch)))).execute().await?;
        // CLAUDE 說要這樣做
        let add_batches = RecordBatchIterator::new(
            vec![Ok(RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(StringArray::from(text_chunks.clone())),
                    Arc::new({
                        let mut lb = FixedSizeListBuilder::new(
                            Float32Builder::with_capacity(total_rows * dim),
                            dim as i32
                        );
                        for vector in &embeddings {
                            lb.values().append_slice(vector);
                            lb.append(true);
                        }
                        lb.finish()
                    }),
                    Arc::new(StringArray::from(vec![doc.metadata.product_name.clone(); total_rows])),
                ],
            )?)],
            schema.clone(),
        );
        table.add(Box::new(add_batches)).execute().await?;
        println!("➕ 成功追加資料到現有 Table");
    } 
    else {
        // db.create_table(table_name, Box::new(std::iter::once(Ok(batch)))).execute().await?;
        // CLAUDE 說要這樣做
        db.create_table(table_name, Box::new(batches)).execute().await?;
        println!("✨ 成功建立新 Table 並寫入資料");
    }

    println!("✨ 資料庫寫入完成，稍等 1 秒確保寫入...");
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // --- 測試搜尋 ---
    // 這裡模擬使用者問問題
    let user_query = "這張保單的身故給付條件是什麼？";
    
    // 呼叫我們剛剛寫的搜尋函式
    // 注意：model 之前是 mut，這裡傳參考即可
    //search_document(&db, &mut model, user_query).await?;
    //
    // 1. 檢索 (Retrieval)
    // 為了方便，我們把 search_document 的邏輯稍微搬過來一點，或者直接在這裡搜
    // (這裡示範直接在 main 寫簡單版，避免大幅改動 search_document 簽章)
    
    println!("\n🔍 [Step 1] 正在檢索...");
    let query_embedding = model.embed(vec![user_query.to_string()], None)?;
    let table = db.open_table("insurance_docs").execute().await?;
    let results = table
        .query()
        .nearest_to(query_embedding[0].clone())?
        .limit(15)
        .execute()
        .await?;
        
    let batches: Vec<RecordBatch> = results.try_collect().await?;
    
    // 2. 組裝 Context
    let mut context_buffer = String::new();
    for batch in batches {
        let text_col = batch.column_by_name("text").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..batch.num_rows() {
            context_buffer.push_str(text_col.value(i));
            context_buffer.push('\n'); // 用換行分隔
        }
    }
    // ★★★ 加入這段 Debug Log ★★★
    println!("\n👀 [Debug] 給 LLM 的 Context 內容預覽 (前 500 字):\n--------------------------------------------------");
    println!("{}", context_buffer.chars().take(500).collect::<String>());
    println!("... (共 {} 字)", context_buffer.len());
    println!("--------------------------------------------------");

    // 3. 生成 (Generation)
    println!("\n🧠 [Step 2] 正在生成回答...");
    ask_llm(&context_buffer, user_query).await?;

    Ok(())
}

/*
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
}*/
