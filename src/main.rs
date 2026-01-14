mod models;

use futures::TryStreamExt;
use dotenvy::dotenv; 
use serde_json::{Value, json};
use walkdir::WalkDir;

use std::env; 
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::error::Error;
use std::thread;
use std::time::{self, Duration};

use models::ParsedDocument;

// LanceDB 與 Arrow 相關引入
use lancedb::{connect, query::{ExecutableQuery, QueryBase}};
use arrow_schema::{Schema, Field, DataType};
use arrow_array::{RecordBatch, StringArray, builder::Float32Builder, builder::FixedSizeListBuilder, Array};
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};

// --- 設定區 ---
const RAW_PDF_DIR: &str = "./data/raw_pdfs"; // 請建立此資料夾並放入您的 100 個 PDF
const DB_URI: &str = "data/lancedb_store";
const TABLE_NAME: &str = "insurance_docs";

// --- 1. Python Bridge (與 Python 溝通) ---
fn run_python_parser(pdf_path: &str) -> Result<ParsedDocument, Box<dyn Error>> {
    println!("🦀 Rust: 呼叫 Python 解析器處理 {}...", pdf_path);

    let output = Command::new("python3")
        .arg("pysrc/pdf_parser.py") 
        .arg(pdf_path)
        .output()?;
    
    // 無論成功與否，都把 Python 的 Log 印出來
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.is_empty() {
        println!("🐍 Python Debug Log:\n{}", stderr);
    }


    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Python 執行失敗: {}", stderr).into());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // 從 stdout 中抓取 JSON 字串 (過濾掉 Log)
    let json_str = find_json_part(&stdout).ok_or("找不到有效的 JSON")?;

    println!("🦀 Rust: 收到 JSON，正在轉換為結構體...");

    // 嘗試解析，如果失敗，就印出那串害死程式的 JSON
    let parsed_doc: ParsedDocument = match serde_json::from_str(json_str) {
        Ok(doc) => doc,
        Err(e) => {
            eprintln!("❌ JSON 解析失敗！錯誤原因: {}", e);
            eprintln!("📜 原始 JSON 內容:\n{}", json_str); // 讓兇手現形
            return Err(Box::new(e));
        }
    };

    // let parsed_doc: ParsedDocument = serde_json::from_str(json_str)?;

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

// --- 4. 向量搜尋 (Retrieval), 暫時不用 ---
async fn search_document(
    db: &lancedb::Connection, 
    model: &mut TextEmbedding, 
    query_text: &str
) -> Result<(), Box<dyn Error>> {
    println!("\n🔍 正在搜尋: \"{}\"", query_text);

    // 1. 將查詢語句轉為向量

    let query_embedding = model.embed(vec![query_text.to_string()], None)?;
    let query_vector = query_embedding[0].clone(); // 拿第一筆(也是唯一一筆)

    // 2. 開啟 Table
    let table = db.open_table("insurance_docs").execute().await?;

    // 3. 執行向量搜尋 (Vector Search)

    let results = table
        .query()
        .nearest_to(query_vector)? // 傳入 query 向量
        .limit(3)
        .execute()
        .await?;

    // 4. 解析並顯示結果

    use futures::TryStreamExt;
    let batches: Vec<RecordBatch> = results.try_collect().await?;

    println!("--------------------------------------------------");
    for batch in batches {
        let text_col = batch.column_by_name("text").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        // LanceDB 搜尋結果會自動多一個 "_distance" 欄位，代表相似度距離 (越小越相似)
        // 如果 LanceDB 版本較舊，可能沒有回傳 distance，這邊先做個防呆
        // let dist_col = batch.column_by_name("_distance"); 

        for i in 0..batch.num_rows() {
            let content = text_col.value(i);
            // 這裡做字串截斷，避免印出太多
            let display_content: String = content.chars().take(100).collect();
            
            println!("📄 [結果 {}]: {}...", i + 1, display_content);
            println!("--------------------------------------------------");
        }
    }

    Ok(())
}

// Semantic Chunking (核心邏輯：注入 Metadata) ---
fn semantic_chunking(doc: &ParsedDocument, filename: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let metadata = &doc.metadata;
    
    // 先用簡單的句點切分
    let raw_sentences: Vec<&str> = doc.full_text.split("。").collect();

    for sentence in raw_sentences {
        let clean_text = sentence.trim();
        if clean_text.is_empty() { continue; }
        
        // 將商品名稱與文號「焊死」在每一段文字前
        // 這樣 Embedding 之後，這段向量就永遠帶有這些屬性
        let enriched_chunk = format!(
            "來源: {} | 商品: {} | 文號: {} | 對象: {} | 內容: {}", // 加入對象
            filename,
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
    let system_prompt = "你是一個專業的保險顧問。請根據以下提供的『參考資料』回答使用者的問題。如果資料中沒有答案，請直接說『資料不足，無法回答』，不要捏造事實。";
    let user_prompt = format!(
        "參考資料：\n{}\n\n使用者問題：{}", 
        context, query
    );

    // 2. 準備 HTTP Client

    let client = reqwest::Client::builder()
        .no_proxy() // 不要管 http_proxy/HTTP_PROXY
        .build()?; 
    
    // 讀取原始的環境變數
    let vllm_endpoint = env::var("VLLM_ENDPOINT")
        .unwrap_or("http://localhost:11434".to_string());
    let model_name = env::var("MODEL_NAME")
        .unwrap_or("gemma2:27b".to_string());
    let token = env::var("BEARER_TOKEN").unwrap_or_default();
    

    let base_url = vllm_endpoint.trim_end_matches('/'); // 對應 .rstrip('/')
    
    let api_url = if base_url.contains("/v1") {
        format!("{}/chat/completions", base_url)
    } else {
        format!("{}/v1/chat/completions", base_url)
    };

    println!("🔗 連線 Endpoint: {}", api_url);
    
    // 發送請求 (OpenAI Compatible API 格式)
    let body = json!({
        "model": model_name, 
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1, // RAG 建議低溫，減少幻覺
        "stream": false
    });

    let mut request_builder = client.post(&api_url)
        .header("Content-Type", "application/json")
        .header("User-Agent", "INSUR-RAG");
    let token_check = token.trim().to_lowercase();
    let invalid_values = ["", "none", "null"];
    if !invalid_values.contains(&token_check.as_str()) {
        request_builder = request_builder.header("Authorization", format!("Bearer {}", token));
    }

    let res = request_builder
        .json(&body)
        .send() 
        .await?;

    // 解析回應
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

// --- 單檔處理核心邏輯 (Core Logic) ---
async fn process_single_file(
    path: &Path, 
    db: &lancedb::Connection, 
    model: &mut TextEmbedding
) -> Result<(), Box<dyn Error>> {
    let filename = path.file_name().unwrap().to_str().unwrap();
    let path_str = path.to_str().unwrap();

    println!("--------------------------------------------------");
    println!("🚀 開始處理: {}", filename);

    // [Check Idempotency] 檢查是否已處理過
    // 簡單查詢 DB 有沒有這個 filename
    if let Ok(table) = db.open_table(TABLE_NAME).execute().await {
        // 使用 SQL style filter
        let filter = format!("source_file = '{}'", filename);
        let count = table.count_rows(Some(filter)).await?;
        if count > 0 {
            println!("⏩ 檔案已存在 ({} 筆紀錄)，跳過處理: {}", count, filename);
            return Ok(());
        }
    }

    // Python 解析
    let doc = match run_python_parser(path_str) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("❌ 解析失敗 [{}]: {}", filename, e);
            return Ok(()); // 回傳 Ok 讓迴圈繼續
        }
    };

    // 切分
    let chunks = semantic_chunking(&doc, filename);
    if chunks.is_empty() {
        println!("⚠️  檔案內容為空，跳過。");
        return Ok(());
    }

    // Embedding (改分批處理以節省記憶體)
    println!("🧠 向量化 {} 個片段...", chunks.len());
    let batch_size = 30; 
    let mut embeddings = Vec::with_capacity(chunks.len());
    // 使用 chunks() 進行切分
    for (_i, batch) in chunks.chunks(batch_size).enumerate() {
        // 轉成 Vec<String> 傳給 model
        let batch_vec = batch.to_vec();
        
        // 執行向量化
        let batch_embeddings = model.embed(batch_vec, None)?;
        embeddings.extend(batch_embeddings);

        // 每一批處理完稍微休息一下，讓 CPU 降溫
        thread::sleep(time::Duration::from_millis(50)); 
        
        // print!(".") 來顯示進度，flush stdout 確保看得到
        use std::io::{self, Write};
        print!(".");
        io::stdout().flush().unwrap();
    }
    println!("\n✅ 向量化完成");

    // 準備 Arrow Batch
    let total_rows = chunks.len();
    let dim = 768;

    let schema = Arc::new(Schema::new(vec![
        Field::new("text", DataType::Utf8, false),
        Field::new("vector", DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32
        ), true),
        Field::new("product_name", DataType::Utf8, false),
        Field::new("source_file", DataType::Utf8, false), // 新增欄位
    ]));

    let text_array = StringArray::from(chunks.clone());
    let product_array = StringArray::from(vec![doc.metadata.product_name.clone(); total_rows]);
    let source_array = StringArray::from(vec![filename.to_string(); total_rows]); 

    let mut list_builder = FixedSizeListBuilder::new(Float32Builder::with_capacity(total_rows * dim), dim as i32);
    for vector in &embeddings {
        list_builder.values().append_slice(vector);
        list_builder.append(true);
    }
    let vector_array = list_builder.finish();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(text_array),
            Arc::new(vector_array),
            Arc::new(product_array),
            Arc::new(source_array),
        ],
    )?;

    // 寫入 DB (Append 模式)
    let table_names = db.table_names().execute().await?;
    if table_names.contains(&TABLE_NAME.to_string()) {
        let table = db.open_table(TABLE_NAME).execute().await?;
        // 這裡需要用 iterator 包起來
        let batches = arrow_array::RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        table.add(Box::new(batches)).execute().await?;
    } 
    else {
        // 第一次建立
        let batches = arrow_array::RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        db.create_table(TABLE_NAME, Box::new(batches)).execute().await?;
    }

    println!("✅ 完成: {}", filename);
    Ok(())
}

// --- Main Workflow ---
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenv().ok(); // 載入環境變數

    // 準備資料庫 (Local File)
    let uri = "data/lancedb_store";
    let db = connect(uri).execute().await?;
    println!("💾 連線至 LanceDB: {}", uri);

    // 準備 Embedding 模型 (BGE-M3 或 Base)
    let mut model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::BGEBaseENV15)
            .with_show_download_progress(true)
    )?;
    // 建立原始檔案目錄 (如果不存在)
    if !Path::new(RAW_PDF_DIR).exists() {
        std::fs::create_dir_all(RAW_PDF_DIR)?;
        println!("⚠️ 請將 PDF 檔案放入 {} 資料夾中", RAW_PDF_DIR);
    }

    // 掃描目錄
    println!("🔍 掃描目錄: {} ...", RAW_PDF_DIR);
    let walker = WalkDir::new(RAW_PDF_DIR).into_iter();

    for entry in walker.filter_map(|e| e.ok()) {
        let path = entry.path();
        // 只處理 .pdf 檔案
        if path.extension().map_or(false, |ext| ext == "pdf") {
            // 呼叫處理函式
            if let Err(e) = process_single_file(path, &db, &mut model).await {
                eprintln!("💥 嚴重錯誤 (Skipped): {:?}", e);
            }

            // 處理完一個檔案，休息 200 毫秒 
            // 讓 OS 有機會進行 I/O Flush 和記憶體回收
            thread::sleep(Duration::from_millis(200));
        }
    }

    println!("\n🎉 所有檔案處理完成！");

    println!("✨ 資料庫寫入完成，稍等 1 秒確保寫入...");
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // --- 測試搜尋 ---
    // 這裡模擬使用者問問題
    let user_query = "有哪些保單是針對30歲以上女性設計的終身壽險？";
    
    // 呼叫我們剛剛寫的搜尋函式
    //search_document(&db, &mut model, user_query).await?;
    //
    // 為了方便，我們把 search_document 的邏輯搬過來直接在這裡搜
    
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
    
    // 組裝 Context
    let mut context_buffer = String::new();
    for batch in batches {
        let text_col = batch.column_by_name("text").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..batch.num_rows() {
            context_buffer.push_str(text_col.value(i));
            context_buffer.push('\n'); // 用換行分隔
        }
    }
    // Debug Log
    println!("\n👀 [Debug] 給 LLM 的 Context 內容預覽 (前 500 字):\n--------------------------------------------------");
    println!("{}", context_buffer.chars().take(500).collect::<String>());
    println!("... (共 {} 字)", context_buffer.len());
    println!("--------------------------------------------------");

    // 生成 (Generation)
    println!("\n🧠 [Step 2] 正在生成回答...");
    ask_llm(&context_buffer, user_query).await?; 

    Ok(())
}
