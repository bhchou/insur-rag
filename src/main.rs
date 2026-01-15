mod models;

use futures::TryStreamExt;
use dotenvy::dotenv; 
use serde_json::{Value, json};
use walkdir::WalkDir;
use sha2::{Sha256, Digest};
use hex;

use std::collections::{HashMap, HashSet};
use std::env; 
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::error::Error;
use std::thread;
use std::time::{self, Duration};
use std::fs;
use std::io::{self, Write};

use models::ParsedDocument;

// LanceDB 與 Arrow 相關引入
use lancedb::{connect, query::{ExecutableQuery, QueryBase}};
use arrow_schema::{Schema, Field, DataType};
use arrow_array::{RecordBatch, RecordBatchIterator, StringArray, builder::Float32Builder, builder::FixedSizeListBuilder, Array};
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};

// --- 設定區 ---
const RAW_PDF_DIR: &str = "./data/raw_pdfs"; // 請建立此資料夾並放入您的 100 個 PDF
const PROCESSED_JSON_DIR: &str = "./data/processed_json";
const DB_URI: &str = "data/lancedb_insure";
const TABLE_NAME: &str = "insurance_docs";
const SYNONYMS_PATH: &str = "./data/synonyms.json";

#[derive(Clone)]
struct ProductSummary {
    name: String,
    intro: String, // 這裡會存：商品類型 + 特色 + 適合對象
}

// 輔助函式：計算字串的 SHA256 Hash
fn calculate_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hex::encode(hasher.finalize())
}

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
    println!("🤖 正在詢問 LLM (這可能需要幾秒鐘)...");

    // 1. 準備 Prompt (🔥 已升級：加入來源引用指令)
    // 我們告訴 LLM，如果 context 裡有檔名，盡量在回答時帶出來
    let system_prompt = "你是一個專業的保險顧問。請根據以下提供的『參考資料』(包含商品摘要與詳細片段) 回答使用者的問題。\
    \n\n重要規則：\
    \n1. 若資料中包含來源檔案名稱 (Source File)，請嘗試在回答中標註。\
    \n2. 如果資料中沒有答案，請直接說『資料不足，無法回答』，不要捏造事實。";

    let user_prompt = format!(
        "參考資料：\n{}\n\n使用者問題：{}", 
        context, query
    );

    // 2. 準備 HTTP Client (保留您的 no_proxy 設定)
    let client = reqwest::Client::builder()
        .no_proxy() // 不要管 http_proxy/HTTP_PROXY
        .build()?; 
    
    // 讀取原始的環境變數
    let vllm_endpoint = env::var("VLLM_ENDPOINT")
        .unwrap_or("http://localhost:11434".to_string());
    
    // 預設模型改為您可能使用的 (e.g., llama3, gemma2)
    let model_name = env::var("MODEL_NAME")
        .unwrap_or("llama3.1".to_string()); 
        
    let token = env::var("BEARER_TOKEN").unwrap_or_default();
    
    // 處理 URL 結尾
    let base_url = vllm_endpoint.trim_end_matches('/'); 
    
    // 自動判斷是否補上 /v1/chat/completions
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
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": user_prompt }
        ],
        "temperature": 0.1, // RAG 建議低溫，減少幻覺
        "stream": false     // 您選擇不使用串流 (適合簡單處理)
    });

    let mut request_builder = client.post(&api_url)
        .header("Content-Type", "application/json")
        .header("User-Agent", "INSUR-RAG");

    // Token 檢查邏輯
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
            println!("⚠️ LLM 回應格式無法解析 (可能無內容): {:?}", response_json);
        }
    } else {
        println!("❌ LLM 請求失敗: Status {}", res.status());
        // 嘗試印出錯誤訊息幫助除錯
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
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
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

/* for JSON and then */

// 讀取單一 JSON 檔案
fn load_policy_json(path: &Path) -> Result<models::PolicyData, Box<dyn Error>> {
    let content = fs::read_to_string(path)?;
    // 使用 serde_json 解析
    let data: models::PolicyData = serde_json::from_str(&content)?;
    Ok(data)
}

fn chunk_policy_data(data: &models::PolicyData) -> Vec<String> {
    let mut chunks = Vec::new();
    let pname = &data.basic_info.product_name;
    let fname = &data.source_filename;
    
    // 🔥🔥🔥【智慧標籤系統】🔥🔥🔥
    // 我們根據 JSON 欄位自動推導出使用者可能會搜的口語關鍵字
    let mut tags = Vec::new();

    // 1. 核心分類 (投資 vs 傳統)
    if data.investment.is_investment_linked {
        tags.push("投資型保單".to_string());
        tags.push("基金保單".to_string()); // 俗稱
        tags.push("變額保險".to_string());
        tags.push("理財型保險".to_string());
        tags.push("高風險高報酬".to_string()); // 特徵
    } else {
        tags.push("傳統型保單".to_string());
        tags.push("固定利率".to_string());
        tags.push("保證給付".to_string());
    }

    // 2. 功能需求 (存錢 vs 保障 vs 退休)
    let type_desc = &data.basic_info.product_type;
    let cov_death = &data.coverage.death_benefit;
    let cov_maturity = &data.coverage.maturity_benefit;

    // 判斷是否為「儲蓄/退休」導向
    // 如果有「滿期金」、「生存金」或類型是「年金」
    if type_desc.contains("年金") || cov_maturity.len() > 5 { 
        tags.push("退休規劃".to_string());
        tags.push("養老金".to_string());
        tags.push("儲蓄險".to_string()); // 雖然現在法規少用此詞，但民眾愛搜
        tags.push("存錢".to_string());
        tags.push("現金流".to_string());
    }

    // 判斷是否為「純保障/壽險」導向
    if type_desc.contains("壽險") || cov_death.len() > 5 {
        tags.push("壽險保障".to_string());
        tags.push("身故賠償".to_string());
        tags.push("留愛給家人".to_string()); // 行銷用語
        tags.push("資產傳承".to_string());   // 高資產族群關鍵字
    }

    // 3. 幣別特性 (美元/外幣)
    let currencies = &data.basic_info.currency;
    if currencies.iter().any(|c| c.contains("USD") || c.contains("美元")) {
        tags.push("美元保單".to_string());
        tags.push("美金保單".to_string());
        tags.push("強勢貨幣".to_string());
    }
    if currencies.iter().any(|c| c != "TWD" && c != "新台幣") {
        tags.push("外幣保單".to_string());
        tags.push("資產配置".to_string());
    }

    // 4. 繳費方式 (躉繳/期繳)
    let payment = &data.basic_info.payment_period;
    if payment.contains("躉") || payment.contains("一次") {
        tags.push("躉繳".to_string());
        tags.push("一次繳清".to_string());
        tags.push("單筆投資".to_string());
    } else {
        tags.push("期繳".to_string());
        tags.push("分期繳費".to_string());
    }

    // 5. 特殊族群 (高齡/小孩)
    let target = &data.rag_data.target_audience;
    if target.contains("65歲") || target.contains("高齡") {
        tags.push("銀髮族保單".to_string());
        tags.push("高齡投保".to_string());
    }
    if target.contains("小孩") || target.contains("子女") {
        tags.push("兒童保單".to_string());
        tags.push("教育基金".to_string());
    }

    // 將原本的關鍵字也加進來 (去重)
    for kw in &data.rag_data.keywords {
        if !tags.contains(kw) {
            tags.push(kw.clone());
        }
    }

    // 生成標籤字串，例如: "[TAGS: 投資型, 基金, 美元保單, 退休]"
    let tags_str = format!("[關鍵字: {}]", tags.join(", "));
    // 🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥

    // 修改 Header，把這些強大的標籤埋進每一個向量片段
    let header = format!("商品: {} | 來源: {} | {}", pname, fname, tags_str);

    // --- 以下 Chunk 生成邏輯保持不變，但因為 Header 變強了，所有 Chunk 都變強了 ---

    // Chunk 1: 基本資訊
    let chunk_basic = format!(
        "{} | [基本資訊] 文號: {} | 類型: {} | 繳費: {} | 幣別: {:?} | 投保年齡: {} | 保費門檻: {}",
        header,
        data.basic_info.product_code,
        data.basic_info.product_type,
        data.basic_info.payment_period,
        data.basic_info.currency,
        data.conditions.age_range,
        data.conditions.premium_limit
    );
    chunks.push(chunk_basic);

    // Chunk 2: 保障內容
    let chunk_cov = format!(
        "{} | [保障內容] 身故/喪葬給付: {} | 滿期/祝壽給付: {} | 其他權益: {:?}",
        header,
        data.coverage.death_benefit,
        data.coverage.maturity_benefit,
        data.coverage.other_benefits
    );
    chunks.push(chunk_cov);

    // Chunk 3: 投資特色
    if data.investment.is_investment_linked {
        let chunk_inv = format!(
            "{} | [投資特色] 此為投資型保單(基金/全委)。特色: {:?} | 風險: {:?}",
            header,
            data.investment.features,
            data.investment.risks
        );
        chunks.push(chunk_inv);
    }

    // Chunk 4: 費用
    let chunk_fee = format!("{} | [費用說明] {}", header, data.conditions.fees_and_discounts);
    chunks.push(chunk_fee);

    // Chunk 5: 客群
    let chunk_meta = format!("{} | [適用客群] {} | 額外標籤: {:?}", header, data.rag_data.target_audience, tags);
    chunks.push(chunk_meta);

    // Chunk 6: FAQ
    for faq in &data.rag_data.faq {
        let chunk_faq = format!("{} | [常見問題] Q: {} | A: {}", header, faq.q, faq.a);
        chunks.push(chunk_faq);
    }

    chunks
}
// 將 PolicyData 切分成帶有語意的文字片段 (Semantic Chunking)
fn chunk_policy_data_old(data: &models::PolicyData) -> Vec<String> {
    let mut chunks = Vec::new();
    let pname = &data.basic_info.product_name;
    let fname = &data.source_filename;
    
    // Helper: 產生標準化的 Context Header
    // 讓每一段文字都知道自己屬於哪個商品
    let header = format!("商品: {} | 來源: {}", pname, fname);

    // Chunk 1: 基本資訊與投保規則
    // 包含: 公司、幣別、類型、年齡、保費限制
    let chunk_basic = format!(
        "{} | [基本資訊] 文號: {} | 類型: {} | 繳費: {} | 幣別: {:?} | 投保年齡: {} | 保費門檻: {}",
        header,
        data.basic_info.product_code,
        data.basic_info.product_type,
        data.basic_info.payment_period,
        data.basic_info.currency,
        data.conditions.age_range,
        data.conditions.premium_limit
    );
    chunks.push(chunk_basic);

    // Chunk 2: 保障內容
    // 包含: 身故、滿期、其他
    let chunk_cov = format!(
        "{} | [保障內容] 身故/喪葬給付: {} | 滿期/祝壽給付: {} | 其他權益: {:?}",
        header,
        data.coverage.death_benefit,
        data.coverage.maturity_benefit,
        data.coverage.other_benefits
    );
    chunks.push(chunk_cov);

    // Chunk 3: 投資特色 (如果有)
    if data.investment.is_investment_linked {
        let chunk_inv = format!(
            "{} | [投資特色] 是否連結投資: 是 | 特色: {:?} | 風險: {:?}",
            header,
            data.investment.features,
            data.investment.risks
        );
        chunks.push(chunk_inv);
    }

    // Chunk 4: 費用與折扣
    let chunk_fee = format!(
        "{} | [費用說明] {}",
        header,
        data.conditions.fees_and_discounts
    );
    chunks.push(chunk_fee);

    // Chunk 5: 客群與關鍵字 (輔助搜尋)
    let chunk_meta = format!(
        "{} | [適用客群] {} | 關鍵字: {:?}",
        header,
        data.rag_data.target_audience,
        data.rag_data.keywords
    );
    chunks.push(chunk_meta);

    // Chunk 6~N: FAQ (黃金資料)
    // 每一題 QA 獨立成一個 Chunk，搜尋命中率極高
    for faq in &data.rag_data.faq {
        let chunk_faq = format!(
            "{} | [常見問題] Q: {} | A: {}",
            header, faq.q, faq.a
        );
        chunks.push(chunk_faq);
    }

    chunks
}

// --- 2. 處理單一檔案流程 (Embedding + DB Insert) ---
async fn process_and_index_json(
    path: &Path,
    table: &lancedb::Table,
    model: &mut TextEmbedding
) -> Result<(), Box<dyn Error>> {
    let filename = path.file_name().unwrap().to_str().unwrap();
    let content = fs::read_to_string(path)?;
    let current_hash = calculate_hash(&content);

    // 1. 讀取 JSON
    let policy = match load_policy_json(path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("❌ JSON 解析失敗 {}: {}", filename, e);
            return Ok(());
        }
    };
    // 🔥🔥🔥【DIFF 核心邏輯】🔥🔥🔥
    // 查詢 DB 中是否已有此檔案，並取出它的 file_hash
    let where_clause = format!("source_file = '{}'", policy.source_filename);
    
    let results = table
        .query()
        .only_if(where_clause)
        .limit(1) // 只要查一筆就知道有沒有
        .execute()
        .await?;

    let batches: Vec<RecordBatch> = results.try_collect().await?;
    
    if let Some(batch) = batches.first() {
        if batch.num_rows() > 0 {
            // DB 裡有這個檔，檢查 Hash 是否一樣
            if let Some(hash_col) = batch.column_by_name("file_hash") {
                 if let Some(str_array) = hash_col.as_any().downcast_ref::<StringArray>() {
                     let db_hash = str_array.value(0); // 取第一列的 Hash
                     
                     if db_hash == current_hash {
                         println!("⏩ [跳過] 檔案未變更: {}", filename);
                         return Ok(()); // Hash 一樣，完全不做事
                     } else {
                         println!("🔄 [更新] 檔案內容已變更 (Hash不同)，重新索引: {}", filename);
                         // Hash 不同，程式會繼續往下走，執行刪除舊資料+寫入新資料
                     }
                 }
            }
        }
    }
    // 🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥
    // 2. 切分
    let chunks = chunk_policy_data(&policy);
    if chunks.is_empty() { return Ok(()); }

    println!("🔄 正在索引: {} (產生 {} 個向量片段)", filename, chunks.len());

    // 3. 向量化 (Embedding)
    let embeddings = model.embed(chunks.clone(), None)?;

    // 4. 準備寫入 LanceDB 的資料
    let total_chunks = chunks.len();
    let embedding_dim = 768; // BGE-Base 的維度

    // 建構 Arrow Arrays
    let source_array = StringArray::from(vec![policy.source_filename.clone(); total_chunks]);
    let hash_array = StringArray::from(vec![current_hash; total_chunks]);
    let text_array = StringArray::from(chunks);
    
    // 建構向量 Array (Flattened list)
    let mut vector_builder = FixedSizeListBuilder::new(
        arrow_array::builder::Float32Builder::new(),
        embedding_dim as i32,
    );
    
    for vec in embeddings {
        for val in vec {
            vector_builder.values().append_value(val);
        }
        vector_builder.append(true);
    }
    let vector_array = vector_builder.finish();

    // 建立 RecordBatch
    let schema = Arc::new(Schema::new(vec![
        Field::new("source_file", DataType::Utf8, false),
        Field::new("file_hash", DataType::Utf8, false),
        Field::new("text", DataType::Utf8, false),
        Field::new("vector", DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            embedding_dim as i32
        ), false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(source_array),
            Arc::new(hash_array),
            Arc::new(text_array),
            Arc::new(vector_array),
        ],
    )?;

    // 5. 寫入 DB (先刪除舊的再寫入，確保不重複)
    // 注意：這裡我們用 source_filename 來刪除，這對應到原始 PDF/DOCX 檔名
    let delete_filter = format!("source_file = '{}'", policy.source_filename);
    table.delete(&delete_filter).await?;

    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

    
    table.add(Box::new(batches)).execute().await?;

    Ok(())
}

// --- 3. 問答邏輯 ---
async fn handle_user_query(
    db: &lancedb::Connection, 
    model: &mut TextEmbedding, 
    user_query: &str,
    synonyms: &HashMap<String, String>,
    summaries: &HashMap<String, ProductSummary>
) -> Result<(), Box<dyn Error>> {

    // 0. 字典擴充
    let mut final_query = user_query.to_string();
    for (slang, term) in synonyms {
        if user_query.contains(slang) {
            println!("💡 [字典命中] '{}' -> 加上 '{}'", slang, term);
            final_query.push_str(" ");
            final_query.push_str(term);
        }
    }

    // 1. 向量化問題
    // let query_embedding = model.embed(vec![user_query.to_string()], None)?;
    // let query_vector = query_embedding[0].clone();
    let query_vec = model.embed(vec![final_query.clone()], None)?[0].clone();

    // 2. 搜尋 DB
    let table = db.open_table(TABLE_NAME).execute().await?;
    let results = table
        .query()
        .nearest_to(query_vec)?
        .limit(10) // 取前 3 個最相關的片段
        .execute()
        .await?;
    
    let batches: Vec<RecordBatch> = results.try_collect().await?;

     // 3. 檢查結果 (簡易信心檢查: 有沒有結果)
    let has_results = !batches.is_empty() && batches[0].num_rows() > 0;

    let mut used_batches = batches;

    // 4. AI 補救 (如果沒結果)
    if !has_results {
        println!("⚠️  初步搜尋無結果，嘗試 AI 深度擴充...");
        if let Some(ai_kw) = expand_query_with_ai(user_query).await {
            let ai_vec = model.embed(vec![ai_kw], None)?[0].clone();
            let ai_results = table.query().nearest_to(ai_vec)?.limit(3).execute().await?;
            used_batches = ai_results.try_collect().await?;
        }
    }

    // 5. 組裝 Context (包含商品摘要)
    let mut hit_files = HashSet::new();
    let mut snippets_text = String::new();

    println!("\n🔍 [RAG 檢索結果]");
    for batch in &used_batches {
        let text_col = batch.column_by_name("text").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        let src_col = batch.column_by_name("source_file").unwrap().as_any().downcast_ref::<StringArray>().unwrap();

        for i in 0..batch.num_rows() {
            let src = src_col.value(i);
            let txt = text_col.value(i);
            hit_files.insert(src.to_string());
            snippets_text.push_str(&format!("📄 [片段] 來源: {}\n內容: {}\n\n", src, txt));
            // println!("   📄 來源: {} \n   📝 內容: {}\n   ---", src, text);
            
           // context_buffer.push_str(text);
           // context_buffer.push('\n');
           // if !sources.contains(&src.to_string()) {
           //     sources.push(src.to_string());
           // }
        }
    }

    /* if context_buffer.is_empty() {
        println!("⚠️  找不到相關資料。");
        return Ok(());
    } */
    if hit_files.is_empty() {
        println!("⚠️  找不到相關資料。");
        // 這裡可以考慮呼叫 AI Expansion
        return Ok(());
    }

    // 6. 注入摘要 (Summary Injection)
    let mut final_context = String::new();
    final_context.push_str("=== 相關商品基本介紹 ===\n");
    for filename in &hit_files {
        if let Some(summary) = summaries.get(filename) {
            final_context.push_str(&format!("📄 來源: {}\n{}\n", filename, summary.intro));
        }
    }
    final_context.push_str("========================\n\n");
    final_context.push_str("=== 詳細檢索片段 ===\n");
    final_context.push_str(&snippets_text);

    // 7. 最後生成
    ask_llm(&final_context, user_query).await?;
    
    println!("\n📚 [系統參考來源文件]");
    let mut sorted_files: Vec<_> = hit_files.into_iter().collect();
    sorted_files.sort(); // 排個序比較好看
    for (idx, filename) in sorted_files.iter().enumerate() {
        // 如果有摘要，順便印出商品名稱，更清楚
        if let Some(summary) = summaries.get(filename) {
            println!(" {}. {} ({})", idx + 1, summary.name, filename);
        } else {
            println!(" {}. {}", idx + 1, filename);
        }
    }
    println!("==================================");
    // println!("🤖 (LLM 會根據上述 Context 回答您的問題: '{}')", user_query);
    // println!("📚 參考文件: {:?}", sources);

    Ok(())
}

fn load_product_summaries() -> HashMap<String, ProductSummary> {
    let mut summaries = HashMap::new();
    let walker = WalkDir::new(PROCESSED_JSON_DIR).into_iter();
    
    for entry in walker.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.extension().map_or(false, |e| e == "json") {
            if let Ok(content) = fs::read_to_string(path) {
                if let Ok(data) = serde_json::from_str::<models::PolicyData>(&content) {
                    
                    // 🔥 組合出一個最強的「商品履歷」
                    let intro = format!(
                        "【商品總覽】\n名稱: {}\n類型: {}\n特色: {:?}\n適合對象: {}\n",
                        data.basic_info.product_name,
                        data.basic_info.product_type,
                        data.investment.features, // 如果是傳統型這裡可能是空，沒關係
                        data.rag_data.target_audience
                    );

                    summaries.insert(data.source_filename.clone(), ProductSummary {
                        name: data.basic_info.product_name,
                        intro,
                    });
                }
            }
        }
    }
    println!("📚 已快取 {} 筆商品摘要", summaries.len());
    summaries
}

// --- Main Workflow ---
/*#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenv().ok(); // 載入環境變數

    // 準備資料庫 (Local File)
    let uri = DB_URI;
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
        // 只處理 .pdf 和 .docx 檔案
        if path.extension().map_or(false, |ext| ext == "pdf" || ext == "docx") {
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
    let user_query = "臻美利美元利率型終身保險的主要給付項目有哪些？";
    
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
}*/

fn load_synonyms() -> HashMap<String, String> {
    if let Ok(content) = fs::read_to_string(SYNONYMS_PATH) {
        // 假設 JSON 格式是 {"mapping": {"口語": "術語"}}，這裡簡化處理直接讀 Map
        // 如果您的 Python 產出是直接的 Dict，這樣寫是對的
        if let Ok(map) = serde_json::from_str::<HashMap<String, String>>(&content) {
            println!("📚 載入離線同義詞字典: {} 筆", map.len());
            return map;
        } 
        // 如果 Python 產出包含 "mapping" key，請改用 Value 解析
    }
    println!("⚠️ 無法載入字典 ({}). 將只使用原字串搜尋。", SYNONYMS_PATH);
    HashMap::new()
}

// --- LLM API：擴充關鍵字 (Query Expansion) ---
async fn expand_query_with_ai(query: &str) -> Option<String> {
    let api_key = std::env::var("GOOGLE_API_KEY").ok()?;
    println!("🤖 [AI 介入] 正在請求 Gemini 分析意圖: '{}'...", query);
    
    let client = reqwest::Client::new();
    let prompt = format!("使用者搜尋: '{}'。請轉換為3個台灣保險專業關鍵字(如:變額萬能壽險, 月撥回)，用空白分隔，不要有其他文字。", query);

    let request_body = json!({
        "contents": [{ "parts": [{ "text": prompt }] }]
    });

    let url = format!("https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={}", api_key);

    match client.post(&url).json(&request_body).send().await {
        Ok(resp) => {
            if let Ok(json) = resp.json::<serde_json::Value>().await {
                if let Some(text) = json["candidates"][0]["content"]["parts"][0]["text"].as_str() {
                    let clean = text.trim().replace("\n", " ");
                    println!("✨ AI 建議關鍵字: {}", clean);
                    return Some(clean);
                }
            }
        }
        Err(_) => {}
    }
    None
}

// --- LLM API：最終回答 (RAG Generation) ---
async fn ask_llm_with_context(context: &str, question: &str) -> Result<(), Box<dyn Error>> {
    let api_key = std::env::var("GOOGLE_API_KEY").expect("GOOGLE_API_KEY not found");
    
    let client = reqwest::Client::new();
    let system_prompt = "你是一位專業保險顧問。請根據提供的【商品介紹】與【詳細片段】回答使用者問題。若資料不足請誠實告知。";
    let full_prompt = format!("{}\n\n參考資料:\n{}\n\n使用者問題: {}", system_prompt, context, question);

    let request_body = json!({
        "contents": [{ "parts": [{ "text": full_prompt }] }]
    });

    let url = format!("https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={}", api_key);

    println!("🤖 正在詢問 LLM (生成回答中)...");
    match client.post(&url).json(&request_body).send().await {
        Ok(resp) => {
            if let Ok(json) = resp.json::<serde_json::Value>().await {
                if let Some(text) = json["candidates"][0]["content"]["parts"][0]["text"].as_str() {
                    println!("\n💬 LLM 回答：\n==================================\n{}\n==================================", text);
                } else {
                    println!("❌ LLM 回傳格式錯誤或無內容");
                }
            } else {
                println!("❌ 無法解析 LLM 回應");
            }
        }
        Err(e) => println!("❌ API 呼叫失敗: {}", e),
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenv().ok();

    // 1. 初始化 DB
    let db = connect(DB_URI).execute().await?;
    println!("💾 連線至資料庫: {}", DB_URI);

    // 建立 Table (如果不存在)
    // 注意: 這裡定義 Schema
    let embedding_dim = 768;
    let schema = Arc::new(Schema::new(vec![
        Field::new("source_file", DataType::Utf8, false),
        Field::new("file_hash", DataType::Utf8, false), // ★ 新增這一欄
        Field::new("text", DataType::Utf8, false),
        Field::new("vector", DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            embedding_dim
        ), false),
    ]));

    /* let table = db.create_table(TABLE_NAME, RecordBatchIterator::new(vec![], schema.clone()))
        .execute_if_not_exists()
        .await?;
        */

    let table_names = db.table_names().execute().await?;
    let table = if table_names.contains(&TABLE_NAME.to_string()) {
        println!("📂 資料表 '{}' 已存在，開啟中...", TABLE_NAME);
        db.open_table(TABLE_NAME).execute().await?
    } 
    else {
        println!("✨ 資料表 '{}' 不存在，建立中...", TABLE_NAME);
        // 建立一個空的迭代器來初始化表結構
        let batches: Vec<Result<RecordBatch, arrow_schema::ArrowError>> = vec![]; 
        db.create_table(TABLE_NAME, RecordBatchIterator::new(batches, schema.clone()))
            .execute()
            .await?
    };

    // 2. 初始化 Embedding 模型
    println!("🧠 載入 Embedding 模型 (BGE-Base)...");
    let mut model = TextEmbedding::try_new(InitOptions::new(EmbeddingModel::BGEBaseENV15))?;
    let synonyms = load_synonyms();           // <--- 這裡產生 synonyms
    let summaries = load_product_summaries(); // <--- 這裡產生 summaries

    // 3. 掃描並索引 JSON
    println!("\n🚀 開始索引 JSON 資料夾: {}", PROCESSED_JSON_DIR);
    if Path::new(PROCESSED_JSON_DIR).exists() {
        let walker = WalkDir::new(PROCESSED_JSON_DIR).into_iter();
        for entry in walker.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "json") {
                // 呼叫索引函式
                if let Err(e) = process_and_index_json(path, &table, &mut model).await {
                    eprintln!("❌ 索引錯誤: {:?}", e);
                }
            }
        }
    } else {
        println!("⚠️  警告: 找不到 {} 資料夾，請確認 Python 腳本是否執行成功。", PROCESSED_JSON_DIR);
    }
    
    println!("\n✅ 所有資料索引完成！");

    // 4. 互動模式
    println!("\n🤖 保險 AI 顧問 (RAG CLI) 已就緒");
    println!("💡 輸入問題 (例如: '安聯新吉星有什麼費用?' 或 'exit' 離開)");
    
    loop {
        print!("\nUser > ");
        io::stdout().flush()?;
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_ok() {
            let q = input.trim();
            if q.eq_ignore_ascii_case("exit") { break; }
            if q.is_empty() { continue; }

            handle_user_query(&db, &mut model, q, &synonyms, &summaries).await?;
        }
    }

    Ok(())
}

