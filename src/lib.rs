pub mod models;

use futures::TryStreamExt;
use dotenvy::dotenv; 
use serde_json::{Value, json};
use walkdir::WalkDir;
use serde::{Serialize, Deserialize};
use regex::Regex;

use std::collections::{HashMap, HashSet};
use std::env; 
use std::sync::Arc;
use std::error::Error;
use std::fs;
use tokio::sync::Mutex;
use std::path::PathBuf;

use sha2::{Sha256, Digest};

// use redis::Client;
use deadpool_redis::{Config, Runtime, Pool};

// LanceDB 與 Arrow 相關引入
use lancedb::{connect, query::{ExecutableQuery, QueryBase, Select}};
use arrow_schema::{Schema, Field, DataType};
use arrow_array::{RecordBatch, RecordBatchIterator, StringArray, Array, Float32Array, FixedSizeListArray};
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};

// --- 設定區 ---
const PROCESSED_JSON_DIR: &str = "./data/processed_json";
const DB_URI: &str = "data/lancedb_insure";
const TABLE_NAME: &str = "insurance_docs";

#[derive(Clone)]
pub struct ProductSummary {
    pub name: String,
    pub intro: String, // 這裡會存：商品類型 + 特色 + 適合對象
}

// --- Rerank API 結構 ---
#[derive(Serialize)]
struct RerankRequest {
    query: String,
    documents: Vec<String>,
}

#[derive(Deserialize)]
struct RerankResponse {
    scores: Vec<f32>,
    indices: Vec<usize>,
}

pub struct AppState {
    pub db: lancedb::Connection,
    pub model: Mutex<TextEmbedding>, // 注意：Model 不是線程安全的，要加 Mutex
    pub synonyms: HashMap<String, String>,
    pub summaries: HashMap<String, ProductSummary>,
    pub llm_provider: String,
    pub google_api_key: String,
    pub local_llm_url: String,
    pub local_llm_model: String,
   // pub redis_client: Option<Client>,
    pub redis_pool: Option<Pool>,
}

#[derive(Serialize, Debug)]
pub struct RagResponse {
    pub answer: String,
    pub sources: Vec<String>,
}

fn calculate_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content);
    hex::encode(hasher.finalize())
}

fn load_system_prompt() -> String {
    // 1. 嘗試從 env 讀取路徑
    let path = env::var("SYSTEM_PROMPT_PATH").unwrap_or("./data/system_prompt.txt".to_string());
    
    // 2. 讀取檔案內容
    match fs::read_to_string(path.clone()) {
        Ok(content) => {
            println!("📜 已載入 System Prompt ({} bytes)", content.len());
            content
        },
        Err(e) => {
            println!("⚠️ 無法讀取 Prompt 檔案 ({})，使用內建預設值。錯誤: {}", path, e);
            // 這裡放一個最簡單的預設值當作備案
            "你是一個專業的保險顧問。請根據參考資料回答問題。".to_string()
        }
    }
}

// --- 5. 生成回答 (Generation) ---
async fn ask_llm(state: &Arc<AppState>, context: &str, query: &str) -> Result<String, Box<dyn Error>> {
    match state.llm_provider.as_str() {
        "local" => ask_local_llm(state, context, query).await,
        "google" => ask_google_gemini(state, context, query).await,
        _ => {
            println!("⚠️ 未知 Provider: {}，預設使用 Google", state.llm_provider);
            ask_google_gemini(state, context, query).await
        }
    }
}

async fn ask_local_llm(state: &Arc<AppState>, context: &str, query: &str) -> Result<String, Box<dyn Error>> {
    let system_prompt_text = load_system_prompt();
    println!("🤖 正在詢問 LLM (這可能需要幾秒鐘)...");


    let user_prompt = format!(
        "參考資料：\n{}\n\n使用者問題：{}", 
        context, query
    );

    // 2. 準備 HTTP Client (保留您的 no_proxy 設定)
    let client = reqwest::Client::builder()
        .no_proxy() // 不要管 http_proxy/HTTP_PROXY
        .build()?; 
    
    let token = env::var("BEARER_TOKEN").unwrap_or_default();
    
    let base_url = state.local_llm_url.trim_end_matches('/');     
    let api_url = if base_url.contains("/v1") {
        format!("{}/chat/completions", base_url)
    } 
    else {
        format!("{}/v1/chat/completions", base_url)
    };

    println!("🔗 連線 Endpoint: {}", api_url);
    
    // 發送請求 (OpenAI Compatible API 格式)
    let body = json!({
        "model": state.local_llm_model, 
        "messages": [
            { "role": "system", "content": system_prompt_text },
            { "role": "user", "content": user_prompt }
        ],
        "temperature": 0.1, 
        "stream": false     
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
            // println!("\n💬 LLM 回答：\n==================================\n{}\n==================================", content);
            return Ok(content.to_string())
        } 
        else {
            return Err(format!("LLM 回應格式錯誤，無法找到回答內容: {:?}", response_json).into());
        }
    } 
    else {
        return Err(format!("❌ LLM 請求失敗: Status {}\nResponse: {}", res.status(), res.text().await?).into());

    }

}

// --- LLM API：最終回答 (RAG Generation) 這部分退休後用 ---
async fn ask_google_gemini(state: &Arc<AppState>, context: &str, query: &str) -> Result<String, Box<dyn Error>> {
    // 檢查有沒有 Key
    if state.google_api_key.is_empty() {
        return Err("缺少 GOOGLE_API_KEY".into());
    }    
    let system_prompt_text = load_system_prompt();
    let client = reqwest::Client::new();
    let full_prompt = format!("{}\n\n參考資料:\n{}\n\n使用者問題: {}", system_prompt_text, context, query);

    let request_body = json!({
        "contents": [{ "parts": [{ "text": full_prompt }] }]
    });

    let url = format!("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={}",
                    state.google_api_key);

    match client.post(&url).json(&request_body).send().await {
        Ok(resp) => {
            if let Ok(json) = resp.json::<serde_json::Value>().await {
                if let Some(text) = json["candidates"][0]["content"]["parts"][0]["text"].as_str() {
                    return Ok(text.to_string());
                } 
                else {
                    return Err("❌ LLM 回傳格式錯誤或無內容".into());
                }
            } else {
                return Err("❌ 無法解析 LLM 回應".into());
            }
        }
        Err(e) => return Err(format!("❌ API 呼叫失敗: {}", e).into())
    }
}

/* for JSON and then */

// --- 3. 問答邏輯 ---
pub async fn process_query(
    state: &Arc<AppState>,
    history: &[Value],
    user_query: &str,
) -> Result<RagResponse, Box<dyn Error>> {
    
    let mut model = state.model.lock().await; 
    let db = &state.db;
    let synonyms = &state.synonyms;
    let summaries = &state.summaries;

    // --- 讀取環境變數 (設定預設值以防沒設) ---
    let recall_limit = env::var("RAG_RECALL_LIMIT").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let rerank_limit = env::var("RAG_RERANK_LIMIT").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let rerank_api = env::var("RERANK_API_URL").unwrap_or("http://localhost:8000/rerank".to_string());
    // -------------------------------------
    // 在 process_query 一開始
    let mut normalized_query = user_query.to_string();

    // 1. 強制將數字與中文之間插入空白
    // 把 "30歲" 變成 "30 歲"，把 "100萬" 變成 "100 萬"
    let re_num_zh = Regex::new(r"(\d+)([\u4e00-\u9fa5])").unwrap();
    normalized_query = re_num_zh.replace_all(&normalized_query, "$1 $2").to_string();

    let re_zh_num = Regex::new(r"([\u4e00-\u9fa5])(\d+)").unwrap();
    normalized_query = re_zh_num.replace_all(&normalized_query, "$1 $2").to_string();

    println!("🔧 正規化查詢: '{}' -> '{}'", user_query, normalized_query);
    let mut search_target = normalized_query.clone();

    // 0. 字典擴充
    // let mut final_query = user_query.to_string();
    for (slang, term) in synonyms {
        if user_query.contains(slang) {
            println!("💡 [字典命中] '{}' -> 加上 '{}'", slang, term);
            search_target.push_str(" ");
            search_target.push_str(term);
        }
    }

    let should_rewrite = history.len() > 1 && user_query.chars().count() < 50;
    if should_rewrite {
        println!("🤔 偵測到短問題且有歷史，嘗試進行「主動意圖改寫」...");
        if let Some(rewritten) = expand_query_with_ai(state, history, user_query).await {
            println!("✅ AI 改寫成功: '{}'", rewritten);
            let mut final_rewritten = rewritten.clone();
            
            if user_query.len() > 6 && !final_rewritten.contains(user_query) {
                println!("⚠️ [防呆觸發] AI 改寫遺失使用者關鍵意圖，強制補回！");
                final_rewritten.push_str(" ");
                final_rewritten.push_str(user_query);
            }

            search_target = final_rewritten;
            println!("✅ 最終搜尋目標: '{}'", search_target);
        }
    } 
    else {
        println!("ℹ️ 無需 AI 改寫 (無歷史或問題夠完整)，使用原始查詢");
    }

    let forced_candidates: Vec<(String, String, f32)> = Vec::new();
    let mut forced_filenames = HashSet::new();
    let mut search_filter: Option<String> = None;

    let re = Regex::new(r#"[『「《【“"‘'（\(](.*?)[」』》】”"’'）\)]"#).unwrap();
    
    for cap in re.captures_iter(user_query) {
        let keyword = &cap[1]; // 提取到的關鍵字，例如 "活利優退"
        println!("🎯 偵測到明確意圖關鍵字: {}", keyword);

        // 2. 掃描 Summary 找對應檔案
        for (filename, summary) in &state.summaries {
            // 規則：只要檔名或商品全名包含這個關鍵字 -> 命中
            if filename.contains(keyword) || summary.name.contains(keyword) {
                println!("✅ 鎖定檔案: {}", filename);
                forced_filenames.insert(filename.clone());
            }
        }
    }
    // 3. 如果有鎖定的檔案，直接去 DB 撈出來 (不透過向量搜尋)
    if !forced_filenames.is_empty() {
        // 組裝 SQL Filter: source_file = 'A' OR source_file = 'B'
        let filter_cond = forced_filenames
            .iter()
            .map(|f| format!("source_file = '{}'", f))
            .collect::<Vec<_>>()
            .join(" OR ");

        search_filter = Some(filter_cond.clone());
    }

    println!("🔍 執行向量搜尋: {}", search_target);

    let mut vector_batches = search_in_lancedb(&mut *model, &db, &search_target, recall_limit, search_filter.clone()).await?;

    if vector_batches.is_empty() && search_target != user_query {
        println!("⚠️ [Fallback Triggered] 精準搜尋無結果 ('{}')，嘗試使用原始問題重搜...", search_target);

        vector_batches = search_in_lancedb(&mut *model, &db, user_query, recall_limit, search_filter).await?;

        search_target = user_query.to_string();
    }


    let mut raw_candidates: Vec<(String, String)> = Vec::new();
    let mut seen_texts = HashSet::new();


    for (src, txt, _) in forced_candidates {
        if seen_texts.insert(txt.clone()) {
            raw_candidates.push((src, txt));
        }
    }


    for b in vector_batches {
        let src_col = b.column_by_name("source_file").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        let txt_col = b.column_by_name("text").unwrap().as_any().downcast_ref::<StringArray>().unwrap();

        for i in 0..b.num_rows() {
            let txt = txt_col.value(i).to_string();
            if seen_texts.insert(txt.clone()) {
                raw_candidates.push((
                    src_col.value(i).to_string(),
                    txt
                ));
            }
        }
    }


    if raw_candidates.is_empty() {
        return Ok(RagResponse {
            answer: "抱歉，資料庫中找不到相關資訊，請嘗試其他關鍵字。".to_string(),
            sources: vec![],
        });
    }
    // 🔥 [非對稱過濾策略]
    // 定義需要「嚴格過濾」的險種。壽險、意外險因為太廣泛，故意不列入，保持寬鬆。
    let strict_rules = vec![
        ("醫療", vec!["醫療", "手術", "住院", "實支實付", "健康保險"]),
        ("癌症", vec!["癌症", "防癌", "惡性腫瘤", "化療", "標靶"]),
        ("長照", vec!["長照", "長期照顧", "失能", "扶助"]),
        ("打工", vec!["打工", "遊學", "度假", "海外"]),
        ("投資", vec!["投資", "基金", "變額", "收益"]),
    ];

    let protected_rules = vec![
        ("壽險", vec!["壽險", "身故", "人壽", "儲蓄", "還本"]),
        ("意外", vec!["意外", "傷害", "骨折", "產險"]),
    ];

    let mut allowed_keywords: Vec<&str> = Vec::new();
    let mut strict_mode_triggered = false;


    for (category, keywords) in &strict_rules {
        if user_query.contains(category) {
            println!("🎯 偵測到嚴格類別意圖: [{}]", category);
            allowed_keywords.extend(keywords.iter().cloned());
            strict_mode_triggered = true;
        }
    }

    if strict_mode_triggered {
        for (category, keywords) in &protected_rules {
            if user_query.contains(category) {
                println!("🛡️ 偵測到混合意圖，加入受保護類別: [{}]", category);
                allowed_keywords.extend(keywords.iter().cloned());
            }
        }
    }

    if !allowed_keywords.is_empty() {
        let before_count = raw_candidates.len();
        
        raw_candidates.retain(|(src, txt)| {
            // 規則：(A OR B OR C...) 只要命中其中一組關鍵字即可保留
            let src_match = allowed_keywords.iter().any(|&k| src.contains(k));
            let txt_match = allowed_keywords.iter().any(|&k| txt.chars().take(200).collect::<String>().contains(k));
            
            src_match || txt_match
        });

        println!("🧹 混合過濾執行: {} -> {} 筆 (關鍵字聯集: {:?})", 
            before_count, raw_candidates.len(), allowed_keywords);


        if raw_candidates.is_empty() {
             println!("⚠️ 過濾後無結果，取消過濾條件。");

        }
    }

    let top_results_all = rerank_documents(&search_target, raw_candidates, summaries, recall_limit, &rerank_api).await?;
    let top_results: Vec<(String, String, f32)> = top_results_all.into_iter().take(rerank_limit).collect();

    if top_results.is_empty() {
         return Ok(RagResponse {
            answer: "雖然有相關文檔，但經過相關性檢測後被過濾掉了。".to_string(),
            sources: vec![],
        });
    }

    // 5. 組裝 Context (包含商品摘要)
    let mut hit_files = HashSet::new();
    let mut snippets_text = String::new();

    println!("\n🔍 [RAG 檢索結果]");
   
    for (src, txt, score) in &top_results {
        hit_files.insert(src.clone());
        // 我們可以在 context 裡稍微標註一下這是精選出來的
        snippets_text.push_str(&format!("📄 [精選片段] (關聯度:{:.1}) 來源: {}\n內容: {}\n\n", score, src, txt));
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


    let llm_answer = ask_llm(state, &final_context, &search_target).await?;
    

    let mut sorted_sources: Vec<String> = hit_files.into_iter().collect();
    sorted_sources.sort();

    Ok(RagResponse {
        answer: llm_answer,
        sources: sorted_sources,
    })
}

pub async fn expand_query_with_ai(state: &Arc<AppState>, history: &[Value], query: &str) -> Option<String> {
    // 建立指代消解專用的 System Prompt
    let system_prompt = r#"
    你是一個 RAG 搜尋意圖優化專家。你的任務是結合「對話歷史」與「最新問題」，產出最精準的搜尋關鍵字。

    【核心規則】：
    1. **繼承人設 (最重要)**：永遠保留歷史中的「年齡」、「性別」、「職業」或「家庭狀況」等資訊。(例如：30歲男性、營造業)。
    2. **意圖切換 (Negative Check)**：
       - 如果最新問題包含「不要...」、「改看...」、「不是...」等否定詞。
       - **必須移除** 歷史中被否定的關鍵字 (例如：使用者說「不要投資型」，你就要把「投資、變額」拿掉，改加入「純壽險、傳統型」)。
       - **解除鎖定**：不要再加入上一輪推薦的具體產品名稱。
    3. **產品鎖定**：只有在使用者「追問」細節 (如：那費用呢？) 時，才鎖定上一輪的產品名稱。

    【合成範例】：
    History: 30歲男性, 推薦投資型 -> AI推薦富邦投資
    Current: "那如果不要投資，純粹壽險呢？"
    Result: "30歲男性 終身壽險 定期壽險 (排除投資型)"  <-- (關鍵：保留年齡，但切換險種)

    History: 50歲女性 -> AI推薦防癌險
    Current: "費用多少"
    Result: "50歲女性 防癌險 費用費率"

    請直接輸出優化後的搜尋字串。
    "#;
    
    let history_text = history.iter()
        .rev() // 從新到舊
        .take(4)
        .rev() // 轉回來
        .map(|v| format!("{}: {}", v["role"].as_str().unwrap_or("unknown"), v["content"].as_str().unwrap_or("")))
        .collect::<Vec<String>>()
        .join("\n");

    let full_context = format!("對話歷史:\n{}\n\n使用者最新問題: {}", history_text, query);

    println!("🤖 [AI 改寫] 正在分析意圖...");

    let result = match state.llm_provider.as_str() {
        "local" => expand_local(state, system_prompt, &full_context).await,
        "google" => expand_google(state, system_prompt, &full_context).await,
        _ => expand_google(state, system_prompt, &full_context).await, // 預設 Google
    };

    match result {
        Ok(rewritten) => {
            let clean = rewritten.trim().replace("\n", " ");
            println!("✨ 原始問題: {}", query);
            println!("✨ 改寫後問題: {}", clean);
            Some(clean)
        },
        Err(e) => {
            eprintln!("❌ 意圖改寫失敗，將使用原始問題: {}", e);
            None // 失敗回傳 None，外層邏輯會自動退回使用原始 query
        }
    }
}


async fn expand_local(state: &Arc<AppState>, system_prompt: &str, user_content: &str) -> Result<String, Box<dyn Error>> {

    let client = reqwest::Client::builder()
        .no_proxy()
        .build()?;

    let base_url = state.local_llm_url.trim_end_matches('/');
    let api_url = if base_url.contains("/v1") {
        format!("{}/chat/completions", base_url)
    } else {
        format!("{}/v1/chat/completions", base_url)
    };

    let token = std::env::var("BEARER_TOKEN").unwrap_or_default();

    let body = json!({
        "model": state.local_llm_model,
        "messages": [
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": user_content } 
        ],
        "temperature": 0.1, 
        "max_tokens": 1024   
    });

    let mut request_builder = client.post(&api_url)
        .header("Content-Type", "application/json");

    if !token.is_empty() && token != "none" {
        request_builder = request_builder.header("Authorization", format!("Bearer {}", token));
    }

    let resp = request_builder.json(&body).send().await?;
    let resp_status = resp.status();

    if resp.status().is_success() {
        let json: Value = resp.json().await?;
        if let Some(content) = json["choices"][0]["message"]["content"].as_str() {
            return Ok(content.to_string());
        }
    }
    
    Err(format!("Local LLM 回應錯誤: {}", resp_status).into())
}


async fn expand_google(state: &Arc<AppState>, system_prompt: &str, user_content: &str) -> Result<String, Box<dyn Error>> {
    if state.google_api_key.is_empty() {
        return Err("缺少 GOOGLE_API_KEY".into());
    }

   
    let client = reqwest::Client::new();

    
    let full_prompt = format!("{}\n\n{}", system_prompt, user_content);

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={}",
        state.google_api_key
    );

    let body = json!({
        "contents": [{ "parts": [{ "text": full_prompt }] }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 1024
        }
    });

    let resp = client.post(&url).json(&body).send().await?;

    let resp_status = resp.status();

    if resp.status().is_success() {
        let json: Value = resp.json().await?;
        if let Some(text) = json["candidates"][0]["content"]["parts"][0]["text"].as_str() {
            return Ok(text.to_string());
        }
    }

    Err(format!("Google API 回應錯誤: {}", resp_status).into())
}


async fn rerank_documents(
    query: &str,
    candidates: Vec<(String, String)>, // (source_file, text)
    summaries: &HashMap<String, ProductSummary>,
    top_k: usize,
    api_url: &str
) -> Result<Vec<(String, String, f32)>, Box<dyn Error>> {

    let max_chunks_per_doc = env::var("MAX_CHUNKS_PER_DOC")
        .unwrap_or("3".to_string())
        .parse::<usize>()
        .unwrap_or(3);
    
    if candidates.is_empty() {
        return Ok(Vec::new());
    }


    let mut doc_texts_for_api: Vec<String> = Vec::new();

    for (src, txt) in &candidates {

        let content_for_judge = if let Some(sum) = summaries.get(src) {
            format!("{}\n文件內容: {}", sum.intro, txt)
        } else {
            txt.clone()
        };
        doc_texts_for_api.push(content_for_judge);
    }


    let client = reqwest::Client::builder()
        .no_proxy()
        .build()?;
    let request_body = RerankRequest {
        query: query.to_string(),
        documents: doc_texts_for_api,
    };

    println!("⚖️ 正在進行 Re-ranking ({} 筆候選, 取 Top {} 到 {})...", candidates.len(), top_k, api_url);

    let rerank_response_result = client.post(api_url)
        .json(&request_body)
        .send()
        .await;

    // 2. 判斷連線結果
    let rerank_res: RerankResponse = match rerank_response_result {
        Ok(resp) if resp.status().is_success() => {

            match resp.json::<RerankResponse>().await {
                Ok(res) => res, 
                Err(e) => {
                    println!("⚠️ [非 Demo 時間] Rerank JSON 解析失敗: {}", e);
                    
                    RerankResponse { indices: vec![], scores: vec![] } 
                }
            }
        },
        Ok(resp) => {
            
            println!("⚠️ [非 Demo 時間] Rerank Server 回傳錯誤代碼: {}", resp.status());
            RerankResponse { indices: vec![], scores: vec![] }
        },
        Err(e) => {
            
            println!("⚠️ [非 Demo 時間] 無法連線至 Rerank Server: {}", e);
            RerankResponse { indices: vec![], scores: vec![] }
        }
    };


    let mut ranked_results = Vec::new();
    let mut file_counts: HashMap<String, usize> = HashMap::new();
    

    if !rerank_res.indices.is_empty() {

        println!("✅ Rerank 成功，使用 AI 重排序結果...");
        for (i, &original_idx) in rerank_res.indices.iter().enumerate() {
            if ranked_results.len() >= top_k { break; }
            
            let score = rerank_res.scores[i];
            

            if score < -5.0 { continue; }

            if let Some((src, txt)) = candidates.get(original_idx) {
                let count = file_counts.entry(src.clone()).or_insert(0);
                if *count < max_chunks_per_doc {
                    println!("   ⭐ [Top {}] 分數: {:.2} | 來源: {}", i+1, score, src);
                    ranked_results.push((src.clone(), txt.clone(), score));
                    *count += 1;
                }
            }
        }
    } 
    else {
        
        println!("🛌 Rerank 休息中，直接回傳 LanceDB 原始排序...");
        
        
        for (i, (src, txt)) in candidates.iter().enumerate() {
            if ranked_results.len() >= top_k { break; }

            
            let count = file_counts.entry(src.clone()).or_insert(0);
            
            if *count < max_chunks_per_doc {
                
                let fake_score = 0.0; 
                println!("   📦 [原始結果 {}] 來源: {}", i+1, src);
                ranked_results.push((src.clone(), txt.clone(), fake_score));
                *count += 1;
            }
        }
    }

    Ok(ranked_results)
}


pub async fn init_system() -> Result<Arc<AppState>, Box<dyn Error>> {
    dotenv().ok();
    
    let db_path = std::env::var("LANCEDB_PATH").unwrap_or(DB_URI.to_string());
    println!("📂 連接 LanceDB 路徑: {}", db_path);
    let db = connect(&db_path).execute().await?;
    
    println!("🧠 載入 Embedding 模型...");
    let cache_dir = env::var("FASTEMBED_CACHE_PATH")
        .unwrap_or_else(|_| ".fastembed_cache".to_string());
    
    println!("📂 使用模型快取路徑: {}", cache_dir);

    // 2. 設定選項
    let mut model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::BGESmallZHV15)
            // 🔥 [關鍵] 顯式指定 Cache 路徑
            .with_cache_dir(PathBuf::from(cache_dir)) 
            .with_show_download_progress(true)
    )?;
    
    let (summaries, synonyms) = sync_database_and_load_cache(&db, &mut model).await?;
    let llm_provider = env::var("LLM_PROVIDER").unwrap_or("google".to_string());
    let google_api_key = env::var("GOOGLE_API_KEY").unwrap_or_default();
    let local_llm_url = env::var("VLLM_ENDPOINT").unwrap_or("http://localhost:8000/v1/chat/completions".to_string());
    let local_llm_model = env::var("MODEL_NAME").unwrap_or("local-model".to_string());

    let redis_pool = match env::var("REDIS_URL") {
        Ok(url) => {

            match Config::from_url(url).create_pool(Some(Runtime::Tokio1)) {
                Ok(pool) => {
                    match pool.get().await {
                        Ok(_) => {
                            println!("✅ Redis 連線池建立成功 (Deadpool) - 連線測試通過");
                            Some(pool)
                        },
                        Err(e) => {
                            eprintln!("⚠️ Redis 設定格式正確，但無法連線至 Server: {}", e);
                            eprintln!("   (將降級使用純記憶體模式)");
                            None 
                        }
                    }
 
                },
                Err(e) => {
                    eprintln!("⚠️ Redis 設定失敗，將使用純前端記憶模式: {}", e);
                    None
                }
            }
        },
        Err(_) => {
            println!("ℹ️ 未設定 REDIS_URL，將使用純前端記憶模式");
            None
        }
    };

    Ok(Arc::new(AppState {
        db,
        model: Mutex::new(model),
        synonyms,
        summaries,
        llm_provider,
        google_api_key,
        local_llm_url,
        local_llm_model,
        redis_pool,
    }))
}


async fn search_in_lancedb(
    model: &mut TextEmbedding,
    db: &lancedb::Connection,
    query_text: &str,
    limit: usize,
    filter: Option<String> 
) -> Result<Vec<RecordBatch>, Box<dyn Error>> {
    

    let query_vec = model.embed(vec![query_text.to_string()], None)?[0].clone();

    let table = db.open_table(TABLE_NAME).execute().await?;

    let mut query_builder = table
        .query()
        .nearest_to(query_vec)?
        .limit(limit);

    if let Some(f) = filter {
        println!("🔍 [Vector Search] 套用過濾條件: {}", f);
        query_builder = query_builder.only_if(f);
    }

    let results = query_builder.execute().await?;


    let batches: Vec<RecordBatch> = results.try_collect().await?;
    Ok(batches)
}

pub async fn sync_database_and_load_cache(
    db: &lancedb::Connection,
    model: &mut TextEmbedding
) -> Result<(HashMap<String, ProductSummary>, HashMap<String, String>), Box<dyn Error>> {
    
    println!("🔄 開始執行資料同步與快取載入...");


    let mut summaries = HashMap::new();
    let mut synonyms = HashMap::new();


    let table_names = db.table_names().execute().await?;
    let table_exists = table_names.contains(&TABLE_NAME.to_string());

    let table = if !table_exists {
        println!("✨ 資料表不存在，建立新表...");

        let schema = Arc::new(Schema::new(vec![
            Field::new("source_file", DataType::Utf8, false),
            Field::new("file_hash", DataType::Utf8, false), 
            Field::new("text", DataType::Utf8, false),
            Field::new("vector", DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                512 
            ), false),
        ]));
        db.create_table(TABLE_NAME, RecordBatchIterator::new(vec![], schema)).execute().await?
    } else {
        db.open_table(TABLE_NAME).execute().await?
    };

    
    let mut existing_hashes: HashMap<String, String> = HashMap::new();
    
    if table_exists {
        
        match table.query()
            .select(Select::Columns(vec!["source_file".to_string(), "file_hash".to_string()]))
            .limit(10000)
            .execute()
            .await {
            Ok(mut stream) => {
                while let Ok(Some(batch)) = stream.try_next().await {
                    let src_col = batch.column_by_name("source_file").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
                    let hash_col = batch.column_by_name("file_hash").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
                    
                    for i in 0..batch.num_rows() {
                        let src = src_col.value(i).to_string();
                        let h = hash_col.value(i).to_string();
                        // 可能有多個 chunk 對應同一個檔案，我們只需要存一次
                        existing_hashes.insert(src, h);
                    }
                }
            },
            Err(_) => println!("⚠️ 無法讀取舊 Hash，將視為全部重新寫入。"),
        }
    }
    
    println!("📊 目前 DB 已索引 {} 份文件", existing_hashes.len());

    let walker = WalkDir::new(PROCESSED_JSON_DIR).into_iter();
    let mut new_chunks_buffer: Vec<(String, String, String)> = Vec::new(); 
    let mut updated_count = 0;
    let mut skipped_count = 0;
    let mut parse_error_count = 0;

    for entry in walker.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.extension().map_or(false, |e| e == "json") {
            //if let Ok(content) = fs::read_to_string(path) {
            let content = match fs::read_to_string(path) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("❌ 無法讀取檔案 {:?}: {}", path, e);
                    continue;
                }
            };

            
            match serde_json::from_str::<models::PolicyData>(&content) {
                Ok(data) => {
                    
                    let intro = format!(
                        "【商品總覽】\n名稱: {}\n類型: {}\n特色: {:?}\n適合對象: {}\n",
                        data.basic_info.product_name,
                        data.basic_info.product_type,
                        data.investment.features,
                        data.rag_data.target_audience
                    );
                    summaries.insert(data.source_filename.clone(), ProductSummary {
                        name: data.basic_info.product_name.clone(), 
                        intro: intro.clone(),
                    });


                    if let Some(mapping) = &data.rag_data.synonym_mapping {
                        let count_before = synonyms.len();
                        for entry in mapping {
                            let slangs: Vec<&str> = entry.slang.split(&['、', ','][..]).collect();
                            for s in slangs {
                                let clean_s = s.trim().to_string();
                                if !clean_s.is_empty() {
                                    synonyms.insert(clean_s, entry.formal.clone());
                                }
                            }
                        }
                        
                        println!("   📚 {} 載入 {} 個同義詞", data.source_filename, synonyms.len() - count_before);
                    } 
                    else {
                        
                        println!("   ⚠️ {} 沒有同義詞設定 (synonym_mapping is null)", data.source_filename);
                    }

                    
                    let current_hash = calculate_hash(&content);
                    let filename = data.source_filename.clone();
                    
                
                    let needs_update = match existing_hashes.get(&filename) {
                        Some(old_hash) => *old_hash != current_hash,
                        None => true, 
                    };

                    if needs_update {
                        if existing_hashes.contains_key(&filename) {
                            println!("📝 [變更] {} 內容已修，更新 DB...", filename);
                            table.delete(&format!("source_file = '{}'", filename)).await?;
                        } 
                        else {
                            println!("➕ [新增] {}", filename);
                        }
                        let mut final_chunks = Vec::new();

                        if !data.rag_data.chunks.is_empty() {
                            
                            final_chunks = data.rag_data.chunks;
                        } 
                        else {
                            
                            println!("   ⚙️ 自動組裝內容...");
                            
                            
                            let chunk_intro = format!(
                                "文件標題: {}\n{}\n【投保規則】\n年齡: {}\n保費限制: {}\n費用: {}\n【保障內容】\n身故: {}\n滿期: {}\n其他: {:?}\n【投資特色】\n{:?}\n風險: {:?}",
                                data.source_filename,
                                intro, 
                                data.conditions.age_range,
                                data.conditions.premium_limit,
                                data.conditions.fees_and_discounts,
                                data.coverage.death_benefit,
                                data.coverage.maturity_benefit,
                                data.coverage.other_benefits,
                                data.investment.features,
                                data.investment.risks
                            );
                            final_chunks.push(chunk_intro);

                            
                            let faqs = &data.rag_data.faq;
                            if !faqs.is_empty() {
                                let mut faq_buffer = String::from("【常見問答 FAQ】\n");
                                for (i, qa) in faqs.iter().enumerate() {
                                    faq_buffer.push_str(&format!("Q: {}\nA: {}\n\n", qa.q, qa.a));
                                    
                                    
                                    if (i + 1) % 3 == 0 || i == faqs.len() - 1 {
                                        final_chunks.push(faq_buffer.clone());
                                        faq_buffer = String::from("【常見問答 FAQ (續)】\n");
                                    }
                                }
                            }
                        }

                        for chunk_text in final_chunks {
                            
                            new_chunks_buffer.push((filename.clone(), current_hash.clone(), chunk_text));
                        }
                        updated_count += 1;
                    } 
                    else {
                        skipped_count += 1;
                    }
                },
                Err(e) => {
                    
                    eprintln!("❌ JSON 解析失敗 {:?}: {}", path.file_name().unwrap(), e);
                    parse_error_count += 1;
                }
            }
        }
    }

    println!("🔎 掃描統計:");
    println!("   - ✅ 成功載入摘要: {} 筆", summaries.len());
    println!("   - ✅ 成功載入同義詞: {} 筆", synonyms.len());
    println!("   - ⏭️ 資料庫略過 (無變更): {} 份", skipped_count);
    println!("   - ♻️ 資料庫更新 (有變更): {} 份", updated_count);
    if parse_error_count > 0 {
        println!("   - ❌ 解析失敗 (請檢查 models.rs): {} 份", parse_error_count);
    }
    
    if !new_chunks_buffer.is_empty() {
        println!("🚀 正在對 {} 個新段落進行 Embedding...", new_chunks_buffer.len());
        
        let batch_size = 50;
        for chunk in new_chunks_buffer.chunks(batch_size) {
            let texts: Vec<String> = chunk.iter().map(|(_, _, t)| t.clone()).collect();
            let sources: Vec<String> = chunk.iter().map(|(s, _, _)| s.clone()).collect();
            let hashes: Vec<String> = chunk.iter().map(|(_, h, _)| h.clone()).collect();
            
           
            let embeddings = model.embed(texts.clone(), None)?;
            
            
            let flat_vectors: Vec<f32> = embeddings.iter().flat_map(|v| v.clone()).collect();
            let dim = 512; // BGE-M3
            let schema = table.schema().await?;
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(StringArray::from(sources)),
                    Arc::new(StringArray::from(hashes)),
                    Arc::new(StringArray::from(texts)),
                    Arc::new(FixedSizeListArray::new(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        dim,
                        Arc::new(Float32Array::from(flat_vectors)),
                        None,
                    )),
                ],
            )?;
            let iterator = RecordBatchIterator::new(
            vec![Ok(batch)], 
                schema.clone()
            );
            table.add(iterator).execute().await?;
        }
        println!("✅ 資料庫同步完成！");
    } 
    else {
        println!("✅ 資料庫已是最新狀態，無需寫入。");
    }

    Ok((summaries, synonyms))
}