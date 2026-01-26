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

use redis::Client;

// LanceDB 與 Arrow 相關引入
use lancedb::{connect, query::{ExecutableQuery, QueryBase}};
use arrow_schema::{Schema, Field, DataType};
use arrow_array::{RecordBatch, RecordBatchIterator, StringArray, Array};
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
    pub redis_client: Option<Client>,
}

#[derive(Serialize, Debug)]
pub struct RagResponse {
    pub answer: String,
    pub sources: Vec<String>,
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

    // [策略 B] 主動式 AI 意圖改寫 (Pre-emptive Rewrite) 🔥 這是剛才討論的重點
    // 條件：有歷史紀錄 AND (問題很短 OR 包含代名詞)
    // 這裡我們簡單用字數判斷 (< 20 字)
    let should_rewrite = !history.is_empty() && user_query.chars().count() < 20;
    
    if should_rewrite {
        println!("🤔 偵測到短問題且有歷史，嘗試進行「主動意圖改寫」...");
        if let Some(rewritten) = expand_query_with_ai(state, history, user_query).await {
            println!("✅ AI 改寫成功: '{}'", rewritten);
            let mut final_rewritten = rewritten.clone();
            
            // 🔥【防呆實作】檢查邏輯
            // 1. 原始問題長度檢查：大於 2 個中文字 (6 bytes)，避免把 "那呢"、"是的" 這種無意義短語補進去。
            // 2. 遺失檢查：如果 AI 改寫後的字串裡，竟然找不到原始使用者的輸入。
            if user_query.len() > 6 && !final_rewritten.contains(user_query) {
                println!("⚠️ [防呆觸發] AI 改寫遺失使用者關鍵意圖，強制補回！");
                
                // 動作：直接把原始問題 append 在後面
                // 範例狀況：
                //   History: "58歲"
                //   Query: "那外幣投資呢"
                //   AI Fail: "58歲男性" (漏了外幣)
                //   Fix Result: "58歲男性 那外幣投資呢" 
                // (雖然句子不通順，但對向量搜尋來說，關鍵字齊全才是最重要的)
                
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

    let mut forced_candidates: Vec<(String, String, f32)> = Vec::new();
    let mut forced_filenames = HashSet::new();

    // 1. 提取括弧內的文字 (支援 『』 「」 或 "")
    // 這邊假設使用者會用這些常見括弧
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

        let table = state.db.open_table(TABLE_NAME).execute().await?;
        let specific_results = table
            .query()
            .only_if(filter_cond)
            .limit(10) // 每個檔案抓前幾段摘要即可
            .execute()
            .await?;

        let batches: Vec<RecordBatch> = specific_results.try_collect().await?;
        
        // 將結果轉為 candidates 格式
        for batch in batches {
            let src_col = batch.column_by_name("source_file").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
            let txt_col = batch.column_by_name("text").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
            
            for i in 0..batch.num_rows() {
                let src = src_col.value(i).to_string();
                let txt = txt_col.value(i).to_string();
                // 🔥 給予無限大的分數 (f32::INFINITY)，確保它在 Re-rank 前絕對是第一名
                forced_candidates.push((src, txt, f32::INFINITY));
            }
        }
    }

    // 1. 向量化問題
    // let query_embedding = model.embed(vec![user_query.to_string()], None)?;
    // let query_vector = query_embedding[0].clone();
    // let query_vec = model.embed(vec![final_query.clone()], None)?[0].clone();
    println!("🔍 執行向量搜尋: {}", search_target);
    let query_vec = model.embed(vec![search_target.clone()], None)?[0].clone();
    // 2. 搜尋 DB
    let table = db.open_table(TABLE_NAME).execute().await?;
    let results = table
        .query()
        .nearest_to(query_vec)?
        .limit(recall_limit) // 取前 3 個最相關的片段
        .execute()
        .await?;


    let vector_batches: Vec<RecordBatch> = results.try_collect().await?;

    // --- 5. 候選結果合併 (Merge & Deduplicate) ---
    let mut raw_candidates: Vec<(String, String)> = Vec::new();
    let mut seen_texts = HashSet::new();

    // (1) 優先放入強制命中的
    for (src, txt, _) in forced_candidates {
        if seen_texts.insert(txt.clone()) {
            raw_candidates.push((src, txt));
        }
    }

    // (2) 再放入向量搜尋的
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

    
    // --- 6. 檢查結果 (Fallback 邏輯可選) ---
    // 由於我們前面已經做了 Pre-emptive Rewrite，這裡的 Fallback 重要性降低
    // 但如果你想保留「搜不到東西時再試一次」的邏輯，可以寫在這裡
    // 不過根據新策略，通常不需要二次 Embedding 了

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
    // 2. 定義「受保護」的寬鬆險種 (當嚴格模式啟動時，這些關鍵字也要被允許)
    let protected_rules = vec![
        ("壽險", vec!["壽險", "身故", "人壽", "儲蓄", "還本"]),
        ("意外", vec!["意外", "傷害", "骨折", "產險"]),
    ];

    let mut allowed_keywords: Vec<&str> = Vec::new();
    let mut strict_mode_triggered = false;

    // 3. 掃描嚴格規則 (支援多重命中)
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

    // 5. 執行過濾 (只有在嚴格模式觸發時才做)
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

        // 防呆：如果濾完變 0 筆 (例如 User 同時問了兩個資料庫都沒有的險種)
        if raw_candidates.is_empty() {
             println!("⚠️ 過濾後無結果，取消過濾條件。");
             // 這裡建議回復備份，或者就讓它回傳無結果
        }
    }



    // --- 7. Re-ranking (關鍵優化) ---
    // 注意：Rerank 時建議用「改寫後的 search_target」還是「原始 user_query」？
    // 建議：用 search_target (因為它消除了代名詞)，Reranker 比較看得懂
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

    // 7. 最後生成
    //ask_llm(&final_context, user_query).await?;
    let llm_answer = ask_llm(state, &final_context, &search_target).await?;
    
    // 整理來源列表
    let mut sorted_sources: Vec<String> = hit_files.into_iter().collect();
    sorted_sources.sort();

    // ✅ 回傳結構化資料
    Ok(RagResponse {
        answer: llm_answer,
        sources: sorted_sources,
    })
}

// 回傳 (摘要Map, 同義詞Map)
fn load_data_from_json_dir() -> (HashMap<String, ProductSummary>, HashMap<String, String>) {
    let mut summaries = HashMap::new();
    let mut synonyms = HashMap::new();
    
    println!("🚀 Rust 正在掃描 JSON 資料夾建立快取...");
    
    let walker = WalkDir::new(PROCESSED_JSON_DIR).into_iter();
    
    for entry in walker.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.extension().map_or(false, |e| e == "json") {
            if let Ok(content) = fs::read_to_string(path) {
                // 嘗試解析 JSON
                if let Ok(data) = serde_json::from_str::<models::PolicyData>(&content) {
                    
                    // --- 1. 處理摘要 (原有邏輯) ---
                    let intro = format!(
                        "【商品總覽】\n名稱: {}\n類型: {}\n特色: {:?}\n適合對象: {}\n",
                        data.basic_info.product_name,
                        data.basic_info.product_type,
                        data.investment.features,
                        data.rag_data.target_audience
                    );

                    summaries.insert(data.source_filename.clone(), ProductSummary {
                        name: data.basic_info.product_name,
                        intro,
                    });

                    // --- 2. 處理同義詞 (新增邏輯) ---
                    // 假設 models::RagData 裡面有 synonym_mapping 欄位
                    // 注意：您需要在 models.rs 裡對應加上這個欄位，如果沒有的話
                    if let Some(mapping) = &data.rag_data.synonym_mapping {
                        for entry in mapping {
                            // 處理逗號分隔 (例如: "死掉, 走了")
                            let slangs: Vec<&str> = entry.slang.split(&['、', ','][..]).collect();
                            for s in slangs {
                                let clean_s = s.trim().to_string();
                                if !clean_s.is_empty() {
                                    // 建立反向索引: 口語 -> 專業術語
                                    synonyms.insert(clean_s, entry.formal.clone());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    println!("📚 資料載入完成！");
    println!("   - 商品摘要: {} 筆", summaries.len());
    println!("   - 同義詞庫: {} 筆", synonyms.len());
    
    (summaries, synonyms)
}

pub async fn expand_query_with_ai(state: &Arc<AppState>, history: &[Value], query: &str) -> Option<String> {
    // 建立指代消解專用的 System Prompt
    let system_prompt = r#"
    你是一個搜尋關鍵字優化機器人。你的唯一任務是將「對話歷史」與「最新問題」合併，產生一個「完整的搜尋語句」。
    
    【合成公式】：
    ❌ 錯誤模式：只輸出歷史背景 (如："30歲男性") -> 禁止！
    ❌ 錯誤模式：只輸出最新問題 (如："壽險推薦") -> 禁止！
    ✅ 正確模式：[使用者畫像] + [最新問題的具體關鍵字]
    
    【執行規則】：
    1. **提取畫像**：從歷史中找出年齡、性別、職業 (例如：58歲男性)。
    2. **鎖定意圖**：從「最新問題」中找出他想問的商品或話題 (例如：外幣投資)。
    3. **指代還原**：如果最新問題有「那...呢」、「它...」，請替換為上一個討論的商品；如果是新話題，則保留新話題。
    
    【範例】：
    History: "我是30歲工程師"
    Current: "那醫療險呢？"
    Result: "適合30歲工程師的醫療險推薦"  <-- (一定要包含 '醫療險')

    History: "58歲退休"
    Current: "想找投資"
    Result: "58歲退休族適合的投資型保單" <-- (一定要包含 '投資')
    
    請直接輸出結果，不要解釋。
    "#;
    
    // 準備歷史訊息字串 (給 Gemini 或 Local LLM 參考用)
    // 我們取最後 4 句就好，避免 Token 爆炸
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

// 路徑 1: 本地 LLM (使用 .no_proxy())
async fn expand_local(state: &Arc<AppState>, system_prompt: &str, user_content: &str) -> Result<String, Box<dyn Error>> {
    // 🔥 關鍵：這裡必須用 no_proxy，否則連不到 host.docker.internal
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
            { "role": "user", "content": user_content } // 這裡把歷史+問題包在一起給它
        ],
        "temperature": 0.1, // 改寫不需要創意，越低越好
        "max_tokens": 1024   // 改寫通常很短
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

// 路徑 2: Google Gemini (使用標準 Proxy)
async fn expand_google(state: &Arc<AppState>, system_prompt: &str, user_content: &str) -> Result<String, Box<dyn Error>> {
    if state.google_api_key.is_empty() {
        return Err("缺少 GOOGLE_API_KEY".into());
    }

    // 🔥 關鍵：這裡使用預設 Client，會自動讀取 HTTPS_PROXY 環境變數
    let client = reqwest::Client::new();

    // Gemini 的 Prompt 組合方式
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


// ✅ 修改函式簽名：輸入改為 candidates: Vec<(String, String)>
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

    // 1. 準備給 Re-ranker API 的資料
    // 我們需要保留原始的 (src, txt) 對應關係，同時準備一份「注入摘要」的版本給 AI 讀
    let mut doc_texts_for_api: Vec<String> = Vec::new();

    for (src, txt) in &candidates {
        // 為了讓 Re-ranker 判斷準確，我們把「摘要」也加進去給它讀
        // 這樣它才知道 "優利精選" 是投資型保單
        let content_for_judge = if let Some(sum) = summaries.get(src) {
            format!("{}\n文件內容: {}", sum.intro, txt)
        } else {
            txt.clone()
        };
        doc_texts_for_api.push(content_for_judge);
    }

    // 2. 呼叫 Python Re-ranker API
    // let client = reqwest::Client::new();
    let client = reqwest::Client::builder()
        .no_proxy()
        .build()?;
    let request_body = RerankRequest {
        query: query.to_string(),
        documents: doc_texts_for_api,
    };

    println!("⚖️ 正在進行 Re-ranking ({} 筆候選, 取 Top {} 到 {})...", candidates.len(), top_k, api_url);

    let resp = client.post(api_url)
        .json(&request_body)
        .send()
        .await?;

    let rerank_res: RerankResponse = resp.json().await?;

    // 3. 根據回傳的 indices 重新組裝結果
    let mut ranked_results = Vec::new();
    let mut file_counts: HashMap<String, usize> = HashMap::new();
    
    for (i, &original_idx) in rerank_res.indices.iter().enumerate() {
        if ranked_results.len() >= top_k { break; }
        
        let score = rerank_res.scores[i];
        
        // 💡 門檻值過濾
        if score < -5.0 { 
            continue; 
        }

        // 🔥 關鍵改變：直接從傳入的 candidates 取值
        // original_idx 是 Python 回傳的原始索引，對應到 candidates 的順序
        let (src, txt) = &candidates[original_idx];
        
        // 檢查這份檔案是否已經額滿 (多樣性過濾)
        let count = file_counts.entry(src.clone()).or_insert(0);
        
        if *count < max_chunks_per_doc {
            println!("   ⭐ [Top {}] 分數: {:.2} | 來源: {}", i+1, score, src);
            ranked_results.push((src.clone(), txt.clone(), score));
            *count += 1;
        }
        else {
            println!("   ⏭️ [跳過] 檔案額滿 ({}/{}): {:.2} | 來源: {}", *count, max_chunks_per_doc, score, src);
        }
    }

    Ok(ranked_results)
}


// 4. 新增初始化函式 (從原本 main 提取)
pub async fn init_system() -> Result<Arc<AppState>, Box<dyn Error>> {
    dotenv().ok();
    
    let db_path = std::env::var("LANCEDB_PATH").unwrap_or(DB_URI.to_string());
    println!("📂 連接 LanceDB 路徑: {}", db_path);
    let db = connect(&db_path).execute().await?;
    // 初始化 DB
    //let db = connect(DB_URI).execute().await?;
    //println!("💾 連線至資料庫: {}", DB_URI);

    //建立 Table (如果不存在)
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

    let table_names = db.table_names().execute().await?;
    let _table = if table_names.contains(&TABLE_NAME.to_string()) {
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
    
    println!("🧠 載入 Embedding 模型...");
    let model = TextEmbedding::try_new(InitOptions::new(EmbeddingModel::BGEBaseENV15))?;
    
    // 載入資料 (這裡假設您已經合併了讀取函式，或保留原本分開的)
    //let summaries = load_product_summaries(); 
    //let synonyms = load_synonyms();
    let (summaries, synonyms) = load_data_from_json_dir();
    let llm_provider = env::var("LLM_PROVIDER").unwrap_or("google".to_string());
    let google_api_key = env::var("GOOGLE_API_KEY").unwrap_or_default();
    let local_llm_url = env::var("VLLM_ENDPOINT").unwrap_or("http://localhost:8000/v1/chat/completions".to_string());
    let local_llm_model = env::var("MODEL_NAME").unwrap_or("local-model".to_string());

    let redis_client = match env::var("REDIS_URL") {
        Ok(url) => {
            match Client::open(url) {
                Ok(client) => {
                    println!("✅ Redis 連線設定成功");
                    Some(client)
                },
                Err(e) => {
                    eprintln!("⚠️ Redis URL 格式錯誤，將使用純前端記憶模式: {}", e);
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
        redis_client,
    }))
}
