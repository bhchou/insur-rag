// src/bin/web.rs

use insur_rag::{init_system, process_query, AppState};
use axum::{
    extract::State,
    routing::post,
    Json, Router,
    http::StatusCode,
};
use tower_http::services::ServeDir; // 🔥 關鍵模組
use std::sync::Arc;
use std::net::SocketAddr;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use redis::AsyncCommands;

// 定義回傳給前端的格式
#[derive(Serialize)]
struct ChatResponse {
    answer: String,
    sources: Vec<String>,
}

// 定義前端傳來的請求格式
#[derive(Deserialize)]
struct ChatRequest {
    query: String,
    #[serde(default)] 
    messages: Vec<Value>, 
    #[serde(default)]
    session_id: Option<String>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    println!("🌐 啟動 Web Server 初始化...");
    
    let state = match init_system().await {
        Ok(s) => s,
        Err(e) => panic!("❌ 系統初始化失敗: {}", e),
    };

    let app = Router::new()
        // 🔥 API 路由優先
        .route("/api/chat", post(chat_handler))
        
        // 🔥 靜態檔案路由 (Fallback)
        // 所有沒對應到的 URL，都會去 "frontend" 資料夾找檔案
        // 訪問 / 會自動找 index.html
        .fallback_service(ServeDir::new("frontend"))
        
        .with_state(state);

    let port_str = std::env::var("PORT").unwrap_or("8080".to_string());
    let port = port_str.parse::<u16>().unwrap_or(8080);

    println!("✅ 系統就緒，Web Server 監聽中: http://localhost:{}", port);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn chat_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, (StatusCode, String)> {
    
    // --- 1. 混合記憶邏輯 ---
    let mut history = payload.messages.clone();
    let mut use_redis = false;
    let redis_key = payload.session_id.as_ref().map(|id| format!("chat:{}", id));

    if let (Some(client), Some(key)) = (&state.redis_client, &redis_key) {
        if let Ok(mut conn) = client.get_multiplexed_async_connection().await {
            let redis_history: Result<Vec<String>, _> = conn.lrange(key, -10, -1).await;
            if let Ok(hist_json) = redis_history {
                if !hist_json.is_empty() {
                    println!("🧠 [Redis] 成功載入 {} 筆歷史紀錄", hist_json.len());
                    history = hist_json.iter()
                        .filter_map(|s| serde_json::from_str(s).ok())
                        .collect();
                    use_redis = true;
                }
            }
        }
    }

    if !use_redis {
        println!("📝 [Fallback] 使用前端傳送的歷史紀錄");
    }

    // --- 2. 呼叫核心 ---
    println!("📩 收到請求: {}", payload.query);
    let rag_result = process_query(&state, &history, &payload.query).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // --- 3. 寫回 Redis ---
    if use_redis {
        if let (Some(client), Some(key)) = (&state.redis_client, &redis_key) {
            if let Ok(mut conn) = client.get_multiplexed_async_connection().await {
                let user_msg = json!({"role": "user", "content": payload.query});
                let ai_msg = json!({"role": "assistant", "content": rag_result.answer});

                let _: redis::RedisResult<()> = redis::pipe()
                    .rpush(key, user_msg.to_string())
                    .rpush(key, ai_msg.to_string())
                    .expire(key, 86400)
                    .query_async(&mut conn).await;
            }
        }
    }

    Ok(Json(ChatResponse {
        answer: rag_result.answer,
        sources: rag_result.sources,
    }))
}