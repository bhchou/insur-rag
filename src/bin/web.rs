// src/bin/web.rs

use insur_rag::{init_system, process_query, AppState};
use axum::{
    extract::State,
    routing::post,
    Json, Router,
};
use tower_http::services::ServeDir;
use std::sync::Arc;
use std::net::SocketAddr;
use serde::Deserialize;
use serde_json::{Value, json};

// 前端傳來的請求格式
#[derive(Deserialize)]
struct ChatRequest {
    query: String,
    
    // 🔥 前端必須傳這個欄位，如果沒傳就是空陣列
    #[serde(default)] 
    messages: Vec<Value>, 
}


#[tokio::main]
async fn main() {
    // 初始化 Log
    tracing_subscriber::fmt::init();
    
    println!("🌐 啟動 Web Server 初始化...");
    
    // 1. 初始化核心系統 (跟 CLI 一樣！)
    let state = match init_system().await {
        Ok(s) => s,
        Err(e) => panic!("❌ 系統初始化失敗: {}", e),
    };

   
    // 2. 設定路由
    let app = Router::new()
        // API 接口
        .route("/api/chat", post(chat_handler))
        // 2. 所有沒對應到的路由 (例如 index.html, css, js)，全部交給 fallback 處理
        // ❌ 舊寫法 (會 Panic): .nest_service("/", ServeDir::new("frontend"))
        // ✅ 新寫法 (Axum 0.7+):
        .fallback_service(ServeDir::new("frontend"))
        .with_state(state);

    // 3. 啟動服務
    let port = std::env::var("PORT")
        .unwrap_or("8080".to_string())
        .parse::<u16>()
        .unwrap_or(8080);

    println!("✅ 系統就緒，Web Server 監聽中: http://localhost:{}", port);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// 處理 Chat 請求
async fn chat_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatRequest>, // 自動解析 JSON
) -> Json<serde_json::Value> {
    
    println!("📩 收到 Web 請求: {}", payload.query);

    // 🔥 3. 把 payload 裡的 messages 傳給 process_query
    match process_query(&state, &payload.messages, &payload.query).await {
        Ok(rag_result) => {
            // 🔥 修正關鍵：手動拆解 rag_result
            Json(json!({
                "status": "success",
                
                // 1. 把文字內容取出來，給前端的 "answer" 欄位
                "answer": rag_result.answer,   
                
                // 2. 把來源列表取出來，給前端的 "sources" 欄位
                "sources": rag_result.sources  
            }))
        },
        Err(e) => {
            Json(json!({
                "status": "error",
                "message": e.to_string()
            }))
        }
    }
}
