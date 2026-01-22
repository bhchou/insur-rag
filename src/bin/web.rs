// src/bin/web.rs

use insur_rag::{init_system, process_query, AppState, RagResponse};
use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use tower_http::services::ServeDir;
use std::sync::Arc;
use std::net::SocketAddr;
use serde::Deserialize;

// 前端傳來的請求格式
#[derive(Deserialize)]
struct ChatRequest {
    message: String,
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
    Json(payload): Json<ChatRequest>,
) -> Json<RagResponse> {
    println!("📩 收到 Web 請求: {}", payload.message);

    // 呼叫核心邏輯
    match process_query(&state, &payload.message).await {
        Ok(response) => Json(response),
        Err(e) => {
            eprintln!("❌ 處理錯誤: {}", e);
            // 發生錯誤時回傳一個空的錯誤訊息 (或是你可以自定義錯誤結構)
            Json(RagResponse {
                answer: format!("系統發生錯誤: {}", e),
                sources: vec![],
            })
        }
    }
}