// src/bin/cli.rs

use insur_rag::{init_system, process_query}; // 引用剛剛的 lib
use std::io::{self, Write};
use std::error::Error;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🔥 初始化系統中...");
    
    // 1. 一行程式碼完成初始化
    let state = init_system().await?;
    let mut chat_history: Vec<serde_json::Value> = Vec::new();
    
    println!("\n🤖 保險 AI 顧問 (CLI 版) 已就緒");
    println!("💡 輸入問題 (例如: '安聯新吉星有什麼費用?' 或 'exit' 離開; 若提到產品名稱可以用單雙引號,括號,或中文引號『「』」《》【】框起來會更準)");

    // 2. 進入互動迴圈
    loop {
        print!("\nUser > ");
        io::stdout().flush()?;
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_ok() {
            let q = input.trim();
            if q.eq_ignore_ascii_case("exit") { break; }
            if q.is_empty() { continue; }

            // 3. 呼叫 Library 處理
            //if let Err(e) = process_query(&state, q).await {
            //    eprintln!("❌ 處理發生錯誤: {}", e);
            //}
            match process_query(&state, &chat_history, q).await {
                Ok(response) => {
                // 🔥 CLI 自己決定怎麼印
                    println!("\n💬 AI 回答：\n=========================");
                    println!("{}", response.answer);
                    println!("=========================\n");
                
                    println!("📚 參考來源：");
                    for (i, src) in response.sources.iter().enumerate() {
                        println!(" {}. {}", i + 1, src);
                    }
                    chat_history.push(json!({ "role": "user", "content": q }));
                    // 存 AI 的回答
                    chat_history.push(json!({ "role": "assistant", "content": response.answer }));
                
                }
                Err(e) => eprintln!("❌ 錯誤: {}", e),
            }
        }
    }

    Ok(())
}