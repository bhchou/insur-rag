// src/bin/cli.rs

use insur_rag::{init_system, process_query}; // 引用剛剛的 lib
use std::io::{self, Write};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🔥 初始化系統中...");
    
    // 1. 一行程式碼完成初始化
    let state = init_system().await?;
    
    println!("\n🤖 保險 AI 顧問 (CLI 版) 已就緒");
    println!("💡 輸入問題 (例如: '安聯新吉星有什麼費用?' 或 'exit' 離開)");

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
            match process_query(&state, q).await {
                Ok(response) => {
                // 🔥 CLI 自己決定怎麼印
                    println!("\n💬 AI 回答：\n=========================");
                    println!("{}", response.answer);
                    println!("=========================\n");
                
                    println!("📚 參考來源：");
                    for (i, src) in response.sources.iter().enumerate() {
                        println!(" {}. {}", i + 1, src);
                    }
                }
                Err(e) => eprintln!("❌ 錯誤: {}", e),
            }
        }
    }

    Ok(())
}