use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct InsuranceMetadata {
    // 對應 JSON: "product_name"
    pub product_name: String,

    // 對應 JSON: "product_code"
    // 因為有時候可能沒抓到或格式不同，用 Option 比較安全，
    // 且您的值其實是備查文號，保留 String 很合適
    pub product_code: Option<String>,

    // 對應 JSON: ["終身壽險", "美元保單"...] -> Rust Vec<String>
    pub insurance_type: Vec<String>,

    // 對應 JSON: ["身故...", "完全失能..."] -> Rust Vec<String>
    pub benefits: Vec<String>,

    // 對應 JSON: "USD"
    pub currency: String,
    pub target_audience: Option<String>,
}

// 這是為了包含全文內容的 Wrapper (假設 Python 最終會吐出 JSON + 全文)
#[derive(Debug, Serialize, Deserialize)]
pub struct ParsedDocument {
    pub metadata: InsuranceMetadata,
    pub full_text: String, 
}
