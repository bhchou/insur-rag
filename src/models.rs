use serde::{Deserialize, Serialize};
use serde::de::Deserializer;
use serde_json::Value;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct InsuranceMetadata {
    // 加上 deserialize_with，處理 product_name 變成陣列的情況 
    // 這樣即使 LLM 回傳 ["A", "B"]，這裡也會自動變成字串 "A, B"，不會報錯
    #[serde(default, deserialize_with = "deserialize_string_or_seq")]
    pub product_name: String,

    #[serde(default, deserialize_with = "deserialize_optional_string_or_seq")]
    pub product_code: Option<String>,

    // 對應 JSON: ["終身壽險", "美元保單"...] -> Rust Vec<String>
    pub insurance_type: Vec<String>,

    // 對應 JSON: ["身故...", "完全失能..."] -> Rust Vec<String>
    pub benefits: Vec<String>,

    // 對應 JSON: "USD"
    #[serde(default, deserialize_with = "deserialize_string_or_seq")]
    pub currency: String,

    #[serde(default, deserialize_with = "deserialize_optional_string_or_seq")]
    pub target_audience: Option<String>,
}


#[derive(Debug, Serialize, Deserialize)]
pub struct ParsedDocument {
    pub metadata: InsuranceMetadata,
    pub full_text: String, 
}


fn deserialize_string_or_seq<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let v: Value = Deserialize::deserialize(deserializer)?;
    match v {
        Value::String(s) => Ok(s),
        Value::Array(arr) => {
            // 把陣列裡的字串全部接起來
            let joined = arr.iter()
                .map(|val| val.as_str().unwrap_or("").to_string())
                .collect::<Vec<String>>()
                .join(", ");
            Ok(joined)
        },
        Value::Null => Ok("Unknown Product".to_string()), // 預設值
        _ => Ok(v.to_string()),
    }
}


fn deserialize_optional_string_or_seq<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let v: Value = Deserialize::deserialize(deserializer)?;
    match v {
        Value::String(s) => {
            if s.trim().is_empty() { Ok(None) } else { Ok(Some(s)) }
        },
        Value::Array(arr) => {
            let joined = arr.iter()
                .map(|val| val.as_str().unwrap_or("").to_string())
                .collect::<Vec<String>>()
                .join(", ");
            Ok(Some(joined))
        },
        Value::Null => Ok(None),
        _ => Ok(Some(v.to_string())),
    }
}

/* 底下是走JSON的資料 */
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PolicyData {
    // 這是 Python 裡塞進去的 "source_filename"
    pub source_filename: String, 
    pub basic_info: BasicInfo,
    pub conditions: Conditions,
    pub coverage: Coverage,
    pub investment: Investment,
    pub rag_data: RagData,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BasicInfo {
    pub product_name: String,
    pub product_code: String,
    pub company: String,
    pub currency: Vec<String>,
    pub product_type: String,
    pub payment_period: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Conditions {
    pub age_range: String,
    pub premium_limit: String,
    pub fees_and_discounts: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Coverage {
    pub death_benefit: String,
    pub maturity_benefit: String,
    pub other_benefits: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Investment {
    pub is_investment_linked: bool,
    pub features: Vec<String>,
    pub risks: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SynonymEntry {
    pub slang: String,
    pub formal: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RagData {
    pub target_audience: String,
    pub keywords: Vec<String>,
    
    #[serde(default)] 
    pub chunks: Vec<String>,
    
    pub synonym_mapping: Option<Vec<SynonymEntry>>,
    pub faq: Vec<FaqItem>,
    
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FaqItem {
    pub q: String,
    pub a: String,
}
