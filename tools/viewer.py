# inspect_db.py
import lancedb
import pandas as pd

# 設定顯示寬度，避免內容被截斷
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 50) # 內容只顯示前50字
pd.set_option('display.width', 1000)

# 1. 連線
uri = "data/lancedb_store"
db = lancedb.connect(uri)

# 2. 列出所有 Table
print(f"📂 資料庫中的 Tables: {db.list_tables()}")

# 3. 讀取 Table
table_name = "insurance_docs"
if table_name in db.list_tables():
    tbl = db.open_table(table_name)
    
    # 4. 顯示統計資訊
    print(f"📊 總筆數: {tbl.count_rows()}")
    
    # 5. SQL 查詢 (沒錯，它支援 SQL!)
    # 例如：找出 product_name 是 Unknown 的髒資料
    df = tbl.search().where("product_name = 'Unknown'").limit(5).to_pandas()
    
    if not df.empty:
        print("\n⚠️ 發現髒資料範例:")
        print(df[['product_name', 'text']])
    else:
        print("\n✅ 沒有發現 'Unknown' 的資料")

    # 6. 隨機看 3 筆正常資料
    print("\n👀 資料預覽 (前 3 筆):")
    print(tbl.head(3).to_pandas()[['product_name', 'text', 'vector']])
