import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
import torch
import os

app = FastAPI(title="Local Rerank Service")

# 設定模型路徑 (您可以預先下載，或第一次執行時會自動下載)
MODEL_NAME = "BAAI/bge-reranker-v2-m3"

print(f"⏳ 正在載入 Re-ranker 模型: {MODEL_NAME} ...")

# 判斷是否有 GPU (WSL2 若有設定好 CUDA 就能用，沒有就跑 CPU)
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 運算裝置: {device}")

# 載入 CrossEncoder
model = CrossEncoder(MODEL_NAME, device=device)
print("✅ 模型載入完成！")

# 定義請求資料結構
class RerankRequest(BaseModel):
    query: str
    documents: list[str] # 這是純文字列表

# 定義回應資料結構
class RerankResponse(BaseModel):
    scores: list[float]
    indices: list[int] # 回傳排序後的索引 (從高分到低分)

@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    if not request.documents:
        return {"scores": [], "indices": []}

    # 準備模型輸入: [(query, doc1), (query, doc2), ...]
    pairs = [[request.query, doc] for doc in request.documents]
    
    # 進行推論 (打分數)
    try:
        scores = model.predict(pairs)
        
        # 轉成 List
        scores_list = scores.tolist()
        
        # 取得排序後的索引 (Argsort Descending)
        # 也就是分數最高的排前面
        sorted_indices = sorted(
            range(len(scores_list)), 
            key=lambda k: scores_list[k], 
            reverse=True
        )
        
        # 也可以選擇在這裡直接過濾掉負分的結果 (視需求而定)
        
        return {
            "scores": [scores_list[i] for i in sorted_indices],
            "indices": sorted_indices
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 跑在 8000 Port (或其他您喜歡的 Port)
    uvicorn.run(app, host="0.0.0.0", port=8009)