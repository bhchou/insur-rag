#!/bin/bash
set -e

# 1. 讀取環境變數，如果沒傳就預設 1000
# 這是為了讓這份 Image 在沒設定 PUID 時也能跑 (Defaults)
target_uid=${PUID:-1000}
target_gid=${PGID:-1000}

echo "🚀 Starting with PUID: ${target_uid}, PGID: ${target_gid}"

# 2. 動態修改 appuser 的 UID/GID
# -o: 允許非唯一 ID (Non-unique)，避免與系統內建帳號衝突
# usermod/groupmod 只有 root 能執行，這就是為什麼容器要用 root 啟動
groupmod -o -g "$target_gid" appuser
usermod -o -u "$target_uid" appuser

# 3. 修正權限 (Fix Permissions)
# 因為 UID 變了，原本屬於舊 ID 的檔案現在會讀不到，所以要刷一遍權限
# 注意：只刷需要的目錄，避免整顆硬碟刷太久
echo "🔧 Fixing permissions..."
chown -R appuser:appuser /app/data
chown -R appuser:appuser /app/lancedb_data
chown -R appuser:appuser /app/frontend 
# 如果有 log 或其他寫入點也要加

# 4. 降權執行 (Switch User)
# 使用 gosu 切換成 appuser 身份執行主程式
# "$@" 代表 Dockerfile 裡的 CMD 指令 (即 /app/server)
echo "✅ Switching to appuser and starting application..."
exec gosu appuser "$@"
