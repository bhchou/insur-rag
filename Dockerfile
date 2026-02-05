# ========================
# Stage 1: Builder (編譯層)
# ========================
# 🔥 [重大改變] 改用 Ubuntu 24.04
# 這能確保 glibc 版本 >= 2.38，解決 __isoc23_strtol 錯誤
FROM ubuntu:24.04 AS builder

WORKDIR /app

# 1. 安裝系統依賴 & 下載工具
# Ubuntu 預設沒有 Rust，我們要手裝
RUN echo "Acquire::https::Verify-Peer \"false\";" > /etc/apt/apt.conf.d/99ignore-ssl && \
    apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    clang \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. 🔥 手動安裝 Rust (安裝最新穩定版)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# 將 cargo 加入 PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# 3. 設定編譯參數
# 使用 Clang 編譯 C/C++ 依賴 (解決 LanceDB/Ort 相容性)
ENV CC=clang
ENV CXX=clang++

# 4. 依賴快取
COPY Cargo.toml Cargo.lock ./

# 建立 Dummy 檔案
RUN mkdir -p src/bin && \
    echo "fn main() {println!(\"dummy\")}" > src/main.rs && \
    echo "fn main() {}" > src/bin/cli.rs && \
    echo "fn main() {}" > src/bin/web.rs && \
    touch src/lib.rs && \
    # 🔥 記得加 -j 4 避免記憶體爆掉
    cargo build --release --bin web -j 4

# 5. 編譯真正的程式碼
COPY src ./src

# 🔥 記得加 -j 4
RUN touch src/main.rs src/lib.rs src/bin/web.rs && \
    cargo build --release --bin web -j 4

# 🔥 [瘦身關鍵] 移除 Debug Symbol
# 這步通常能把 150MB 的執行檔變成 15MB
RUN strip /app/target/release/web

# ========================
# Stage 2: Runtime (執行層)
# ========================
# 🔥 Runtime 也要用 Ubuntu 24.04，確保 glibc 版本一致
FROM ubuntu:24.04

#ARG USER_ID=1000
#ARG GROUP_ID=1000
# 建立使用者，不要用 root 跑 (Trivy 很在意這點)
#RUN groupadd -g ${GROUP_ID} appuser || true && \
#    useradd -m -u ${USER_ID} -g ${GROUP_ID} -o --no-log-init appuser

#WORKDIR /app

# 安裝 Runtime 依賴
RUN echo "Acquire::https::Verify-Peer \"false\";" > /etc/apt/apt.conf.d/99ignore-ssl && \
    apt-get update && apt-get install -y \
    openssl \
    ca-certificates \
    gosu \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    # 👇 [關鍵修正] 先刪除佔用 1000 的 ubuntu 使用者與群組
    && (userdel -r ubuntu || true) \
    && (groupdel ubuntu || true) \
    && groupadd -g 1000 appuser \
    && useradd -m -u 1000 -g appuser appuser

WORKDIR /app

# 複製 Binary
COPY --from=builder /app/target/release/web /app/server

# 建立資料夾
RUN mkdir -p data frontend data/processed_json lancedb_data data/model_cache

COPY data/processed_json /app/data/processed_json
COPY frontend /app/frontend

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
# 🔥 3. [關鍵一步] 更改權限 (把 /app 下所有東西送給 appuser)
# 如果沒做這步，appuser 之後會無法寫入 /app/data 或產生 log
# RUN chown -R appuser:appuser /app

# 環境變數
ENV RUST_LOG=info
ENV HOST=0.0.0.0
ENV PORT=8081

EXPOSE 8081

USER root

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD ["/app/server"]