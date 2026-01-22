# ========================
# Stage 1: Builder (編譯層)
# ========================
# 🔥 [重大改變] 改用 Ubuntu 24.04
# 這能確保 glibc 版本 >= 2.38，解決 __isoc23_strtol 錯誤
FROM ubuntu:24.04 as builder

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

# ========================
# Stage 2: Runtime (執行層)
# ========================
# 🔥 Runtime 也要用 Ubuntu 24.04，確保 glibc 版本一致
FROM ubuntu:24.04

WORKDIR /app

# 安裝 Runtime 依賴
RUN echo "Acquire::https::Verify-Peer \"false\";" > /etc/apt/apt.conf.d/99ignore-ssl && \
    apt-get update && apt-get install -y \
    openssl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 複製 Binary
COPY --from=builder /app/target/release/web /app/server

# 建立資料夾
RUN mkdir -p data frontend

# 環境變數
ENV RUST_LOG=info
ENV HOST=0.0.0.0
ENV PORT=8081

EXPOSE 8081

CMD ["/app/server"]