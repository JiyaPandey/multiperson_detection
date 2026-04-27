FROM python:3.10-slim

WORKDIR /app

# System deps (important for opencv, torch, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python deps (preload build-time deps first)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install numpy cython scipy pillow opencv-python gdown matplotlib pandas scikit-learn tqdm pyyaml tensorboard && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    grep -v "^git+https://github.com/KaiyangZhou/deep-person-reid.git" requirements.txt > requirements.no-reid.txt && \
    pip install --no-build-isolation -r requirements.no-reid.txt && \
    pip install --no-build-isolation --no-deps git+https://github.com/KaiyangZhou/deep-person-reid.git

# Copy everything
COPY . .

# Streamlit config (no browser open)
RUN mkdir -p /root/.streamlit
RUN echo "\
[server]\n\
headless = true\n\
port = 8501\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml

EXPOSE 8501

CMD ["streamlit", "run", "src/ui/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]