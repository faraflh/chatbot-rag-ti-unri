# Dockerfile untuk Hugging Face Spaces (Docker SDK).
# Catatan: HF Spaces FREE TIER tidak punya GPU. Image ini dibuat untuk mode
# OpenAI-compatible (LLM_BACKEND=openai_compat). Untuk mode hf_local (SeaLLM),
# Anda butuh upgrade ke HF Spaces berbayar dengan GPU dan menyesuaikan image
# dengan PyTorch CUDA.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/tmp/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/tmp/hf_cache \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

# Pre-build index (opsional; mempercepat boot pertama).
# Hilangkan baris ini kalau ingin build dilakukan saat runtime.
RUN python scripts/build_index.py || echo "Index build skipped (akan dilakukan saat runtime)"

EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
    "--server.port=7860", \
    "--server.address=0.0.0.0", \
    "--server.enableCORS=false", \
    "--server.enableXsrfProtection=false"]
