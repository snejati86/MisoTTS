FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    MISO_TTS_STUDIO_OUTPUT_DIR=/app/outputs/studio \
    MISO_TTS_AUTOLOAD=0 \
    NO_TORCH_COMPILE=1 \
    PYTORCH_ENABLE_MPS_FALLBACK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ffmpeg \
        git \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "scripts/start_studio.py", "--host", "0.0.0.0", "--port", "7860", "--no-browser"]
