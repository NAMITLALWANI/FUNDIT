# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────
#  FUNDIT — AI Mutual Fund Decision Engine
#  Production Dockerfile
# ─────────────────────────────────────────────────────────────

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /app

# ── System dependencies ─────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ─────────────────────────────────────
# Copy only requirements first so this layer is cached unless deps change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding + reranker models into the image so the container
# does not need outbound internet access at runtime.
# Skip if you want smaller images and will rely on HuggingFace cache mounted as a volume.
ARG PREDOWNLOAD_MODELS=true
RUN if [ "$PREDOWNLOAD_MODELS" = "true" ]; then \
      python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu'); \
print('Models downloaded.')"; \
    fi

# ── Application source ───────────────────────────────────────
COPY app/      app/
COPY data/     data/
COPY scripts/  scripts/
COPY main.py   .
COPY pyproject.toml .

# ── Non-root user ────────────────────────────────────────────
RUN useradd --create-home --shell /bin/bash fundit && \
    chown -R fundit:fundit /app
USER fundit

# ── Ports ────────────────────────────────────────────────────
EXPOSE 8000

# ── Healthcheck ───────────────────────────────────────────────
HEALTHCHECK --interval=20s --timeout=8s --start-period=60s --retries=4 \
    CMD curl -sf http://localhost:8000/health | python3 -c \
        "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('status')=='ok' else 1)" \
    || exit 1

# ── Entrypoint ────────────────────────────────────────────────
# Production: no --reload; workers=1 for small demo deployment
CMD ["uvicorn", "app.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--no-access-log"]
