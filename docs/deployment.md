# Containerization & Deployment Guide

### Deployment Architecture
The AI Decision Engine V2 is packaged as a containerized microservice that can be orchestrated alongside a standalone Qdrant vector database using Docker and Docker Compose.

---

### Running with Docker Compose

1. **Start Qdrant Vector Store:**
   ```bash
   docker compose up -d
   ```
   This exposes:
   - `http://localhost:6333`: Qdrant REST API
   - `http://localhost:6334`: Qdrant gRPC API

2. **Verify Qdrant Status:**
   ```bash
   curl -s http://localhost:6333/healthz
   ```

---

### Building and Running the API Container

1. **Build Container Image:**
   ```bash
   docker build -t ai-decision-engine-v2:latest .
   ```

2. **Run Application Container:**
   ```bash
   docker run -d \
     --name ai_decision_engine \
     -p 8000:8000 \
     -e QDRANT_URL=http://host.docker.internal:6333 \
     -e LLM_PROVIDER=gemini \
     -e GEMINI_API_KEY=your_api_key \
     ai-decision-engine-v2:latest
   ```

3. **Check Container Health:**
   ```bash
   curl -s http://localhost:8000/health
   curl -s http://localhost:8000/ready
   ```

---

### Production Tuning & Environment Configuration

| Variable | Default | Description |
| :--- | :--- | :--- |
| `APP_ENV` | `production` | Deployment mode (`development`, `production`) |
| `LOG_LEVEL` | `INFO` | Root logging level (`DEBUG`, `INFO`, `WARNING`) |
| `QDRANT_URL` | `http://localhost:6333` | Address of Qdrant instance |
| `EMBEDDING_DEVICE` | `cpu` | Device for embedding inference (`cpu`, `cuda`, `mps`) |
| `EMBEDDING_BATCH_SIZE` | `32` | Batch size for vector encoding |
| `RERANKER_BATCH_SIZE` | `16` | Batch size for cross-encoder inference |
| `LLM_PROVIDER` | `gemini` | LLM backend (`gemini` for Google Gemini, `mock` for offline testing) |
| `GEMINI_API_KEY` | `""` | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Configurable Gemini model identifier |
