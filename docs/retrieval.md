# Hybrid Retrieval & Reciprocal Rank Fusion (RRF) Deep Dive

### Motivation
Standard vector search (dense embeddings) excels at capturing conceptual semantics (e.g. associating *"machine learning"* with *"GPU tensor cores"* and *"deep learning"*), but frequently underperforms on exact lexical matches, part numbers, and numerical constraints (e.g. differentiating *"RTX 4070"* from *"RTX 4060"* or matching specific model numbers like *"GA403"*).

Conversely, lexical BM25 search excels at exact keyword hits but fails when queries use paraphrased terminology without exact token overlap.

The **AI Decision Engine V2** implements a multi-stage hybrid retrieval strategy:

1. **Deterministic Metadata Pre-Filtering**
2. **Dense Semantic Retrieval (Qdrant + BAAI/bge-small-en-v1.5)**
3. **Sparse Lexical Retrieval (Okapi BM25)**
4. **Reciprocal Rank Fusion (RRF)**
5. **Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)**

---

### Reciprocal Rank Fusion (RRF) Formula

Rather than attempting to linearly combine uncalibrated score distributions (e.g. cosine similarities $\in [-1, 1]$ vs unbounded BM25 scores $\in [0, \infty)$), Reciprocal Rank Fusion (RRF) combines candidates based on their ordinal rank positions:

$$RRF(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{k + \text{rank}_m(d)}$$

Where:
- $d$: Document chunk candidate.
- $m$: Retrieval method (Dense Semantic or Sparse BM25).
- $\text{rank}_m(d)$: 1-based rank position of document $d$ within method $m$.
- $k$: Smoothing constant (default $k=60$), preventing high-ranking outliers in one method from disproportionately dominating the fused ranking.

---

### Cross-Encoder vs Bi-Encoder Architecture

| Feature | Bi-Encoder (Dense Embedding) | Cross-Encoder (Reranker) |
| :--- | :--- | :--- |
| **Model** | `BAAI/bge-small-en-v1.5` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Input Format** | Encodes Query and Document separately into vectors | Encodes `(Query, Document)` pair jointly via cross-attention |
| **Complexity** | $O(N)$ vector dot products (fast, indexable in Qdrant) | $O(N \times L^2)$ transformer forward passes (compute heavy) |
| **Usage** | First-stage retrieval over entire corpus ($N=1000s$) | Second-stage reranking over candidate pool ($K=20-30$) |
| **Score Output** | Cosine similarity $[-1, 1]$ | Logit relevance score |

---

### Diagnostic Telemetry
When `debug=true` is passed to `POST /api/v1/decisions`, the API returns detailed retrieval diagnostics:

```json
{
  "retrieval_diagnostics": {
    "dense_candidates_count": 10,
    "sparse_candidates_count": 8,
    "fused_candidates_count": 12,
    "filters_applied": {
      "budget_max": 150000.0,
      "ram_min_gb": 16,
      "gpu_vendor": "NVIDIA"
    },
    "retrieval_time_ms": 14.2
  }
}
```
