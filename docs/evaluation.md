# Evaluation & Benchmark Harness

### Evaluation Strategy
To ensure the Decision-Augmented RAG system maintains both statistical ranking quality and deterministic constraint adherence, the engine includes an automated benchmarking testbed in `app/evaluation/` executed via `scripts/evaluate.py`.

---

### Key Information Retrieval (IR) Metrics

1. **Recall@K:**
   $$\text{Recall@K} = \frac{|\text{Retrieved Top K} \cap \text{Relevant Items}|}{|\text{Relevant Items}|}$$
   Measures the fraction of relevant mutual funds retrieved within the top $K$ candidates.

2. **Precision@K:**
   $$\text{Precision@K} = \frac{|\text{Retrieved Top K} \cap \text{Relevant Items}|}{K}$$
   Measures the fraction of top $K$ retrieved items that are genuinely relevant.

3. **Mean Reciprocal Rank (MRR):**
   $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
   Measures how high the first relevant candidate appears in the ranking.

4. **Normalized Discounted Cumulative Gain (NDCG@K):**
   $$\text{DCG@K} = \sum_{i=1}^{K} \frac{r_i}{\log_2(i + 1)}, \quad \text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$
   Measures ranking quality with position-based logarithmic discounting.

---

### Generation & Faithfulness Metrics

1. **Citation Coverage:**
   $$\text{Citation Coverage} = \frac{\text{Count of Valid Referenced Citations}}{\text{Total Citations Injected}}$$
   Measures whether generated claims contain numbered citation references (`[1]`, `[2]`).

2. **Constraint Faithfulness:**
   $$\text{Constraint Faithfulness} = \frac{\text{Matching Extracted Constraints}}{\text{Total Expected Constraints}}$$
   Measures whether natural language numbers (e.g. ₹1.5L -> 150000, 16GB -> 16) were extracted with 100% precision.

---

### Running Evaluation

Run the CLI runner:
```bash
python scripts/evaluate.py --output data/evaluation/report.json
```

Sample output:
```text
================================================================================
AI DECISION ENGINE V2 - INFORMATION RETRIEVAL & GROUNDING EVALUATION REPORT
================================================================================
[RETRIEVAL METRICS]
  Recall@1:    0.7000
  Recall@3:    0.9000
  Recall@5:    0.9000
  Precision@1: 0.7000
  Precision@3: 0.5333
  Precision@5: 0.3600
  MRR:         1.0000
  NDCG@3:      0.9408
  NDCG@5:      0.9408

[GENERATION & DECISION METRICS]
  Citation Coverage:       0.7000
  Constraint Faithfulness: 1.0000
  Answer Relevance:        1.0000
================================================================================
```
