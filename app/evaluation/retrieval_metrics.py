"""
Standard Information Retrieval (IR) evaluation metrics calculation.
"""

import math
from typing import List, Set


def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Calculate Recall@K: proportion of relevant items retrieved in top K."""
    if not relevant_ids:
        return 1.0  # If no relevant items expected, treat as satisfied
    top_k_set = set(retrieved_ids[:k])
    rel_set = set(relevant_ids)
    hits = len(top_k_set.intersection(rel_set))
    return hits / float(len(rel_set))


def calculate_precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Calculate Precision@K: proportion of retrieved top K items that are relevant."""
    if k <= 0:
        return 0.0
    if not relevant_ids:
        return 1.0 if not retrieved_ids else 0.0
    top_k_items = retrieved_ids[:k]
    rel_set = set(relevant_ids)
    hits = sum(1 for item in top_k_items if item in rel_set)
    return hits / float(k)


def calculate_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Calculate Mean Reciprocal Rank (MRR): 1 / (rank of first relevant item)."""
    if not relevant_ids:
        return 1.0
    rel_set = set(relevant_ids)
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in rel_set:
            return 1.0 / float(rank)
    return 0.0


def calculate_ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance."""
    if not relevant_ids or k <= 0:
        return 1.0

    rel_set = set(relevant_ids)
    dcg = 0.0
    for i, item in enumerate(retrieved_ids[:k]):
        if item in rel_set:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because 1-based rank + 1

    # Ideal DCG (all relevant items at top positions up to k)
    ideal_hits = min(k, len(rel_set))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg
