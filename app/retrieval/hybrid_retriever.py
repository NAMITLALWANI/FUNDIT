"""
Hybrid retriever fusing dense semantic and sparse lexical search using Reciprocal Rank Fusion (RRF).

RRF_score(d) = sum_i 1 / (k + rank_i(d))

Only ranks are fused; raw cosine and BM25 scores are reported for diagnostics but never
compared directly because their distributions are not commensurable.
"""

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.ingestion.models import DocumentChunk
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.models import RetrievalDiagnostics, RetrievalResult
from app.retrieval.sparse_retriever import BM25Retriever

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    dense_results: Sequence[Tuple[DocumentChunk, float, int]],
    sparse_results: Sequence[Tuple[DocumentChunk, float, int]],
    rrf_k: int,
) -> List[RetrievalResult]:
    """Fuse two ranked lists into a single list ordered by RRF score (descending)."""
    candidate_map: Dict[str, Dict[str, Any]] = {}

    def _entry(chunk: DocumentChunk) -> Dict[str, Any]:
        return candidate_map.setdefault(
            chunk.chunk_id,
            {
                "chunk": chunk,
                "dense_score": None,
                "dense_rank": None,
                "sparse_score": None,
                "sparse_rank": None,
                "fused_score": 0.0,
            },
        )

    for chunk, score, rank in dense_results:
        entry = _entry(chunk)
        entry["dense_score"] = score
        entry["dense_rank"] = rank
        entry["fused_score"] += 1.0 / (rrf_k + rank)

    for chunk, score, rank in sparse_results:
        entry = _entry(chunk)
        entry["sparse_score"] = score
        entry["sparse_rank"] = rank
        entry["fused_score"] += 1.0 / (rrf_k + rank)

    ordered = sorted(candidate_map.values(), key=lambda item: (-item["fused_score"], item["chunk"].chunk_id))
    return [
        RetrievalResult(
            chunk=item["chunk"],
            dense_score=item["dense_score"],
            sparse_score=item["sparse_score"],
            dense_rank=item["dense_rank"],
            sparse_rank=item["sparse_rank"],
            fused_score=round(item["fused_score"], 6),
        )
        for item in ordered
    ]


class HybridRetriever:
    """Combines dense semantic search and BM25 sparse search with Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: BM25Retriever,
        rrf_k: Optional[int] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.settings = settings or get_settings()
        self.rrf_k = rrf_k or self.settings.retrieval_rrf_k

    def retrieve(
        self,
        query: str,
        fund_ids: Optional[Sequence[str]] = None,
        include_global: bool = True,
        dense_top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
        candidate_k: Optional[int] = None,
    ) -> Tuple[List[RetrievalResult], RetrievalDiagnostics]:
        """Run dense and sparse retrieval bounded to ``fund_ids`` and fuse the rankings."""
        t0 = time.perf_counter()
        k_dense = dense_top_k or self.settings.retrieval_dense_top_k
        k_sparse = sparse_top_k or self.settings.retrieval_sparse_top_k
        k_out = candidate_k or self.settings.retrieval_candidate_k

        t_dense = time.perf_counter()
        dense_results = self.dense_retriever.retrieve(
            query=query, top_k=k_dense, fund_ids=fund_ids, include_global=include_global
        )
        dense_ms = (time.perf_counter() - t_dense) * 1000.0

        t_sparse = time.perf_counter()
        sparse_results = self.sparse_retriever.retrieve(
            query=query, top_k=k_sparse, fund_ids=fund_ids, include_global=include_global
        )
        sparse_ms = (time.perf_counter() - t_sparse) * 1000.0

        fused = reciprocal_rank_fusion(dense_results, sparse_results, self.rrf_k)[:k_out]

        diagnostics = RetrievalDiagnostics(
            dense_candidates_count=len(dense_results),
            sparse_candidates_count=len(sparse_results),
            fused_candidates_count=len(fused),
            candidate_fund_count=len(fund_ids) if fund_ids is not None else None,
            include_global_documents=include_global,
            embedding_backend=self.dense_retriever.embedding_service.backend,
            dense_time_ms=round(dense_ms, 2),
            sparse_time_ms=round(sparse_ms, 2),
            retrieval_time_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            extra={"rrf_k": self.rrf_k, "dense_top_k": k_dense, "sparse_top_k": k_sparse},
        )
        logger.info(
            "Hybrid retrieval %.1fms: dense=%d sparse=%d fused=%d funds=%s",
            diagnostics.retrieval_time_ms, len(dense_results), len(sparse_results), len(fused),
            diagnostics.candidate_fund_count,
        )
        return fused, diagnostics
