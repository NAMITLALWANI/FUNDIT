"""
Reranking data models.
"""

from typing import Optional

from pydantic import BaseModel, Field

from app.ingestion.models import DocumentChunk
from app.retrieval.models import RetrievalResult


class RerankedResult(BaseModel):
    """Reranked candidate incorporating cross-encoder relevance scoring.

    ``reranker_score`` is the raw cross-encoder logit. It is only meaningful for ordering
    candidates within the same query; it is not a probability and is not comparable across queries.
    """

    retrieval_result: RetrievalResult
    reranker_score: float = Field(description="Raw cross-encoder relevance logit (ordering only)")
    rerank_position: int = Field(description="1-based position post-reranking")
    relative_relevance: Optional[float] = Field(
        default=None,
        description="Within-query min-max normalised position in [0,1]; 1 = best chunk for this query",
    )

    @property
    def chunk(self) -> DocumentChunk:
        return self.retrieval_result.chunk


class RerankingDiagnostics(BaseModel):
    """Telemetry captured during cross-encoder reranking."""

    candidate_count: int
    reranked_count: int
    top_score: Optional[float] = None
    lowest_score: Optional[float] = None
    backend: str = "unknown"
    model_name: Optional[str] = None
    model_load_time_ms: Optional[float] = Field(default=None, description="Populated on the cold-start call only")
    reranking_time_ms: float = 0.0
