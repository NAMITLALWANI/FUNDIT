"""
Retrieval data models for search results and diagnostic scoring.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.ingestion.models import DocumentChunk


class RetrievalResult(BaseModel):
    """Retrieval candidate chunk with dense, sparse, and fused RRF metrics."""

    chunk: DocumentChunk
    dense_score: Optional[float] = Field(default=None, description="Cosine similarity from vector search")
    sparse_score: Optional[float] = Field(default=None, description="Raw BM25 lexical score")
    dense_rank: Optional[int] = Field(default=None, description="1-based dense retrieval rank position")
    sparse_rank: Optional[int] = Field(default=None, description="1-based sparse retrieval rank position")
    fused_score: float = Field(default=0.0, description="Reciprocal Rank Fusion unified score")


class RetrievalDiagnostics(BaseModel):
    """Diagnostic telemetry captured during hybrid retrieval."""

    dense_candidates_count: int = 0
    sparse_candidates_count: int = 0
    fused_candidates_count: int = 0
    candidate_fund_count: Optional[int] = Field(default=None, description="Size of the SQL candidate set used as filter")
    include_global_documents: bool = True
    embedding_backend: str = "unknown"
    dense_time_ms: float = 0.0
    sparse_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    extra: Dict[str, Any] = Field(default_factory=dict)


class IndexBuildReport(BaseModel):
    """Summary of an index build run."""

    chunks_indexed: int
    embedding_dimension: int
    embedding_backend: str
    collection_name: str
    bm25_documents: int
    embedding_cache_hits: int = 0
    embed_time_ms: float
    upsert_time_ms: float
    warnings: List[str] = Field(default_factory=list)
