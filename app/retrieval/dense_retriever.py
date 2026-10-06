"""
Dense semantic vector retriever using embeddings and Qdrant, bounded to candidate funds.
"""

from typing import List, Optional, Sequence, Tuple

from app.core.logging import get_logger
from app.ingestion.models import DocumentChunk
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import QdrantVectorStore, build_fund_filter

logger = get_logger(__name__)


class DenseRetriever:
    """Executes dense vector similarity search over the Qdrant collection."""

    def __init__(self, embedding_service: EmbeddingService, vector_store: QdrantVectorStore) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        fund_ids: Optional[Sequence[str]] = None,
        include_global: bool = True,
    ) -> List[Tuple[DocumentChunk, float, int]]:
        """
        Embed the query and return ``(chunk, cosine_score, rank)`` tuples.

        When ``fund_ids`` is provided the search is restricted (via payload filter) to chunks of
        those funds plus universe-wide documents, so retrieval never surfaces out-of-scope funds.
        """
        if not query.strip():
            return []

        query_vector = self.embedding_service.embed_query(query)
        raw_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            query_filter=build_fund_filter(fund_ids, include_global=include_global),
        )
        return [(chunk, score, rank) for rank, (chunk, score) in enumerate(raw_results, start=1)]
