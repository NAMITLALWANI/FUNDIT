"""
Evidence retriever bridging the SQL candidate set and the hybrid chunk index.

Responsibilities
* Build/refresh the BM25 index from the chunks stored in the relational database.
* Embed and upsert chunks into Qdrant (``build_index``) and report what was indexed.
* Serve bounded hybrid retrieval: only chunks belonging to the structured candidate funds
  (plus universe-wide regulatory documents) can be returned.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.core.config import Settings, get_settings
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.ingestion.models import DocumentChunk
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.models import IndexBuildReport, RetrievalDiagnostics, RetrievalResult
from app.retrieval.sparse_retriever import BM25Retriever
from app.retrieval.vector_store import QdrantVectorStore

logger = get_logger(__name__)


class EvidenceRetriever:
    """Coordinates indexing and bounded hybrid retrieval over fund document chunks."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[QdrantVectorStore] = None,
        sparse_retriever: Optional[BM25Retriever] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedding_service = embedding_service or EmbeddingService(settings=self.settings)
        self.vector_store = vector_store or QdrantVectorStore(settings=self.settings)
        self.sparse_retriever = sparse_retriever or BM25Retriever()
        self.dense_retriever = DenseRetriever(self.embedding_service, self.vector_store)
        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.dense_retriever,
            sparse_retriever=self.sparse_retriever,
            settings=self.settings,
        )
        self.indexed_chunk_count = 0
        self.cache_path: Optional[Path] = (
            Path(self.settings.embedding_cache_path) if self.settings.embedding_cache_path else None
        )

    @staticmethod
    def embedding_text(chunk: DocumentChunk) -> str:
        return f"{chunk.metadata.title}: {chunk.text}"

    # ------------------------------------------------------------ embedding cache

    def _load_cache(self) -> Dict[str, List[float]]:
        """Load cached embeddings keyed by chunk text hash (valid for the configured model only)."""
        if self.cache_path is None or not self.cache_path.exists():
            return {}
        try:
            data = np.load(self.cache_path, allow_pickle=False)
            if str(data["model_name"]) != self.embedding_service.model_name:
                return {}
            keys = [str(k) for k in data["keys"]]
            return {k: v.tolist() for k, v in zip(keys, data["vectors"])}
        except (OSError, ValueError, KeyError) as e:
            logger.warning(f"Embedding cache unreadable ({e}); recomputing embeddings")
            return {}

    def _save_cache(self, cache: Dict[str, List[float]]) -> None:
        if self.cache_path is None or not cache:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(cache.keys())
        np.savez_compressed(
            self.cache_path,
            model_name=np.array(self.embedding_service.model_name),
            keys=np.array(keys),
            vectors=np.array([cache[k] for k in keys], dtype=np.float32),
        )

    @staticmethod
    def _cache_key(chunk: DocumentChunk, text: str) -> str:
        import hashlib

        return f"{chunk.chunk_id}:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]}"

    def embed_chunks(self, chunks: Sequence[DocumentChunk]) -> Tuple[List[List[float]], int]:
        """Embed chunks, reusing cached vectors for unchanged text. Returns (vectors, cache_hits)."""
        texts = [self.embedding_text(c) for c in chunks]
        keys = [self._cache_key(c, t) for c, t in zip(chunks, texts)]
        cache = self._load_cache() if self.embedding_service.backend != "hash-fallback" else {}
        missing = [i for i, k in enumerate(keys) if k not in cache]
        if missing:
            new_vectors = self.embedding_service.embed_documents([texts[i] for i in missing])
            for i, vec in zip(missing, new_vectors):
                cache[keys[i]] = vec
            if self.embedding_service.backend == "sentence-transformers":
                self._save_cache({k: cache[k] for k in keys})
        return [cache[k] for k in keys], len(keys) - len(missing)

    def build_sparse_index(self, chunks: Sequence[DocumentChunk]) -> None:
        """(Re)build the in-memory BM25 index. Cheap for a few hundred chunks; done at startup."""
        self.sparse_retriever.index(list(chunks))

    def build_index(self, chunks: Sequence[DocumentChunk], rebuild: bool = False) -> IndexBuildReport:
        """Embed chunks, upsert into Qdrant and build BM25. Idempotent (point IDs are deterministic)."""
        chunk_list = list(chunks)
        warnings: List[str] = []
        if not chunk_list:
            raise RetrievalError("No chunks provided for indexing; run scripts/ingest_funds.py first")

        self.build_sparse_index(chunk_list)

        self.embedding_service.warm_up()
        t0 = time.perf_counter()
        embeddings, cache_hits = self.embed_chunks(chunk_list)
        embed_ms = (time.perf_counter() - t0) * 1000.0
        if self.embedding_service.backend != "sentence-transformers":
            warnings.append("Embedding model unavailable; deterministic hash embeddings were used (offline/test quality only)")

        dimension = len(embeddings[0])
        t1 = time.perf_counter()
        if rebuild:
            self.vector_store.recreate_collection(dimension=dimension)
        self.vector_store.upsert_chunks(chunk_list, embeddings)
        upsert_ms = (time.perf_counter() - t1) * 1000.0
        self.indexed_chunk_count = len(chunk_list)

        return IndexBuildReport(
            chunks_indexed=len(chunk_list),
            embedding_dimension=dimension,
            embedding_backend=self.embedding_service.backend,
            collection_name=self.vector_store.collection_name,
            bm25_documents=self.sparse_retriever.size,
            embedding_cache_hits=cache_hits,
            embed_time_ms=round(embed_ms, 2),
            upsert_time_ms=round(upsert_ms, 2),
            warnings=warnings,
        )

    def is_ready(self) -> bool:
        """True when both the dense collection and the BM25 index contain data."""
        try:
            return self.sparse_retriever.size > 0 and self.vector_store.count() > 0
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Readiness check failed: {e}")
            return False

    def retrieve(
        self,
        query: str,
        fund_ids: Optional[Sequence[str]] = None,
        include_global: bool = True,
        candidate_k: Optional[int] = None,
    ) -> Tuple[List[RetrievalResult], RetrievalDiagnostics]:
        """Hybrid retrieval bounded to the candidate funds."""
        if not query or not query.strip():
            raise RetrievalError("Cannot retrieve evidence for an empty query")
        return self.hybrid_retriever.retrieve(
            query=query,
            fund_ids=fund_ids,
            include_global=include_global,
            candidate_k=candidate_k,
        )
