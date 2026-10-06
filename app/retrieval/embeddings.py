"""
Embedding service abstraction supporting Sentence Transformers and deterministic embeddings.
"""

from typing import List, Optional, Union
import numpy as np
from app.core.config import Settings, get_settings
from app.core.exceptions import ConfigurationError
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Provides vector embeddings for queries and document chunks."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        normalize: Optional[bool] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.embedding_model
        self.device = device or self.settings.embedding_device
        self.batch_size = batch_size or self.settings.embedding_batch_size
        self.normalize = normalize if normalize is not None else self.settings.embedding_normalize
        self._model = None
        self._dimension: Optional[int] = None
        self._load_attempted = False
        self.load_time_ms: Optional[float] = None

    @property
    def backend(self) -> str:
        """'sentence-transformers' when the real model is loaded, otherwise 'hash-fallback'."""
        if not self._load_attempted:
            return "not-loaded"
        return "sentence-transformers" if self._model is not None else "hash-fallback"

    def warm_up(self) -> None:
        """Eagerly load the model (used for optional startup preloading and cold-start measurement)."""
        self._load_model()

    def _load_model(self) -> None:
        """Lazy load SentenceTransformer model."""
        if self._load_attempted:
            return
        self._load_attempted = True
        import time

        t0 = time.perf_counter()
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(
                f"Loading SentenceTransformer embedding model '{self.model_name}' on device '{self.device}'"
            )
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._dimension = self._model.get_embedding_dimension()
        except Exception as e:  # noqa: BLE001 - model loading failures must not crash the service
            logger.warning(
                f"Failed to load SentenceTransformer '{self.model_name}': {e}. "
                "Falling back to deterministic hash embeddings (retrieval quality degraded; offline/test use only)."
            )
            self._model = None
            self._dimension = 384
        self.load_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    @property
    def dimension(self) -> int:
        """Return the vector embedding dimension."""
        if self._dimension is None:
            self._load_model()
        return self._dimension or 384

    def _deterministic_mock_embed(self, text: str, dim: int = 384) -> List[float]:
        """Fast, reproducible pseudorandom vector for offline or test environments."""
        import hashlib

        hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
        # Seed numpy RNG with integer from hash
        seed = int.from_bytes(hash_bytes[:4], byteorder="big")
        rng = np.random.RandomState(seed)
        vec = rng.randn(dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string into a vector."""
        text = query.strip()
        if not text:
            return [0.0] * self.dimension

        self._load_model()
        if self._model is not None:
            embedding = self._model.encode(
                text,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
            return (
                embedding.tolist()
                if isinstance(embedding, np.ndarray)
                else list(embedding)
            )
        return self._deterministic_mock_embed(text, self.dimension)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document texts into vectors."""
        if not texts:
            return []

        cleaned_texts = [t.strip() if t.strip() else " " for t in texts]
        self._load_model()

        if self._model is not None:
            embeddings = self._model.encode(
                cleaned_texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
            return (
                embeddings.tolist()
                if isinstance(embeddings, np.ndarray)
                else [list(e) for e in embeddings]
            )

        return [self._deterministic_mock_embed(t, self.dimension) for t in cleaned_texts]
