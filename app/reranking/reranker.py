"""
Cross-encoder reranking service for candidate evidence chunks.

Design notes
* Raw cross-encoder logits are used for ordering only. No fixed-range clamp such as
  ``(score + 5) / 10`` is applied because logit ranges are model-specific.
* ``relative_relevance`` is a within-query min-max rescaling used only to compare chunks of the
  same query (e.g. when attaching evidence to funds). It is not a calibrated probability.
* A lexical fallback exists so the service and tests stay runnable when the model cannot be
  loaded (no network / CI). The backend in use is always reported in diagnostics.
"""

import time
from typing import List, Optional, Tuple

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.reranking.models import RerankedResult, RerankingDiagnostics
from app.retrieval.models import RetrievalResult
from app.retrieval.sparse_retriever import tokenize_text

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Reranks retrieved candidate pools using a transformer cross-encoder."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        settings: Optional[Settings] = None,
        allow_lexical_fallback: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.reranker_model
        self.device = device or self.settings.reranker_device
        self.batch_size = batch_size or self.settings.reranker_batch_size
        self.allow_lexical_fallback = allow_lexical_fallback
        self._model = None
        self._load_attempted = False
        self.load_time_ms: Optional[float] = None

    @property
    def backend(self) -> str:
        if not self._load_attempted:
            return "not-loaded"
        return "cross-encoder" if self._model is not None else "lexical-fallback"

    def warm_up(self) -> None:
        self._load_model()

    def _load_model(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        t0 = time.perf_counter()
        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading CrossEncoder reranker '{self.model_name}' on device '{self.device}'")
            self._model = CrossEncoder(self.model_name, device=self.device, max_length=512)
        except Exception as e:  # noqa: BLE001 - degrade gracefully, report backend in diagnostics
            if not self.allow_lexical_fallback:
                raise
            logger.warning(
                f"Failed to load CrossEncoder '{self.model_name}': {e}. Using lexical-overlap fallback "
                "(ordering quality degraded; offline/test use only)."
            )
            self._model = None
        self.load_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    @staticmethod
    def _lexical_scores(query: str, results: List[RetrievalResult]) -> List[float]:
        q_tokens = set(tokenize_text(query))
        scores: List[float] = []
        for r in results:
            c_tokens = set(tokenize_text(r.chunk.text))
            overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
            scores.append(round(overlap + r.fused_score, 6))
        return scores

    def rerank(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> Tuple[List[RerankedResult], RerankingDiagnostics]:
        """Score query/chunk pairs and return the top-k ordered by relevance."""
        t0 = time.perf_counter()
        k = top_k or self.settings.rerank_top_k
        if not retrieval_results:
            return [], RerankingDiagnostics(
                candidate_count=0, reranked_count=0, backend=self.backend, model_name=self.model_name,
                reranking_time_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        cold_start = not self._load_attempted
        self._load_model()
        if self._model is not None:
            pairs = [(query, f"{r.chunk.metadata.title}: {r.chunk.text}") for r in retrieval_results]
            raw_scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
            scores = [float(s) for s in raw_scores]
        else:
            scores = self._lexical_scores(query, retrieval_results)

        scored = sorted(zip(retrieval_results, scores), key=lambda item: item[1], reverse=True)[:k]
        hi = scored[0][1]
        lo = scored[-1][1]
        span = hi - lo

        reranked = [
            RerankedResult(
                retrieval_result=res,
                reranker_score=round(score, 4),
                rerank_position=pos,
                relative_relevance=round((score - lo) / span, 4) if span > 0 else 1.0,
            )
            for pos, (res, score) in enumerate(scored, start=1)
        ]

        diagnostics = RerankingDiagnostics(
            candidate_count=len(retrieval_results),
            reranked_count=len(reranked),
            top_score=reranked[0].reranker_score,
            lowest_score=reranked[-1].reranker_score,
            backend=self.backend,
            model_name=self.model_name,
            model_load_time_ms=self.load_time_ms if cold_start else None,
            reranking_time_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )
        logger.info(
            "Reranking %.1fms (%s): %d candidates -> %d chunks",
            diagnostics.reranking_time_ms, diagnostics.backend, len(retrieval_results), len(reranked),
        )
        return reranked, diagnostics
