"""Cross-encoder reranker: ordering, raw logits, within-query relative relevance, diagnostics."""

from app.ingestion.models import DocumentChunk, SourceMetadata
from app.reranking.reranker import CrossEncoderReranker
from app.retrieval.models import RetrievalResult


def _result(chunk_id: str, text: str, fused: float) -> RetrievalResult:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        text=text,
        metadata=SourceMetadata(document_id=f"doc-{chunk_id}", document_type="Regulatory", source="Synthetic", title=f"T {chunk_id}"),
    )
    return RetrievalResult(chunk=chunk, fused_score=fused)


class _FakeModel:
    """Stand-in cross-encoder returning fixed logits (keeps the test hermetic)."""

    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        return self.scores[: len(pairs)]


def _reranker(settings, scores=None) -> CrossEncoderReranker:
    rr = CrossEncoderReranker(settings=settings)
    rr._load_attempted = True
    rr._model = _FakeModel(scores) if scores is not None else None
    rr.load_time_ms = 1.0
    return rr


def test_rerank_orders_by_raw_logit_without_clamping(settings):
    results = [_result("a", "x", 0.03), _result("b", "y", 0.02), _result("c", "z", 0.01)]
    reranked, diag = _reranker(settings, scores=[-4.2, 7.9, 1.3]).rerank("q", results, top_k=3)
    assert [r.chunk.chunk_id for r in reranked] == ["b", "c", "a"]
    assert [r.rerank_position for r in reranked] == [1, 2, 3]
    assert reranked[0].reranker_score == 7.9 and reranked[-1].reranker_score == -4.2  # raw logits kept
    assert reranked[0].relative_relevance == 1.0 and reranked[-1].relative_relevance == 0.0
    assert diag.backend == "cross-encoder" and diag.top_score == 7.9 and diag.lowest_score == -4.2
    assert reranked[0].retrieval_result.fused_score == 0.02  # retrieval diagnostics preserved


def test_rerank_top_k_truncates(settings):
    results = [_result(str(i), f"text {i}", 0.1 - i * 0.01) for i in range(6)]
    reranked, diag = _reranker(settings, scores=[1, 6, 3, 5, 2, 4]).rerank("q", results, top_k=2)
    assert [r.chunk.chunk_id for r in reranked] == ["1", "3"]
    assert diag.candidate_count == 6 and diag.reranked_count == 2


def test_lexical_fallback_is_reported_in_diagnostics(settings):
    results = [_result("a", "liquid fund parking cash", 0.01), _result("b", "equity small cap growth", 0.02)]
    reranked, diag = _reranker(settings, scores=None).rerank("liquid fund", results)
    assert diag.backend == "lexical-fallback"
    assert reranked[0].chunk.chunk_id == "a"


def test_empty_input(settings):
    reranked, diag = _reranker(settings, scores=[]).rerank("q", [])
    assert reranked == [] and diag.candidate_count == 0
