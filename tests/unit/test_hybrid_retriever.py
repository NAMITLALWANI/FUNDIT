"""RRF arithmetic, candidate-set bounding, and BM25 behaviour on synthetic chunks."""

from unittest.mock import MagicMock

import pytest

from app.ingestion.models import DocumentChunk, SourceMetadata
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from app.retrieval.sparse_retriever import BM25Retriever


def make_chunk(chunk_id: str, text: str, fund_id=None) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        text=text,
        metadata=SourceMetadata(
            document_id=f"doc-{chunk_id}",
            document_type="SchemeSummary" if fund_id else "Regulatory",
            source="Synthetic",
            title=f"Title {chunk_id}",
            fund_id=fund_id,
        ),
    )


def test_rrf_arithmetic_and_ordering():
    a, b, c = make_chunk("a", "x"), make_chunk("b", "y"), make_chunk("c", "z")
    dense = [(a, 0.95, 1), (b, 0.85, 2)]
    sparse = [(b, 12.0, 1), (c, 7.0, 2)]
    fused = reciprocal_rank_fusion(dense, sparse, rrf_k=60)
    by_id = {r.chunk.chunk_id: r for r in fused}
    assert by_id["b"].fused_score == pytest.approx(1 / 62 + 1 / 61, abs=1e-6)
    assert by_id["a"].fused_score == pytest.approx(1 / 61, abs=1e-6)
    assert by_id["c"].fused_score == pytest.approx(1 / 62, abs=1e-6)
    assert [r.chunk.chunk_id for r in fused] == ["b", "a", "c"]
    # diagnostics preserved, raw scores never mixed
    assert by_id["b"].dense_rank == 2 and by_id["b"].sparse_rank == 1
    assert by_id["b"].dense_score == 0.85 and by_id["b"].sparse_score == 12.0
    assert by_id["a"].sparse_rank is None and by_id["c"].dense_rank is None


def test_hybrid_retriever_passes_fund_bounds_to_both_retrievers(settings):
    a = make_chunk("a", "alpha", fund_id="SYN-A")
    dense = MagicMock(spec=DenseRetriever)
    dense.retrieve.return_value = [(a, 0.9, 1)]
    dense.embedding_service = MagicMock(backend="hash-fallback")
    sparse = MagicMock(spec=BM25Retriever)
    sparse.retrieve.return_value = [(a, 5.0, 1)]

    retriever = HybridRetriever(dense, sparse, settings=settings)
    results, diag = retriever.retrieve("alpha", fund_ids=["SYN-A"], include_global=False, candidate_k=5)

    assert dense.retrieve.call_args.kwargs["fund_ids"] == ["SYN-A"]
    assert sparse.retrieve.call_args.kwargs["fund_ids"] == ["SYN-A"]
    assert dense.retrieve.call_args.kwargs["include_global"] is False
    assert len(results) == 1 and results[0].chunk.chunk_id == "a"
    assert diag.candidate_fund_count == 1 and diag.fused_candidates_count == 1
    assert diag.extra["rrf_k"] == settings.retrieval_rrf_k


def test_bm25_respects_candidate_funds_and_global_docs():
    chunks = [
        make_chunk("a", "liquid fund low interest rate risk parking", fund_id="SYN-A"),
        make_chunk("b", "liquid fund for parking cash short term", fund_id="SYN-B"),
        make_chunk("g", "SEBI liquid fund definition maturity up to 91 days"),
        # unrelated documents so that query terms have positive IDF
        make_chunk("d1", "small cap equity scheme volatility market capitalisation", fund_id="SYN-D"),
        make_chunk("d2", "gilt scheme government securities interest rate duration", fund_id="SYN-E"),
        make_chunk("d3", "ELSS tax saver statutory lock-in three years section 80C", fund_id="SYN-F"),
    ]
    bm25 = BM25Retriever()
    bm25.index(chunks)
    assert bm25.size == 6

    bounded = bm25.retrieve("liquid fund parking", top_k=10, fund_ids=["SYN-A"], include_global=True)
    ids = [c.chunk_id for c, _, _ in bounded]
    assert "b" not in ids and "a" in ids  # SYN-B is outside the candidate set
    assert [rank for _, _, rank in bounded] == list(range(1, len(bounded) + 1))

    # universe-wide (fund_id=None) documents are included only when include_global=True
    with_global = bm25.retrieve("maturity up to 91 days", top_k=10, fund_ids=["SYN-A"], include_global=True)
    assert [c.chunk_id for c, _, _ in with_global] == ["g"]
    without_global = bm25.retrieve("maturity up to 91 days", top_k=10, fund_ids=["SYN-A"], include_global=False)
    assert without_global == []

    strict = bm25.retrieve("liquid fund parking", top_k=10, fund_ids=["SYN-A"], include_global=False)
    assert [c.chunk_id for c, _, _ in strict] == ["a"]

    assert bm25.retrieve("zzzz unknown tokens", top_k=5) == []


def test_bm25_empty_index_returns_nothing():
    bm25 = BM25Retriever()
    bm25.index([])
    assert bm25.retrieve("anything") == []
