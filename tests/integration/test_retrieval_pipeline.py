"""
Retrieval -> rerank -> evidence validation -> decision, end to end on synthetic chunks.

Uses the in-memory Qdrant client and whatever embedding backend is available (the hash fallback
is acceptable here: the test checks plumbing and bounding, not semantic quality).
"""

from datetime import date

import pytest

from app.decision.engine import DecisionEngine
from app.decision.models import CandidateInput
from app.evidence.validator import EvidenceValidator
from app.ingestion.models import DocumentChunk, SourceMetadata
from app.reranking.reranker import CrossEncoderReranker
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.retriever import EvidenceRetriever
from app.retrieval.vector_store import QdrantVectorStore
from tests.conftest import AS_OF, make_fund, make_metrics, make_query


def _chunk(cid: str, fund_id, section: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=cid,
        document_id=f"{fund_id or 'reg'}-doc",
        text=f"{section}: {text}",
        metadata=SourceMetadata(
            document_id=f"{fund_id or 'reg'}-doc",
            document_type="SchemeSummary" if fund_id else "Regulatory",
            source="Synthetic",
            source_url="https://example.invalid/doc",
            title=f"{fund_id or 'Regulation'} summary",
            fund_id=fund_id,
            publication_date=date(2026, 9, 1),
            section=section,
        ),
    )


CHUNKS = [
    _chunk("a0", "SYN-A", "Investment objective", "long-term capital appreciation through a diversified flexi cap equity portfolio"),
    _chunk("a1", "SYN-A", "Costs and size", "The total expense ratio of the Direct Plan was 0.50% as of 2026-09-30."),
    _chunk("b0", "SYN-B", "Investment objective", "capital appreciation by investing in large cap equity stocks"),
    _chunk("c0", "SYN-C", "Investment objective", "income by investing in short duration debt and money market instruments"),
    _chunk("z0", "SYN-Z", "Investment objective", "equity growth fund that is not in the candidate set"),
    _chunk("g0", None, "Flexi Cap Fund", "SEBI defines flexi cap funds as dynamic equity schemes investing across market caps"),
]


@pytest.fixture(scope="module")
def retriever():
    settings_kwargs = dict(_env_file=None, qdrant_url=None, embedding_cache_path=None)
    from app.core.config import Settings

    settings = Settings(**settings_kwargs)
    embedding = EmbeddingService(settings=settings)
    store = QdrantVectorStore(settings=settings, in_memory=True)
    r = EvidenceRetriever(embedding_service=embedding, vector_store=store, settings=settings)
    r.build_index(CHUNKS)
    return r


def test_index_build_reports_backend_and_counts(retriever):
    assert retriever.is_ready()
    assert retriever.sparse_retriever.size == len(CHUNKS)
    assert retriever.vector_store.count() == len(CHUNKS)
    assert retriever.embedding_service.backend in {"sentence-transformers", "hash-fallback"}


def test_retrieval_is_bounded_to_candidate_funds(retriever):
    results, diag = retriever.retrieve("flexi cap equity capital appreciation", fund_ids=["SYN-A", "SYN-B"], include_global=True)
    assert results
    returned_funds = {r.chunk.metadata.fund_id for r in results}
    assert "SYN-Z" not in returned_funds and "SYN-C" not in returned_funds
    assert returned_funds <= {"SYN-A", "SYN-B", None}
    assert diag.candidate_fund_count == 2
    assert all(r.dense_rank is not None or r.sparse_rank is not None for r in results)


def test_full_pipeline_to_decision(retriever, settings):
    funds = [
        make_fund("SYN-A", "Synthetic A", expense_ratio=0.5),
        make_fund("SYN-B", "Synthetic B", expense_ratio=0.9),
        make_fund("SYN-C", "Synthetic C", category="Debt", sub_category="Short Duration Fund", risk_level="Low to Moderate", expense_ratio=0.4),
    ]
    metrics = {
        "SYN-A": make_metrics("SYN-A", sharpe_3y=1.0),
        "SYN-B": make_metrics("SYN-B", sharpe_3y=0.6),
        "SYN-C": make_metrics("SYN-C", sharpe_3y=0.8, volatility_3y=0.02, max_drawdown_3y=-0.01),
    }
    query = make_query("flexi cap equity fund for long-term capital appreciation")
    ids = [f.fund_id for f in funds]

    results, _ = retriever.retrieve(query.rewritten_query, fund_ids=ids)
    reranked, rdiag = CrossEncoderReranker(settings=settings).rerank(query.rewritten_query, results, top_k=6)
    assert rdiag.backend in {"cross-encoder", "lexical-fallback"}

    bundles, global_ctx, vreport = EvidenceValidator(settings=settings).validate(reranked, funds, today=AS_OF)
    assert vreport.unbound_discarded == 0  # SYN-Z was never retrieved thanks to bounding
    assert all(e.chunk_id in {c.chunk_id for c in CHUNKS} for b in bundles.values() for e in b.evidence)

    candidates = [CandidateInput(fund=f, metrics=metrics[f.fund_id], evidence=bundles[f.fund_id].evidence) for f in funds]
    result = DecisionEngine(settings=settings).decide(query, candidates, today=AS_OF)

    assert result.candidates_considered == 3
    assert {c.fund_id for c in result.ranked} == set(ids)
    if not result.abstained:
        assert result.winner is not None and result.winner.evidence_available
        assert all(e.source_url for e in result.winner.evidence)
    else:
        assert result.abstention_reason
