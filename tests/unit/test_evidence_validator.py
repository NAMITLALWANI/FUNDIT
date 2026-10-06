"""Evidence validation: provenance binding, candidate-set bounding and structured consistency."""

from datetime import date

from app.evidence.validator import EvidenceValidator
from app.ingestion.models import DocumentChunk, SourceMetadata
from app.reranking.models import RerankedResult
from app.retrieval.models import RetrievalResult
from tests.conftest import AS_OF, make_fund


def _reranked(chunk_id: str, fund_id, text: str, position: int, relevance: float = 0.9, pub: date = AS_OF) -> RerankedResult:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id=f"{fund_id or 'global'}-doc",
        text=text,
        metadata=SourceMetadata(
            document_id=f"{fund_id or 'global'}-doc",
            document_type="SchemeSummary" if fund_id else "Regulatory",
            source="Synthetic source",
            source_url="https://example.invalid/doc",
            title="Synthetic doc",
            fund_id=fund_id,
            publication_date=pub,
            chunk_index=0,
        ),
    )
    return RerankedResult(
        retrieval_result=RetrievalResult(chunk=chunk, fused_score=0.01),
        reranker_score=1.0,
        rerank_position=position,
        relative_relevance=relevance,
    )


def test_chunks_bound_to_candidates_and_out_of_scope_discarded(settings):
    validator = EvidenceValidator(settings=settings)
    candidates = [make_fund("SYN-A", "A", expense_ratio=0.5), make_fund("SYN-B", "B")]
    reranked = [
        _reranked("a-c0", "SYN-A", "Investment objective: synthetic growth.", 1),
        _reranked("z-c0", "SYN-Z", "Chunk about a fund that is not a candidate.", 2),
        _reranked("g-c0", None, "SEBI riskometer levels are Low to Very High.", 3),
        _reranked("b-c0", "SYN-B", "Scheme classification and risk text.", 4),
    ]
    bundles, global_ctx, report = validator.validate(reranked, candidates, today=AS_OF)
    assert bundles["SYN-A"].count == 1 and bundles["SYN-B"].count == 1
    assert "SYN-Z" not in bundles
    assert report.unbound_discarded == 1
    assert len(global_ctx) == 1 and report.global_context == 1
    assert report.funds_with_evidence == 2
    assert bundles["SYN-A"].evidence[0].chunk_id == "a-c0"
    assert bundles["SYN-A"].evidence[0].source_url == "https://example.invalid/doc"


def test_structured_conflict_marks_chunk_conflicting(settings):
    validator = EvidenceValidator(settings=settings)
    fund = make_fund("SYN-A", "A", expense_ratio=0.5)
    conflicting = _reranked("a-c1", "SYN-A", "Costs and size: The total expense ratio of the Direct Plan was 1.50% as of 2026-09-30.", 1)
    consistent = _reranked("a-c2", "SYN-A", "Costs and size: The total expense ratio of the Direct Plan was 0.50% as of 2026-09-30.", 2)
    bundles, _, report = validator.validate([conflicting, consistent], [fund], today=AS_OF)
    assert report.conflicting == 1 and report.verified == 1
    assert bundles["SYN-A"].count == 1 and bundles["SYN-A"].evidence[0].chunk_id == "a-c2"
    assert "expense ratio 1.50%" in bundles["SYN-A"].conflicts[0]


def test_risk_level_conflict_detected(settings):
    validator = EvidenceValidator(settings=settings)
    fund = make_fund("SYN-A", "A", risk_level="Moderate")
    chunk = _reranked("a-c3", "SYN-A", "The SEBI Risk-o-meter level disclosed for the scheme is Very High. The data refers to the Direct Plan.", 1)
    bundles, _, report = validator.validate([chunk], [fund], today=AS_OF)
    assert report.conflicting == 1 and bundles["SYN-A"].count == 0


def test_stale_documents_counted(settings):
    validator = EvidenceValidator(settings=settings)
    fund = make_fund("SYN-A", "A")
    old = _reranked("a-old", "SYN-A", "Investment objective text.", 1, pub=date(2019, 1, 1))
    _, _, report = validator.validate([old], [fund], today=AS_OF)
    assert report.stale == 1 and report.verified == 1


def test_no_evidence_for_fund_yields_empty_bundle_not_default(settings):
    validator = EvidenceValidator(settings=settings)
    funds = [make_fund("SYN-A", "A"), make_fund("SYN-B", "B")]
    bundles, _, report = validator.validate([_reranked("a-c0", "SYN-A", "text", 1)], funds, today=AS_OF)
    assert bundles["SYN-B"].count == 0 and bundles["SYN-B"].evidence == []
    assert report.funds_with_evidence == 1
