"""
LangGraph workflow nodes — each node is a pure function: DecisionState → DecisionState.

Node responsibilities:
  analyze_query       : Run QueryAnalyzer → DecisionQuery; block on blocking ambiguities
  retrieve_candidates : SQL FundFilter → bounded candidate fund_ids
  retrieve_evidence   : Hybrid RAG bounded to SQL candidates
  rerank_evidence     : Cross-encoder reranking of retrieved chunks
  validate_evidence   : EvidenceValidator → EvidenceBundle per candidate
  run_decision_engine : ScoringEngine + confidence gate → DecisionResult
  generate_response   : GenerationService → DecisionResponse
  handle_abstention   : Build abstention DecisionResponse without LLM
"""

import time
from datetime import date
from typing import Any, Dict, List, Optional

from app.core.config import Settings
from app.core.logging import get_logger
from app.db.repository import FundFilter, FundRepository
from app.decision.engine import DecisionEngine
from app.decision.models import CandidateInput
from app.evidence.validator import EvidenceValidator
from app.generation.generator import GenerationService
from app.generation.models import DecisionTraceStep
from app.ingestion.models import FundRecord
from app.query.analyzer import QueryAnalyzer
from app.reranking.reranker import CrossEncoderReranker
from app.retrieval.retriever import EvidenceRetriever
from app.workflow.state import DecisionState

logger = get_logger(__name__)


def _ts() -> float:
    return time.perf_counter()


def _trace(state: DecisionState, stage: str, summary: str, details: Optional[Dict[str, Any]] = None) -> None:
    trace = state.get("decision_trace", [])
    trace.append(DecisionTraceStep(stage=stage, summary=summary, details=details or {}))
    state["decision_trace"] = trace  # type: ignore[assignment]


# ===================================================================== nodes


def analyze_query(
    state: DecisionState,
    analyzer: QueryAnalyzer,
) -> DecisionState:
    """Node 1: Parse raw query into a typed DecisionQuery."""
    t0 = _ts()
    raw = state.get("raw_query", "")
    try:
        decision_query = analyzer.analyze(raw)
        state["decision_query"] = decision_query
        blocking = decision_query.blocking_ambiguities
        elapsed = (_ts() - t0) * 1000
        timing = state.get("timing", {})
        timing["query_analysis_ms"] = round(elapsed, 2)
        state["timing"] = timing  # type: ignore[assignment]

        if blocking:
            state["query_error"] = "; ".join(a.message for a in blocking)
            _trace(state, "query_analysis", f"Blocking ambiguities found: {len(blocking)} issue(s).", {
                "blocking": [a.message for a in blocking],
                "intent": decision_query.intent,
            })
        else:
            _trace(state, "query_analysis", f"Query analyzed: intent={decision_query.intent}.", {
                "intent": decision_query.intent,
                "constraints": decision_query.constraints.active_fields(),
                "preferences": list(decision_query.preferences.weights),
                "targets": len(decision_query.comparison_targets),
                "ambiguities": len(decision_query.ambiguities),
            })
    except Exception as exc:  # noqa: BLE001
        state["query_error"] = str(exc)
        _trace(state, "query_analysis", f"Query analysis failed: {exc}", {"error": str(exc)})
    return state


def retrieve_candidates(
    state: DecisionState,
    repository: FundRepository,
    settings: Settings,
) -> DecisionState:
    """Node 2: SQL-based structured candidate retrieval → bounded fund_id list."""
    t0 = _ts()
    query = state.get("decision_query")
    if query is None:
        state["candidate_fund_ids"] = []
        return state

    c = query.constraints
    # Comparison intent: use exactly the named fund IDs
    if query.comparison_targets and query.resolved_target_ids:
        fund_ids = query.resolved_target_ids
        state["candidate_fund_ids"] = fund_ids
        state["sql_candidates_count"] = len(fund_ids)
        _trace(state, "candidate_retrieval", f"Comparison mode: using {len(fund_ids)} explicit fund IDs.", {
            "fund_ids": fund_ids,
        })
        return state

    fund_filter = FundFilter(
        sip_amount=c.sip_amount,
        lump_sum_amount=c.lump_sum_amount,
        max_expense_ratio=c.max_expense_ratio,
        max_risk_level=c.max_risk_level,
        categories=c.categories,
        sub_categories=c.sub_categories,
        min_aum_crores=c.min_aum_crores,
        max_lock_in_years=c.max_lock_in_years,
        plan_type=c.plan_type,
        amcs=c.required_amcs,
        limit=settings.max_structured_candidates,
    )
    rows = repository.search_funds(fund_filter)
    fund_ids = [r.fund_id for r in rows]

    elapsed = (_ts() - t0) * 1000
    timing = state.get("timing", {})
    timing["candidate_retrieval_ms"] = round(elapsed, 2)
    state["timing"] = timing  # type: ignore[assignment]

    state["candidate_fund_ids"] = fund_ids
    state["sql_candidates_count"] = len(fund_ids)
    _trace(state, "candidate_retrieval", f"SQL filter produced {len(fund_ids)} candidate(s).", {
        "filter": fund_filter.model_dump(exclude_none=True),
        "count": len(fund_ids),
    })
    return state


def retrieve_evidence(
    state: DecisionState,
    retriever: EvidenceRetriever,
) -> DecisionState:
    """Node 3: Bounded hybrid RAG retrieval (dense + BM25 + RRF)."""
    t0 = _ts()
    query = state.get("decision_query")
    fund_ids = state.get("candidate_fund_ids", [])

    if query is None or not fund_ids:
        state["retrieval_results"] = []
        _trace(state, "evidence_retrieval", "Skipped: no query or no candidates.")
        return state

    try:
        results, diagnostics = retriever.retrieve(
            query=query.rewritten_query,
            fund_ids=fund_ids,
            include_global=True,
        )
        state["retrieval_results"] = results
        state["retrieval_diagnostics"] = diagnostics

        elapsed = (_ts() - t0) * 1000
        timing = state.get("timing", {})
        timing["evidence_retrieval_ms"] = round(elapsed, 2)
        state["timing"] = timing  # type: ignore[assignment]

        _trace(state, "evidence_retrieval", f"Retrieved {len(results)} chunks via hybrid RAG.", {
            "dense_candidates_count": diagnostics.dense_candidates_count if diagnostics else None,
            "sparse_candidates_count": diagnostics.sparse_candidates_count if diagnostics else None,
            "fused_candidates_count": diagnostics.fused_candidates_count if diagnostics else None,
        })
    except Exception as exc:  # noqa: BLE001
        logger.warning("Evidence retrieval failed: %s", exc)
        state["retrieval_results"] = []
        _trace(state, "evidence_retrieval", f"Retrieval error: {exc}", {"error": str(exc)})
    return state


def rerank_evidence(
    state: DecisionState,
    reranker: CrossEncoderReranker,
    settings: Settings,
) -> DecisionState:
    """Node 4: Cross-encoder reranking of retrieved chunks."""
    t0 = _ts()
    query = state.get("decision_query")
    results = state.get("retrieval_results", [])

    if not results or query is None:
        state["reranked_results"] = []
        _trace(state, "reranking", "Skipped: no retrieval results.")
        return state

    try:
        reranked, diagnostics = reranker.rerank(
            query=query.rewritten_query,
            retrieval_results=results,
            top_k=settings.rerank_top_k,
        )
        state["reranked_results"] = reranked
        state["reranking_diagnostics"] = diagnostics

        elapsed = (_ts() - t0) * 1000
        timing = state.get("timing", {})
        timing["reranking_ms"] = round(elapsed, 2)
        state["timing"] = timing  # type: ignore[assignment]

        _trace(state, "reranking", f"Reranked to {len(reranked)} top chunks.", {
            "top_score": diagnostics.top_score,
            "model": diagnostics.model_name,
            "reranking_time_ms": diagnostics.reranking_time_ms,
        })
    except Exception as exc:  # noqa: BLE001
        logger.warning("Reranking failed: %s — using raw retrieval order.", exc)
        # graceful degradation: convert retrieval results to reranked format with dummy scores
        from app.reranking.models import RerankedResult

        state["reranked_results"] = [
            RerankedResult(
                retrieval_result=r,
                reranker_score=0.0,
                rerank_position=i + 1,
                relative_relevance=1.0 - i / max(len(results), 1),
            )
            for i, r in enumerate(results[: settings.rerank_top_k])
        ]
        _trace(state, "reranking", f"Reranking failed ({exc}); using retrieval order.", {"error": str(exc)})
    return state


def validate_evidence(
    state: DecisionState,
    validator: EvidenceValidator,
    repository: FundRepository,
) -> DecisionState:
    """Node 5: Evidence validation — bind chunks to candidate funds, detect conflicts."""
    t0 = _ts()
    fund_ids = state.get("candidate_fund_ids", [])
    reranked = state.get("reranked_results", [])

    if not fund_ids:
        state["evidence_bundles"] = {}
        state["global_context_refs"] = []
        _trace(state, "evidence_validation", "Skipped: no candidate funds.")
        return state

    rows = repository.get_funds(fund_ids)
    fund_records: List[FundRecord] = [repository.row_to_fund_record(r) for r in rows]

    bundles, global_refs, report = validator.validate(reranked, fund_records)
    state["evidence_bundles"] = bundles
    state["global_context_refs"] = global_refs
    state["validation_report"] = report

    elapsed = (_ts() - t0) * 1000
    timing = state.get("timing", {})
    timing["evidence_validation_ms"] = round(elapsed, 2)
    state["timing"] = timing  # type: ignore[assignment]

    _trace(state, "evidence_validation", (
        f"Validated {report.chunks_in} chunks: "
        f"{report.verified} verified, {report.conflicting} conflicting, "
        f"{report.unbound_discarded} discarded."
    ), {
        "verified": report.verified,
        "conflicting": report.conflicting,
        "unbound": report.unbound_discarded,
        "funds_with_evidence": report.funds_with_evidence,
    })
    return state


def run_decision_engine(
    state: DecisionState,
    engine: DecisionEngine,
    repository: FundRepository,
    today: Optional[date] = None,
) -> DecisionState:
    """Node 6: Build CandidateInput objects from SQL data + verified evidence → DecisionResult."""
    t0 = _ts()
    query = state.get("decision_query")
    fund_ids = state.get("candidate_fund_ids", [])

    if query is None or not fund_ids:
        _trace(state, "decision_engine", "Skipped: no query or candidates.")
        return state

    rows = repository.get_funds(fund_ids)
    metrics_map = repository.get_metrics(fund_ids)
    bundles = state.get("evidence_bundles", {})

    candidates: List[CandidateInput] = []
    for row in rows:
        fund = repository.row_to_fund_record(row)
        metrics_row = metrics_map.get(fund.fund_id)
        metrics = repository.row_to_metrics_record(metrics_row) if metrics_row else None
        evidence = bundles.get(fund.fund_id)
        candidates.append(CandidateInput(
            fund=fund,
            metrics=metrics,
            evidence=evidence.evidence if evidence else [],
        ))

    if not candidates:
        _trace(state, "decision_engine", "No candidates to score (DB lookup returned empty).")
        return state

    data_as_of = repository.latest_data_as_of()
    state["data_as_of"] = data_as_of  # type: ignore[assignment]

    result = engine.decide(query=query, candidates=candidates, data_as_of=data_as_of, today=today)
    state["decision_result"] = result

    elapsed = (_ts() - t0) * 1000
    timing = state.get("timing", {})
    timing["decision_engine_ms"] = round(elapsed, 2)
    state["timing"] = timing  # type: ignore[assignment]

    _trace(state, "decision_engine", (
        f"Scored {len(candidates)} candidates; "
        f"winner={result.winner.fund_name if result.winner else 'None'}; "
        f"abstained={result.abstained}; confidence={result.confidence.level}."
    ), {
        "candidates_scored": len(candidates),
        "eligible": len(result.ranked),
        "excluded": len(result.excluded),
        "winner": result.winner.fund_id if result.winner else None,
        "confidence_level": result.confidence.level,
        "confidence_composite": result.confidence.composite,
        "abstained": result.abstained,
    })
    return state


def generate_response(
    state: DecisionState,
    generation_service: GenerationService,
) -> DecisionState:
    """Node 7/8: Grounded generation OR deterministic abstention response."""
    query = state.get("decision_query")
    result = state.get("decision_result")

    if query is None or result is None:
        _trace(state, "generation", "Skipped: missing query or decision result.")
        return state

    t0 = _ts()
    trace_so_far = list(state.get("decision_trace", []))
    response = generation_service.generate(query=query, decision=result, extra_trace=trace_so_far)

    elapsed = (_ts() - t0) * 1000
    timing = state.get("timing", {})
    timing["generation_ms"] = round(elapsed, 2)
    state["timing"] = timing  # type: ignore[assignment]

    # Merge timing into debug diagnostics if debug mode
    if state.get("debug"):
        response.debug_diagnostics = {"timing": timing}

    state["decision_response"] = response
    state["decision_trace"] = response.decision_trace  # type: ignore[assignment]
    return state
