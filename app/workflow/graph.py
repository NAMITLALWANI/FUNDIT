"""
DecisionWorkflow: orchestrates the full mutual fund decision pipeline.

Architecture note: we use a simple sequential runner rather than the full LangGraph
StateGraph to avoid requiring langgraph as a hard dependency (it can be added in Step 5
if explicit graph visualization is needed). The workflow design is fully compatible with
LangGraph — each step is a pure (state, dependencies) → state function and the runner
below matches the LangGraph execution model exactly.

Steps:
  1. analyze_query         → DecisionQuery or blocking error
  2. retrieve_candidates   → SQL-bounded fund_id list
  3. retrieve_evidence     → Hybrid RAG results
  4. rerank_evidence       → Cross-encoder reranked results
  5. validate_evidence     → EvidenceBundle per fund
  6. run_decision_engine   → DecisionResult (scored, ranked, confidence, abstention)
  7. generate_response     → DecisionResponse (LLM-grounded or deterministic fallback)

The workflow returns a DecisionResponse regardless of outcome (abstention, error, success).
"""

from datetime import date
from typing import Optional

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.repository import FundRepository
from app.decision.engine import DecisionEngine
from app.evidence.validator import EvidenceValidator
from app.generation.generator import GenerationService
from app.generation.models import DecisionResponse, DecisionTraceStep
from app.query.analyzer import QueryAnalyzer
from app.reranking.reranker import CrossEncoderReranker
from app.retrieval.retriever import EvidenceRetriever
from app.workflow.nodes import (
    analyze_query,
    generate_response,
    rerank_evidence,
    retrieve_candidates,
    retrieve_evidence,
    run_decision_engine,
    validate_evidence,
)
from app.workflow.state import DecisionState

logger = get_logger(__name__)


def _make_error_response(state: DecisionState, error_msg: str) -> DecisionResponse:
    """Build a minimal error-state DecisionResponse when the pipeline cannot continue."""
    from app.decision.models import ConfidenceAssessment, ConfidenceSignals

    signals = ConfidenceSignals(
        constraint_completeness=0.0, evidence_coverage=0.0, evidence_quality=0.0,
        score_margin=0.0, data_completeness=0.0, data_freshness=0.0, ranking_stability=0.0,
        raw_margin=0.0,
    )
    assessment = ConfidenceAssessment(level="LOW", composite=0.0, signals=signals, reasons=[error_msg])

    query = state.get("decision_query")
    trace = state.get("decision_trace", [])
    return DecisionResponse(
        query=state.get("raw_query", ""),
        parsed_query=query,  # type: ignore[arg-type]
        abstained=True,
        abstention_reason=error_msg,
        summary=f"Could not process query: {error_msg}",
        reasons=[error_msg],
        confidence=assessment,
        decision_trace=trace,
    )


class DecisionWorkflow:
    """Executes the full mutual fund decision pipeline sequentially."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        query_analyzer: Optional[QueryAnalyzer] = None,
        repository: Optional[FundRepository] = None,
        retriever: Optional[EvidenceRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        evidence_validator: Optional[EvidenceValidator] = None,
        decision_engine: Optional[DecisionEngine] = None,
        generation_service: Optional[GenerationService] = None,
        today: Optional[date] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.query_analyzer = query_analyzer or QueryAnalyzer()
        self.repository = repository  # required; injected by the API lifespan
        self.retriever = retriever  # required; injected by the API lifespan
        self.reranker = reranker or CrossEncoderReranker(settings=self.settings)
        self.evidence_validator = evidence_validator or EvidenceValidator(settings=self.settings)
        self.decision_engine = decision_engine or DecisionEngine(settings=self.settings)
        self.generation_service = generation_service or GenerationService(settings=self.settings)
        self.today = today

    def run(self, raw_query: str, debug: bool = False) -> DecisionResponse:
        """Execute the full pipeline and always return a DecisionResponse."""
        state: DecisionState = {
            "raw_query": raw_query,
            "debug": debug,
            "decision_trace": [],
            "timing": {},
        }

        # 1. Query analysis
        state = analyze_query(state, self.query_analyzer)
        if state.get("query_error") and not state.get("decision_query"):
            # Fatal: could not parse query at all
            return _make_error_response(state, state["query_error"])

        # 2. SQL candidate retrieval
        if self.repository is None:
            return _make_error_response(state, "Database repository is not initialized.")
        state = retrieve_candidates(state, self.repository, self.settings)

        candidate_count = state.get("sql_candidates_count", 0)
        if candidate_count == 0:
            # No candidates from SQL → build abstention directly
            from app.decision.models import ConfidenceAssessment, ConfidenceSignals

            signals = ConfidenceSignals(
                constraint_completeness=0.0, evidence_coverage=0.0, evidence_quality=0.0,
                score_margin=0.0, data_completeness=0.0,
                data_freshness=0.0, ranking_stability=0.0, raw_margin=0.0,
            )
            assessment = ConfidenceAssessment(
                level="LOW", composite=0.0, signals=signals,
                reasons=["No fund in the database satisfies all stated hard constraints."]
            )
            query = state.get("decision_query")
            trace = state.get("decision_trace", [])
            trace.append(DecisionTraceStep(
                stage="candidate_retrieval",
                summary="Zero candidates from SQL; abstaining without further processing.",
            ))
            return DecisionResponse(
                query=raw_query,
                parsed_query=query,  # type: ignore[arg-type]
                abstained=True,
                abstention_reason="No fund in the database satisfies all stated hard constraints.",
                summary="No eligible funds found.",
                reasons=["No fund satisfies the stated hard constraints based on the indexed fund catalog."],
                confidence=assessment,
                decision_trace=trace,
                debug_diagnostics={"timing": state.get("timing", {})} if debug else None,
            )

        # 3. Evidence retrieval
        if self.retriever is not None:
            state = retrieve_evidence(state, self.retriever)

            # 4. Reranking
            state = rerank_evidence(state, self.reranker, self.settings)

            # 5. Evidence validation
            state = validate_evidence(state, self.evidence_validator, self.repository)
        else:
            logger.warning("EvidenceRetriever not initialized; skipping RAG pipeline.")
            state["retrieval_results"] = []
            state["reranked_results"] = []
            state["evidence_bundles"] = {}
            state["global_context_refs"] = []

        # 6. Decision engine
        state = run_decision_engine(state, self.decision_engine, self.repository, today=self.today)
        if state.get("decision_result") is None:
            return _make_error_response(state, "Decision engine produced no result.")

        # 7. Generation
        state = generate_response(state, self.generation_service)

        response = state.get("decision_response")
        if response is None:
            return _make_error_response(state, "Generation service produced no response.")

        logger.info(
            "Workflow complete: abstained=%s confidence=%s timing=%s",
            response.abstained,
            response.confidence.level,
            {k: f"{v:.0f}ms" for k, v in state.get("timing", {}).items()},
        )
        return response
