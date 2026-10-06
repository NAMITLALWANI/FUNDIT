"""
LangGraph typed state for the mutual fund decision workflow.

DecisionState is a TypedDict that flows through every workflow node.
Each node reads what it needs and writes its outputs — nodes never mutate
each other's fields. The state is the single source of truth at each step.
"""

from datetime import date
from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict

from app.decision.models import DecisionResult
from app.evidence.validator import EvidenceBundle, ValidationReport
from app.generation.models import DecisionResponse, DecisionTraceStep
from app.query.models import DecisionQuery
from app.reranking.models import RerankedResult, RerankingDiagnostics
from app.retrieval.models import RetrievalDiagnostics, RetrievalResult


class DecisionState(TypedDict, total=False):
    """Typed state dictionary flowing through the LangGraph decision workflow."""

    # ---------- Input ----------
    raw_query: str
    debug: bool
    top_k: int

    # ---------- Query analysis ----------
    decision_query: Optional[DecisionQuery]
    query_error: Optional[str]

    # ---------- SQL candidate retrieval ----------
    candidate_fund_ids: List[str]
    sql_candidates_count: int

    # ---------- Evidence retrieval ----------
    retrieval_results: List[RetrievalResult]
    retrieval_diagnostics: Optional[RetrievalDiagnostics]

    # ---------- Reranking ----------
    reranked_results: List[RerankedResult]
    reranking_diagnostics: Optional[RerankingDiagnostics]

    # ---------- Evidence validation ----------
    evidence_bundles: Dict[str, EvidenceBundle]
    global_context_refs: List[Any]
    validation_report: Optional[ValidationReport]

    # ---------- Decision engine ----------
    decision_result: Optional[DecisionResult]

    # ---------- Generation ----------
    decision_response: Optional[DecisionResponse]

    # ---------- Trace ----------
    decision_trace: List[DecisionTraceStep]
    timing: Dict[str, float]
    data_as_of: Optional[date]
    error: Optional[str]
