"""
Generation domain models for structured LLM outputs, citations, and the final API response.

The ``DecisionResponse`` is the complete external API payload; it is assembled by the
generation layer after the deterministic engine has produced ``DecisionResult``.
"""

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.decision.models import (
    CandidateScore,
    ConfidenceAssessment,
    SensitivityReport,
    Tradeoff,
)
from app.query.models import DecisionQuery

DISCLAIMER = (
    "Disclaimer: This analysis is provided for educational and decision-support purposes "
    "based on historical data and publicly available scheme documents. Past performance does "
    "not guarantee future results. Mutual fund investments are subject to market risks. "
    "Please read all scheme-related documents carefully and consult a SEBI-registered "
    "financial advisor before making investment decisions."
)


class Citation(BaseModel):
    """Traceable reference to a retrieved document chunk."""

    citation_id: str = Field(description="Short reference tag, e.g., '[1]'")
    document_id: str = Field(description="Origin document ID")
    chunk_id: str = Field(description="Origin chunk ID")
    fund_id: Optional[str] = Field(default=None, description="Associated fund (None for regulatory docs)")
    source: str = Field(description="Source name / publisher")
    source_url: Optional[str] = Field(default=None, description="Origin URL or file path")
    title: str = Field(description="Source title")
    document_type: str = Field(description="Document type, e.g., SID, Factsheet")
    publication_date: Optional[date] = None
    snippet: str = Field(description="Relevant text excerpt from chunk (<=400 chars)")


class GeneratedDecision(BaseModel):
    """Raw structured output schema enforced on LLM generation.

    The LLM receives verified evidence chunks + the deterministic decision and must output
    ONLY this schema as JSON. cited_chunk_ids is validated post-generation against the
    actual retrieved chunk IDs to detect any ungrounded references.
    """

    recommendation_text: str = Field(
        description="Clear, grounded recommendation explaining why the winner was selected; must cite evidence markers"
    )
    summary: str = Field(description="High-level 1-2 sentence summary of the decision")
    reasons: List[str] = Field(default_factory=list, description="Bulleted rationale supporting the recommendation")
    tradeoff_notes: List[str] = Field(default_factory=list, description="Qualitative notes on trade-offs vs alternatives")
    cited_chunk_ids: List[str] = Field(
        default_factory=list, description="Chunk IDs referenced in the output (validated against retrieved set)"
    )


class DecisionTraceStep(BaseModel):
    """One auditable step in the decision pipeline."""

    stage: str
    summary: str
    details: Dict[str, Any] = Field(default_factory=dict)


class DecisionResponse(BaseModel):
    """Complete API response payload for a mutual fund decision query."""

    query: str = Field(description="Original user query")
    parsed_query: DecisionQuery = Field(description="Structured representation of the query")
    abstained: bool = Field(default=False, description="True when the system could not produce a recommendation")
    abstention_reason: Optional[str] = Field(default=None, description="Deterministic reason for abstention")
    winner: Optional[CandidateScore] = Field(default=None, description="Recommended winning fund")
    top_candidates: List[CandidateScore] = Field(default_factory=list, description="Ranked eligible candidates")
    excluded_candidates: List[CandidateScore] = Field(
        default_factory=list, description="Candidates excluded by hard constraints"
    )
    recommendation_text: str = Field(default="", description="Grounded LLM-produced recommendation text")
    summary: str = Field(default="", description="High-level summary of the decision")
    reasons: List[str] = Field(default_factory=list, description="Key reasons for the recommendation")
    tradeoffs: List[Tradeoff] = Field(default_factory=list, description="Structured pairwise trade-offs")
    citations: List[Citation] = Field(default_factory=list, description="Traceable chunk citations")
    confidence: ConfidenceAssessment
    sensitivity: Optional[SensitivityReport] = None
    decision_trace: List[DecisionTraceStep] = Field(default_factory=list)
    weights_used: Dict[str, float] = Field(default_factory=dict)
    disclaimer: str = DISCLAIMER
    debug_diagnostics: Optional[Dict[str, Any]] = Field(
        default=None, description="Diagnostic telemetry (debug mode only)"
    )
