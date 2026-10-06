"""
Decision engine data models.

Two kinds of facts are kept strictly apart:
* ``StructuredFact`` – a value read from the relational tables (funds / fund_metrics) with its
  as-of date. These are the only facts the engine computes with.
* ``EvidenceRef`` – a reference to a verified document chunk retrieved for the fund. Evidence
  influences the evidence-quality component and provides citations; it never changes a
  structured number.
"""

from datetime import date
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.ingestion.models import FundMetricsRecord, FundRecord

ConstraintStatus = Literal["PASS", "FAIL", "UNKNOWN"]
ConfidenceLevel = Literal["LOW", "MEDIUM", "HIGH"]

STATUS_RANK: Dict[str, int] = {"PASS": 0, "UNKNOWN": 1, "FAIL": 2}


class StructuredFact(BaseModel):
    """A single value sourced from the structured database."""

    field: str
    value: Any
    unit: Optional[str] = None
    as_of: Optional[date] = None
    source: Literal["funds", "fund_metrics"] = "funds"


class EvidenceRef(BaseModel):
    """Verified document chunk attached to a candidate."""

    chunk_id: str
    document_id: str
    document_type: str
    title: str
    source_url: Optional[str] = None
    publication_date: Optional[date] = None
    relative_relevance: float = Field(description="Within-query [0,1] reranker position; not a probability")
    reranker_score: float
    snippet: str


class ConstraintResult(BaseModel):
    """Outcome of one hard-constraint check for one candidate."""

    field: str
    status: ConstraintStatus
    expected: Any = None
    actual: Any = None
    reason: str


class ConstraintEvaluation(BaseModel):
    """Aggregate tri-state outcome for a candidate."""

    status: ConstraintStatus
    results: List[ConstraintResult] = Field(default_factory=list)

    @property
    def failures(self) -> List[ConstraintResult]:
        return [r for r in self.results if r.status == "FAIL"]

    @property
    def unknowns(self) -> List[ConstraintResult]:
        return [r for r in self.results if r.status == "UNKNOWN"]

    @property
    def passes(self) -> List[ConstraintResult]:
        return [r for r in self.results if r.status == "PASS"]


class ComponentScore(BaseModel):
    """One weighted component of the utility score."""

    name: str
    weight: float
    score: Optional[float] = Field(default=None, description="[0,1] within the candidate set; None when data is missing")
    contribution: float = 0.0
    basis: str = Field(default="", description="Which structured fields / evidence produced the score")


class CandidateInput(BaseModel):
    """What the engine receives per candidate: structured rows plus verified evidence only."""

    fund: FundRecord
    metrics: Optional[FundMetricsRecord] = None
    evidence: List[EvidenceRef] = Field(default_factory=list)


class CandidateScore(BaseModel):
    """Fully explained score for a candidate."""

    fund_id: str
    fund_name: str
    amc: str
    category: str
    sub_category: Optional[str] = None
    constraint_status: ConstraintStatus
    constraint_results: List[ConstraintResult] = Field(default_factory=list)
    violations: List[str] = Field(default_factory=list)
    unknowns: List[str] = Field(default_factory=list)
    components: List[ComponentScore] = Field(default_factory=list)
    unknown_penalty: float = 0.0
    final_score: float
    data_completeness: float = Field(description="Share of scoring components with available data")
    evidence_available: bool
    evidence: List[EvidenceRef] = Field(default_factory=list)
    structured_facts: List[StructuredFact] = Field(default_factory=list)
    rank: Optional[int] = None

    def component(self, name: str) -> Optional[ComponentScore]:
        return next((c for c in self.components if c.name == name), None)


class Tradeoff(BaseModel):
    """Structured pairwise comparison between the winner and an alternative."""

    fund_a: str
    fund_b: str
    advantages_a: List[str] = Field(default_factory=list)
    advantages_b: List[str] = Field(default_factory=list)
    summary: str
    fact_basis: List[str] = Field(default_factory=list, description="Structured fields used for each statement")


class ConfidenceSignals(BaseModel):
    """Diagnostic inputs to the confidence level. Each signal is in [0,1]."""

    constraint_completeness: float
    evidence_coverage: float
    evidence_quality: float
    score_margin: float
    data_completeness: float
    data_freshness: float
    ranking_stability: float
    raw_margin: float = Field(description="Winner final_score minus runner-up final_score")
    unknown_constraint_count: int = 0
    ambiguity_count: int = 0


class ConfidenceAssessment(BaseModel):
    """System confidence in the decision (evidence/decision confidence, NOT a return forecast)."""

    level: ConfidenceLevel
    composite: float
    signals: ConfidenceSignals
    reasons: List[str] = Field(default_factory=list)
    description: str = (
        "System confidence reflects constraint verification, evidence coverage, score separation and data "
        "completeness. It is not a probability of future investment returns."
    )


class SensitivityReport(BaseModel):
    """Ranking stability under small perturbations of the decision weights."""

    perturbation: float
    trials: int
    winner_flip_rate: float
    top3_jaccard_mean: float
    alternative_winners: Dict[str, int] = Field(default_factory=dict)
    baseline_winner: Optional[str] = None


class DecisionResult(BaseModel):
    """Complete deterministic decision output handed to the confidence gate / generator."""

    candidates_considered: int
    winner: Optional[CandidateScore] = None
    ranked: List[CandidateScore] = Field(default_factory=list, description="PASS then UNKNOWN candidates, best first")
    excluded: List[CandidateScore] = Field(default_factory=list, description="FAIL candidates with violations")
    tradeoffs: List[Tradeoff] = Field(default_factory=list)
    confidence: ConfidenceAssessment
    sensitivity: Optional[SensitivityReport] = None
    abstained: bool = False
    abstention_reason: Optional[str] = None
    selection_reasons: List[str] = Field(default_factory=list)
    rejection_reasons: Dict[str, List[str]] = Field(default_factory=dict)
    weights_used: Dict[str, float] = Field(default_factory=dict)
