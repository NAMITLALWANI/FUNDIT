"""
Query domain models: structured constraints, soft preferences, ambiguities and the DecisionQuery.

Hard constraints eliminate candidates (or mark them UNKNOWN when data is missing); preferences
only influence ranking. Anything the extractor could not resolve is recorded as an explicit
``Ambiguity`` instead of being silently defaulted.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.ingestion.models import RISK_LEVEL_ORDER

InvestmentFrequency = Literal["monthly_sip", "lump_sum"]
QueryIntent = Literal["recommendation", "comparison", "explanation", "filter"]
AmbiguitySeverity = Literal["warning", "blocking"]

PREFERENCE_KEYS = (
    "low_cost",
    "low_volatility",
    "high_return",
    "consistency",
    "fund_size",
    "preferred_amc",
)


class QueryConstraints(BaseModel):
    """Hard constraints. Every field is optional; ``None`` means 'not requested by the user'."""

    sip_amount: Optional[float] = Field(default=None, description="Monthly SIP budget; fund min SIP must be <= this")
    lump_sum_amount: Optional[float] = Field(default=None, description="One-time amount; fund min lump sum must be <= this")
    max_expense_ratio: Optional[float] = Field(default=None, description="Maximum total expense ratio in percent")
    max_risk_level: Optional[str] = Field(default=None, description="Highest acceptable SEBI Risk-o-meter label")
    categories: Optional[List[str]] = Field(default=None, description="Allowed SEBI broad categories")
    sub_categories: Optional[List[str]] = Field(default=None, description="Allowed SEBI sub-categories")
    min_aum_crores: Optional[float] = None
    max_lock_in_years: Optional[float] = Field(default=None, description="Derived from the horizon: lock-in must not exceed it")
    exit_load_free: Optional[bool] = Field(default=None, description="User requires no exit load")
    min_return_3y: Optional[float] = Field(
        default=None, description="Minimum historical 3-year CAGR (decimal); evaluated on past data only"
    )
    required_amcs: Optional[List[str]] = Field(default=None, description="Only set when the user says 'only <AMC> funds'")
    plan_type: Optional[Literal["Direct", "Regular"]] = Field(
        default=None, description="Not assumed; None unless the user explicitly says Direct or Regular"
    )

    @field_validator("max_risk_level")
    @classmethod
    def _validate_risk(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in RISK_LEVEL_ORDER:
            raise ValueError(f"Unknown risk level '{value}'")
        return value

    @field_validator("max_expense_ratio")
    @classmethod
    def _validate_expense(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not (0.0 <= value <= 5.0):
            raise ValueError("max_expense_ratio must be between 0 and 5 percent")
        return value

    def active_fields(self) -> List[str]:
        """Names of constraints the user actually set."""
        return [k for k, v in self.model_dump().items() if v is not None]


class UserPreferences(BaseModel):
    """Soft preferences as 0..1 emphasis weights; absent keys mean 'not mentioned'."""

    weights: Dict[str, float] = Field(default_factory=dict)
    preferred_amcs: List[str] = Field(default_factory=list)

    @field_validator("weights")
    @classmethod
    def _validate_weights(cls, value: Dict[str, float]) -> Dict[str, float]:
        for key, weight in value.items():
            if key not in PREFERENCE_KEYS:
                raise ValueError(f"Unknown preference '{key}'")
            if not (0.0 <= weight <= 1.0):
                raise ValueError(f"Preference weight for '{key}' must be in [0, 1]")
        return value

    @property
    def is_empty(self) -> bool:
        return not self.weights and not self.preferred_amcs


class Ambiguity(BaseModel):
    """An unresolved or contradictory aspect of the query that the system refuses to guess."""

    field: str
    message: str
    severity: AmbiguitySeverity = "warning"
    candidates: List[str] = Field(default_factory=list)


class ComparisonTarget(BaseModel):
    """A fund the user named explicitly, resolved against the fund table."""

    mention: str
    fund_id: Optional[str] = None
    fund_name: Optional[str] = None
    match_score: float = 0.0


class DecisionQuery(BaseModel):
    """Validated structured representation of a mutual fund decision request."""

    raw_query: str
    rewritten_query: str
    intent: QueryIntent = "recommendation"
    domain: Literal["mutual_funds"] = "mutual_funds"
    investment_amount: Optional[float] = None
    investment_frequency: Optional[InvestmentFrequency] = None
    horizon_years: Optional[float] = None
    risk_tolerance: Optional[str] = Field(default=None, description="SEBI risk label the user is willing to accept")
    objective: Optional[str] = None
    constraints: QueryConstraints = Field(default_factory=QueryConstraints)
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    comparison_targets: List[ComparisonTarget] = Field(default_factory=list)
    ambiguities: List[Ambiguity] = Field(default_factory=list)
    extraction_method: str = "deterministic"

    @property
    def blocking_ambiguities(self) -> List[Ambiguity]:
        return [a for a in self.ambiguities if a.severity == "blocking"]

    @property
    def resolved_target_ids(self) -> List[str]:
        return [t.fund_id for t in self.comparison_targets if t.fund_id]
