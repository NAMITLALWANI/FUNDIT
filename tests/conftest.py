"""
Shared test fixtures.

All fund data here is SYNTHETIC (fund IDs prefixed ``SYN-``) and exists only to exercise
deterministic logic. It is never loaded into the demo database and does not describe any real
scheme.
"""

from datetime import date
from typing import List, Optional

import pytest

from app.core.config import Settings
from app.decision.models import CandidateInput, EvidenceRef
from app.ingestion.models import FundMetricsRecord, FundRecord
from app.query.models import DecisionQuery, QueryConstraints, UserPreferences

AS_OF = date(2026, 9, 30)


@pytest.fixture
def settings() -> Settings:
    """Settings isolated from any local .env file."""
    return Settings(_env_file=None, llm_provider="mock", database_url="sqlite:///:memory:")


def make_fund(
    fund_id: str,
    name: str,
    *,
    amc: str = "Synthetic AMC",
    category: str = "Equity",
    sub_category: Optional[str] = "Flexi Cap Fund",
    risk_level: Optional[str] = "Very High",
    expense_ratio: Optional[float] = 0.8,
    aum_crores: Optional[float] = 10_000.0,
    min_sip: Optional[float] = 500.0,
    min_lump: Optional[float] = 5_000.0,
    lock_in: Optional[float] = 0.0,
    exit_load: Optional[str] = None,
    plan_type: str = "Direct",
) -> FundRecord:
    return FundRecord(
        fund_id=fund_id,
        scheme_code=abs(hash(fund_id)) % 100_000,
        fund_name=name,
        amc=amc,
        category=category,
        sub_category=sub_category,
        risk_level=risk_level,  # type: ignore[arg-type]
        expense_ratio=expense_ratio,
        expense_ratio_date=AS_OF,
        aum_crores=aum_crores,
        min_sip_amount=min_sip,
        min_lump_sum=min_lump,
        lock_in_years=lock_in,
        exit_load=exit_load,
        plan_type=plan_type,  # type: ignore[arg-type]
        data_as_of=AS_OF,
    )


def make_metrics(
    fund_id: str,
    *,
    cagr_3y: Optional[float] = 0.15,
    cagr_5y: Optional[float] = 0.14,
    volatility_3y: Optional[float] = 0.12,
    sharpe_3y: Optional[float] = 0.7,
    max_drawdown_3y: Optional[float] = -0.15,
    rolling: Optional[float] = 0.9,
) -> FundMetricsRecord:
    return FundMetricsRecord(
        fund_id=fund_id,
        as_of_date=AS_OF,
        history_start=date(2015, 1, 1),
        history_years=11.7,
        cagr_1y=0.1,
        cagr_3y=cagr_3y,
        cagr_5y=cagr_5y,
        volatility_3y=volatility_3y,
        sharpe_3y=sharpe_3y,
        max_drawdown_3y=max_drawdown_3y,
        rolling_3y_positive_pct=rolling,
        risk_free_rate_used=0.065,
    )


def make_evidence(fund_id: str, n: int = 2, relevance: float = 0.8) -> List[EvidenceRef]:
    return [
        EvidenceRef(
            chunk_id=f"{fund_id}-summary-c{i}",
            document_id=f"{fund_id}-scheme-summary",
            document_type="SchemeSummary",
            title=f"{fund_id} summary",
            source_url="https://example.invalid/synthetic",
            publication_date=AS_OF,
            relative_relevance=relevance,
            reranker_score=1.0,
            snippet="synthetic evidence snippet",
        )
        for i in range(n)
    ]


def make_query(
    raw: str = "synthetic query",
    constraints: Optional[QueryConstraints] = None,
    preferences: Optional[UserPreferences] = None,
    horizon: Optional[float] = None,
    ambiguities=None,
) -> DecisionQuery:
    return DecisionQuery(
        raw_query=raw,
        rewritten_query=raw,
        constraints=constraints or QueryConstraints(),
        preferences=preferences or UserPreferences(),
        horizon_years=horizon,
        ambiguities=ambiguities or [],
    )


@pytest.fixture
def three_candidates() -> List[CandidateInput]:
    """A: best Sharpe/cheapest, B: middle, C: expensive & volatile. All with evidence."""
    a = make_fund("SYN-A", "Synthetic Alpha Fund", expense_ratio=0.5, aum_crores=20_000)
    b = make_fund("SYN-B", "Synthetic Beta Fund", expense_ratio=0.9, aum_crores=8_000)
    c = make_fund("SYN-C", "Synthetic Gamma Fund", expense_ratio=1.4, aum_crores=2_000)
    return [
        CandidateInput(fund=a, metrics=make_metrics("SYN-A", sharpe_3y=1.0, volatility_3y=0.10, max_drawdown_3y=-0.10), evidence=make_evidence("SYN-A", 3)),
        CandidateInput(fund=b, metrics=make_metrics("SYN-B", sharpe_3y=0.6, volatility_3y=0.13, max_drawdown_3y=-0.16), evidence=make_evidence("SYN-B", 2)),
        CandidateInput(fund=c, metrics=make_metrics("SYN-C", sharpe_3y=0.2, volatility_3y=0.20, max_drawdown_3y=-0.30), evidence=make_evidence("SYN-C", 1)),
    ]
