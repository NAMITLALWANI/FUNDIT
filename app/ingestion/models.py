"""
Mutual fund domain models shared by ingestion, retrieval, and decision components.

`FundRecord` / `FundMetricsRecord` mirror the relational tables; `DocumentChunk` is the
provenance-preserving unit indexed by the hybrid retriever.
"""

from datetime import date
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

RiskLevel = Literal["Low", "Low to Moderate", "Moderate", "Moderately High", "High", "Very High"]
DocumentType = Literal["SID", "Factsheet", "KIM", "SchemeSummary", "Regulatory", "Commentary"]

RISK_LEVEL_ORDER: Dict[str, int] = {
    "Low": 1,
    "Low to Moderate": 2,
    "Moderate": 3,
    "Moderately High": 4,
    "High": 5,
    "Very High": 6,
}


class FundRecord(BaseModel):
    """Structured mutual fund scheme attributes (one row in the `funds` table)."""

    fund_id: str
    scheme_code: int = Field(description="AMFI scheme code")
    isin: Optional[str] = None
    fund_name: str
    amc: str
    category: str = Field(description="SEBI broad category, e.g. Equity, Debt, Hybrid, Other")
    sub_category: Optional[str] = Field(default=None, description="SEBI sub-category, e.g. Flexi Cap Fund")
    risk_level: Optional[RiskLevel] = Field(default=None, description="SEBI riskometer label")
    expense_ratio: Optional[float] = Field(default=None, description="Total expense ratio in percent")
    expense_ratio_date: Optional[date] = None
    aum_crores: Optional[float] = Field(default=None, description="Assets under management in INR crore")
    min_sip_amount: Optional[float] = None
    min_lump_sum: Optional[float] = None
    lock_in_years: Optional[float] = None
    exit_load: Optional[str] = None
    benchmark: Optional[str] = None
    investment_objective: Optional[str] = None
    fund_manager: Optional[str] = None
    scheme_type: str = "Open Ended"
    plan_type: Literal["Direct", "Regular"] = "Direct"
    option_type: str = "Growth"
    inception_date: Optional[date] = None
    data_as_of: date
    source_id: Optional[str] = None
    source_url: Optional[str] = None


class NavPoint(BaseModel):
    """Single NAV observation."""

    date: date
    nav: float


class FundMetricsRecord(BaseModel):
    """Derived risk/return metrics computed from NAV history (one row in `fund_metrics`)."""

    fund_id: str
    as_of_date: date
    history_start: Optional[date] = None
    history_years: Optional[float] = None
    cagr_1y: Optional[float] = Field(default=None, description="1-year point-to-point return (decimal)")
    cagr_3y: Optional[float] = Field(default=None, description="3-year annualised return (decimal)")
    cagr_5y: Optional[float] = Field(default=None, description="5-year annualised return (decimal)")
    volatility_3y: Optional[float] = Field(
        default=None, description="Annualised std-dev of daily log returns over 3 years (decimal)"
    )
    sharpe_3y: Optional[float] = Field(default=None, description="(cagr_3y - risk_free) / volatility_3y")
    max_drawdown_3y: Optional[float] = Field(default=None, description="Max peak-to-trough decline over 3 years (decimal, negative)")
    rolling_3y_positive_pct: Optional[float] = Field(
        default=None, description="Share of monthly-stepped 3Y rolling windows with positive return"
    )
    risk_free_rate_used: float
    method_version: str = "v1"


class DataSourceRecord(BaseModel):
    """Provenance record for an external dataset."""

    source_id: str
    name: str
    url: str
    collected_at: date
    description: str
    license_notes: str


class SourceMetadata(BaseModel):
    """Provenance metadata attached to every document chunk."""

    document_id: str
    document_type: DocumentType
    source: str = Field(description="Publishing organisation or dataset name")
    source_url: Optional[str] = None
    title: str
    fund_id: Optional[str] = Field(default=None, description="Associated fund, None for regulatory docs")
    publication_date: Optional[date] = None
    section: Optional[str] = None
    chunk_index: int = 0
    extra: Dict[str, Any] = Field(default_factory=dict)


class RawDocument(BaseModel):
    """Unprocessed document payload prior to chunking."""

    document_id: str
    document_type: DocumentType
    title: str
    source: str
    source_url: Optional[str] = None
    fund_id: Optional[str] = None
    publication_date: Optional[date] = None
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """Normalized chunk ready for embedding and indexing. Provenance is never dropped."""

    chunk_id: str
    document_id: str
    text: str
    metadata: SourceMetadata

    @property
    def fund_id(self) -> Optional[str]:
        return self.metadata.fund_id

    @property
    def title(self) -> str:
        return self.metadata.title

    @property
    def source(self) -> str:
        return self.metadata.source

    @property
    def source_url(self) -> Optional[str]:
        return self.metadata.source_url


class DataQualityIssue(BaseModel):
    """A single data-quality problem detected during ingestion."""

    fund_id: Optional[str]
    severity: Literal["info", "warning", "error"]
    field: str
    message: str


class DataQualityReport(BaseModel):
    """Summary emitted by the ingestion pipeline."""

    generated_at: date
    funds_total: int
    funds_loaded: int
    funds_with_attribute_gaps: int
    nav_points_loaded: int
    documents_loaded: int
    chunks_created: int
    issues: List[DataQualityIssue] = Field(default_factory=list)
