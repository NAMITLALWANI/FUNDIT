"""
Parameterized data-access layer for funds, metrics, NAV history, and document chunks.

All queries are built with the SQLAlchemy expression language (bound parameters only). The
structured filter deliberately removes *definite failures* only: rows whose attribute is NULL
are retained so the decision engine can label them UNKNOWN instead of silently passing them.
"""

from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence

from pydantic import BaseModel, Field
from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session, selectinload

from app.db.models import (
    DataSourceRow,
    DocumentChunkRow,
    DocumentRow,
    FundMetricsRow,
    FundPerformanceRow,
    FundRow,
)
from app.ingestion.models import (
    RISK_LEVEL_ORDER,
    DataSourceRecord,
    DocumentChunk,
    FundMetricsRecord,
    FundRecord,
    NavPoint,
    RawDocument,
    SourceMetadata,
)


class FundFilter(BaseModel):
    """Structured candidate filter translated into parameterized SQL predicates."""

    fund_ids: Optional[List[str]] = None
    sip_amount: Optional[float] = Field(default=None, description="Exclude funds whose min SIP exceeds this")
    lump_sum_amount: Optional[float] = Field(default=None, description="Exclude funds whose min lump sum exceeds this")
    max_expense_ratio: Optional[float] = None
    max_risk_level: Optional[str] = Field(default=None, description="Highest acceptable SEBI riskometer label")
    categories: Optional[List[str]] = None
    sub_categories: Optional[List[str]] = None
    min_aum_crores: Optional[float] = None
    max_lock_in_years: Optional[float] = None
    plan_type: Optional[str] = None
    amcs: Optional[List[str]] = Field(default=None, description="Case-insensitive AMC name fragments (hard 'only X funds')")
    limit: int = 50


class FundRepository:
    """Repository exposing typed read/write operations over the fund schema."""

    def __init__(self, session: Session) -> None:
        self.session = session

    # ------------------------------------------------------------------ reads

    def count_funds(self) -> int:
        return int(self.session.scalar(select(func.count()).select_from(FundRow)) or 0)

    def list_funds(self) -> List[FundRow]:
        stmt = select(FundRow).options(selectinload(FundRow.metrics)).order_by(FundRow.fund_name)
        return list(self.session.scalars(stmt).all())

    def get_fund(self, fund_id: str) -> Optional[FundRow]:
        stmt = select(FundRow).options(selectinload(FundRow.metrics)).where(FundRow.fund_id == fund_id)
        return self.session.scalars(stmt).first()

    def get_funds(self, fund_ids: Sequence[str]) -> List[FundRow]:
        if not fund_ids:
            return []
        stmt = (
            select(FundRow)
            .options(selectinload(FundRow.metrics))
            .where(FundRow.fund_id.in_(list(fund_ids)))
        )
        return list(self.session.scalars(stmt).all())

    def search_funds(self, filters: FundFilter) -> List[FundRow]:
        """Return funds that do not definitively violate any structured filter."""
        stmt = select(FundRow).options(selectinload(FundRow.metrics))

        if filters.fund_ids:
            stmt = stmt.where(FundRow.fund_id.in_(filters.fund_ids))
        if filters.sip_amount is not None:
            stmt = stmt.where(
                or_(FundRow.min_sip_amount.is_(None), FundRow.min_sip_amount <= filters.sip_amount)
            )
        if filters.lump_sum_amount is not None:
            stmt = stmt.where(
                or_(FundRow.min_lump_sum.is_(None), FundRow.min_lump_sum <= filters.lump_sum_amount)
            )
        if filters.max_expense_ratio is not None:
            stmt = stmt.where(
                or_(FundRow.expense_ratio.is_(None), FundRow.expense_ratio <= filters.max_expense_ratio)
            )
        if filters.max_risk_level is not None and filters.max_risk_level in RISK_LEVEL_ORDER:
            allowed = [
                label
                for label, rank in RISK_LEVEL_ORDER.items()
                if rank <= RISK_LEVEL_ORDER[filters.max_risk_level]
            ]
            stmt = stmt.where(or_(FundRow.risk_level.is_(None), FundRow.risk_level.in_(allowed)))
        if filters.categories:
            stmt = stmt.where(FundRow.category.in_(filters.categories))
        if filters.sub_categories:
            stmt = stmt.where(
                or_(FundRow.sub_category.is_(None), FundRow.sub_category.in_(filters.sub_categories))
            )
        if filters.min_aum_crores is not None:
            stmt = stmt.where(
                or_(FundRow.aum_crores.is_(None), FundRow.aum_crores >= filters.min_aum_crores)
            )
        if filters.max_lock_in_years is not None:
            stmt = stmt.where(
                or_(FundRow.lock_in_years.is_(None), FundRow.lock_in_years <= filters.max_lock_in_years)
            )
        if filters.plan_type:
            stmt = stmt.where(FundRow.plan_type == filters.plan_type)
        if filters.amcs:
            stmt = stmt.where(or_(*[func.lower(FundRow.amc).like(f"%{a.lower()}%") for a in filters.amcs]))

        stmt = stmt.order_by(FundRow.fund_name).limit(filters.limit)
        return list(self.session.scalars(stmt).all())

    def get_metrics(self, fund_ids: Sequence[str]) -> Dict[str, FundMetricsRow]:
        if not fund_ids:
            return {}
        stmt = select(FundMetricsRow).where(FundMetricsRow.fund_id.in_(list(fund_ids)))
        return {row.fund_id: row for row in self.session.scalars(stmt).all()}

    def get_nav_history(self, fund_id: str, start: Optional[date] = None) -> List[NavPoint]:
        stmt = select(FundPerformanceRow).where(FundPerformanceRow.fund_id == fund_id)
        if start is not None:
            stmt = stmt.where(FundPerformanceRow.nav_date >= start)
        stmt = stmt.order_by(FundPerformanceRow.nav_date)
        return [NavPoint(date=row.nav_date, nav=row.nav) for row in self.session.scalars(stmt).all()]

    def count_nav_points(self) -> int:
        return int(self.session.scalar(select(func.count()).select_from(FundPerformanceRow)) or 0)

    def latest_data_as_of(self) -> Optional[date]:
        return self.session.scalar(select(func.max(FundRow.data_as_of)))

    def list_chunks(self, fund_ids: Optional[Sequence[str]] = None, include_global: bool = True) -> List[DocumentChunk]:
        """Load document chunks (optionally restricted to funds) as provenance-bearing models."""
        stmt = select(DocumentChunkRow).options(selectinload(DocumentChunkRow.document))
        if fund_ids is not None:
            if include_global:
                stmt = stmt.where(
                    or_(DocumentChunkRow.fund_id.in_(list(fund_ids)), DocumentChunkRow.fund_id.is_(None))
                )
            else:
                stmt = stmt.where(DocumentChunkRow.fund_id.in_(list(fund_ids)))
        stmt = stmt.order_by(DocumentChunkRow.document_id, DocumentChunkRow.chunk_index)
        return [self.row_to_chunk(row) for row in self.session.scalars(stmt).all()]

    def count_documents(self) -> int:
        return int(self.session.scalar(select(func.count()).select_from(DocumentRow)) or 0)

    def list_sources(self) -> List[DataSourceRow]:
        return list(self.session.scalars(select(DataSourceRow)).all())

    # ----------------------------------------------------------------- writes

    def upsert_source(self, source: DataSourceRecord) -> None:
        row = self.session.get(DataSourceRow, source.source_id) or DataSourceRow(source_id=source.source_id)
        row.name = source.name
        row.url = source.url
        row.collected_at = source.collected_at
        row.description = source.description
        row.license_notes = source.license_notes
        self.session.merge(row)

    def upsert_fund(self, fund: FundRecord) -> None:
        row = self.session.get(FundRow, fund.fund_id) or FundRow(fund_id=fund.fund_id)
        for field, value in fund.model_dump(exclude={"fund_id"}).items():
            setattr(row, field, value)
        self.session.merge(row)

    def replace_nav_history(self, fund_id: str, points: Iterable[NavPoint]) -> int:
        self.session.execute(delete(FundPerformanceRow).where(FundPerformanceRow.fund_id == fund_id))
        rows = [FundPerformanceRow(fund_id=fund_id, nav_date=p.date, nav=p.nav) for p in points]
        self.session.add_all(rows)
        return len(rows)

    def upsert_metrics(self, metrics: FundMetricsRecord) -> None:
        row = self.session.get(FundMetricsRow, metrics.fund_id) or FundMetricsRow(fund_id=metrics.fund_id)
        for field, value in metrics.model_dump(exclude={"fund_id"}).items():
            setattr(row, field, value)
        self.session.merge(row)

    def replace_document(self, document: RawDocument, chunks: Sequence[DocumentChunk]) -> None:
        existing = self.session.get(DocumentRow, document.document_id)
        if existing is not None:
            self.session.delete(existing)
            self.session.flush()
        row = DocumentRow(
            document_id=document.document_id,
            fund_id=document.fund_id,
            document_type=document.document_type,
            title=document.title,
            source=document.source,
            source_url=document.source_url,
            publication_date=document.publication_date,
            content=document.content,
        )
        row.chunks = [
            DocumentChunkRow(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                fund_id=c.metadata.fund_id,
                chunk_index=c.metadata.chunk_index,
                section=c.metadata.section,
                text=c.text,
                metadata_json=c.metadata.model_dump(mode="json"),
            )
            for c in chunks
        ]
        self.session.add(row)

    def commit(self) -> None:
        self.session.commit()

    # ------------------------------------------------------------- converters

    @staticmethod
    def row_to_fund_record(row: FundRow) -> FundRecord:
        return FundRecord(
            fund_id=row.fund_id,
            scheme_code=row.scheme_code,
            isin=row.isin,
            fund_name=row.fund_name,
            amc=row.amc,
            category=row.category,
            sub_category=row.sub_category,
            risk_level=row.risk_level,  # type: ignore[arg-type]
            expense_ratio=row.expense_ratio,
            expense_ratio_date=row.expense_ratio_date,
            aum_crores=row.aum_crores,
            min_sip_amount=row.min_sip_amount,
            min_lump_sum=row.min_lump_sum,
            lock_in_years=row.lock_in_years,
            exit_load=row.exit_load,
            benchmark=row.benchmark,
            investment_objective=row.investment_objective,
            fund_manager=row.fund_manager,
            scheme_type=row.scheme_type,
            plan_type=row.plan_type,  # type: ignore[arg-type]
            option_type=row.option_type,
            inception_date=row.inception_date,
            data_as_of=row.data_as_of,
            source_id=row.source_id,
            source_url=row.source_url,
        )

    @staticmethod
    def row_to_metrics_record(row: FundMetricsRow) -> FundMetricsRecord:
        return FundMetricsRecord(
            fund_id=row.fund_id,
            as_of_date=row.as_of_date,
            history_start=row.history_start,
            history_years=row.history_years,
            cagr_1y=row.cagr_1y,
            cagr_3y=row.cagr_3y,
            cagr_5y=row.cagr_5y,
            volatility_3y=row.volatility_3y,
            sharpe_3y=row.sharpe_3y,
            max_drawdown_3y=row.max_drawdown_3y,
            rolling_3y_positive_pct=row.rolling_3y_positive_pct,
            risk_free_rate_used=row.risk_free_rate_used,
            method_version=row.method_version,
        )

    @staticmethod
    def row_to_chunk(row: DocumentChunkRow) -> DocumentChunk:
        metadata = SourceMetadata.model_validate(row.metadata_json)
        return DocumentChunk(
            chunk_id=row.chunk_id,
            document_id=row.document_id,
            text=row.text,
            metadata=metadata,
        )
