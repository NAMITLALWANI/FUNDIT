"""
SQLAlchemy ORM table definitions for the mutual fund data layer.

The schema is dialect-neutral: it runs on SQLite for local development and on PostgreSQL in
deployment without changes.
"""

from datetime import date
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Date,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all tables."""


class DataSourceRow(Base):
    __tablename__ = "data_sources"

    source_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    url: Mapped[str] = mapped_column(String(512), nullable=False)
    collected_at: Mapped[date] = mapped_column(Date, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    license_notes: Mapped[str] = mapped_column(Text, nullable=False, default="")


class FundRow(Base):
    __tablename__ = "funds"

    fund_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    scheme_code: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    isin: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    fund_name: Mapped[str] = mapped_column(String(255), nullable=False)
    amc: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    sub_category: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    risk_level: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    expense_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    expense_ratio_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    aum_crores: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    min_sip_amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    min_lump_sum: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lock_in_years: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    exit_load: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    benchmark: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    investment_objective: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    fund_manager: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    scheme_type: Mapped[str] = mapped_column(String(32), nullable=False, default="Open Ended")
    plan_type: Mapped[str] = mapped_column(String(16), nullable=False, default="Direct")
    option_type: Mapped[str] = mapped_column(String(32), nullable=False, default="Growth")
    inception_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    data_as_of: Mapped[date] = mapped_column(Date, nullable=False)
    source_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("data_sources.source_id"), nullable=True
    )
    source_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    metrics: Mapped[Optional["FundMetricsRow"]] = relationship(
        back_populates="fund", uselist=False, cascade="all, delete-orphan"
    )
    performance: Mapped[List["FundPerformanceRow"]] = relationship(
        back_populates="fund", cascade="all, delete-orphan"
    )
    documents: Mapped[List["DocumentRow"]] = relationship(back_populates="fund")

    __table_args__ = (
        Index("ix_funds_category", "category"),
        Index("ix_funds_sub_category", "sub_category"),
        Index("ix_funds_risk_level", "risk_level"),
        Index("ix_funds_expense_ratio", "expense_ratio"),
    )


class FundPerformanceRow(Base):
    __tablename__ = "fund_performance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fund_id: Mapped[str] = mapped_column(String(64), ForeignKey("funds.fund_id"), nullable=False)
    nav_date: Mapped[date] = mapped_column(Date, nullable=False)
    nav: Mapped[float] = mapped_column(Float, nullable=False)

    fund: Mapped[FundRow] = relationship(back_populates="performance")

    __table_args__ = (
        UniqueConstraint("fund_id", "nav_date", name="uq_fund_performance_fund_date"),
        Index("ix_fund_performance_fund_date", "fund_id", "nav_date"),
    )


class FundMetricsRow(Base):
    __tablename__ = "fund_metrics"

    fund_id: Mapped[str] = mapped_column(String(64), ForeignKey("funds.fund_id"), primary_key=True)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    history_start: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    history_years: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cagr_1y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cagr_3y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cagr_5y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    volatility_3y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sharpe_3y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_drawdown_3y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rolling_3y_positive_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    risk_free_rate_used: Mapped[float] = mapped_column(Float, nullable=False)
    method_version: Mapped[str] = mapped_column(String(16), nullable=False, default="v1")

    fund: Mapped[FundRow] = relationship(back_populates="metrics")


class DocumentRow(Base):
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    fund_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("funds.fund_id"), nullable=True)
    document_type: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[str] = mapped_column(String(255), nullable=False)
    source_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    publication_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    fund: Mapped[Optional[FundRow]] = relationship(back_populates="documents")
    chunks: Mapped[List["DocumentChunkRow"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunkRow(Base):
    __tablename__ = "document_chunks"

    chunk_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("documents.document_id"), nullable=False
    )
    fund_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    section: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    document: Mapped[DocumentRow] = relationship(back_populates="chunks")

    __table_args__ = (Index("ix_document_chunks_fund", "fund_id"),)
