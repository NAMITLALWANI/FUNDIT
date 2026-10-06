"""
End-to-end ingestion coordinator: raw snapshot -> validated records -> metrics -> SQL -> chunks.

Steps
1. Load the curated fund universe and each fund's NAV + attribute snapshot.
2. Validate and normalise into ``FundRecord`` / ``NavPoint`` objects, collecting quality issues.
3. Compute derived metrics from the NAV series (never from provider-reported returns).
4. Generate a provenance-annotated "scheme summary" Markdown document per fund from the
   structured disclosures, so the hybrid retriever has fund-specific textual evidence.
5. Load regulatory documents, chunk everything section-aware, and persist to the database.
6. Emit a data-quality report. Problem records are reported, not silently dropped.
"""

import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import IngestionError
from app.core.logging import get_logger
from app.db.repository import FundRepository
from app.ingestion.chunkers.chunker import SentenceAwareChunker
from app.ingestion.cleaners.cleaner import TextCleaner
from app.ingestion.loaders.document_loader import MarkdownDocumentLoader
from app.ingestion.loaders.fund_data_loader import FundDataLoader
from app.ingestion.metrics import compute_fund_metrics
from app.ingestion.models import (
    DataQualityIssue,
    DataQualityReport,
    DataSourceRecord,
    DocumentChunk,
    FundMetricsRecord,
    FundRecord,
    NavPoint,
    RawDocument,
)

logger = get_logger(__name__)


def _fmt_pct(value: Optional[float]) -> str:
    return "not disclosed in snapshot" if value is None else f"{value:.2f}%"


def _fmt_inr(value: Optional[float]) -> str:
    return "not disclosed in snapshot" if value is None else f"Rs {value:,.0f}"


def build_scheme_summary(fund: FundRecord, metrics: Optional[FundMetricsRecord]) -> RawDocument:
    """Create a scheme summary document from structured disclosures (clearly labelled as derived)."""
    lines: List[str] = []
    lines.append("## Investment objective")
    if fund.investment_objective:
        lines.append(fund.investment_objective.strip())
    else:
        lines.append("The investment objective is not available in the public data snapshot for this scheme.")

    lines.append("")
    lines.append("## Scheme classification and risk")
    risk_text = (
        f"The SEBI Risk-o-meter level disclosed for the scheme is {fund.risk_level}."
        if fund.risk_level
        else "The Risk-o-meter level is not available in the public data snapshot."
    )
    lines.append(
        f"{fund.fund_name} is an {fund.scheme_type.lower()} {fund.category.lower()} scheme managed by {fund.amc}"
        + (f" in the SEBI {fund.sub_category} category." if fund.sub_category else ".")
        + f" {risk_text} The data refers to the {fund.plan_type} Plan, {fund.option_type} option."
    )

    lines.append("")
    lines.append("## Investment terms")
    terms = [
        f"Minimum SIP instalment: {_fmt_inr(fund.min_sip_amount)}.",
        f"Minimum lump sum investment: {_fmt_inr(fund.min_lump_sum)}.",
    ]
    if fund.lock_in_years:
        terms.append(f"Statutory lock-in period: {fund.lock_in_years:g} years.")
    else:
        terms.append("No statutory lock-in period applies." if fund.lock_in_years == 0 else "Lock-in information is not available in the snapshot.")
    terms.append("Exit load terms are not included in the public data snapshot; refer to the Scheme Information Document.")
    lines.append(" ".join(terms))

    lines.append("")
    lines.append("## Costs and size")
    er_date = f" as of {fund.expense_ratio_date.isoformat()}" if fund.expense_ratio_date else ""
    aum_text = (
        f"Assets under management were approximately Rs {fund.aum_crores:,.0f} crore."
        if fund.aum_crores is not None
        else "Assets under management are not available in the snapshot."
    )
    lines.append(f"The total expense ratio of the Direct Plan was {_fmt_pct(fund.expense_ratio)}{er_date}. {aum_text}")

    lines.append("")
    lines.append("## Management and history")
    mgmt = []
    if fund.fund_manager:
        mgmt.append(f"Fund manager(s): {fund.fund_manager}.")
    if fund.inception_date:
        mgmt.append(f"Direct Plan NAV history in the snapshot starts on {fund.inception_date.isoformat()}.")
    if metrics and metrics.history_years is not None:
        mgmt.append(f"The snapshot covers {metrics.history_years:.1f} years of NAV history up to {metrics.as_of_date.isoformat()}.")
    lines.append(" ".join(mgmt) if mgmt else "Management details are not available in the snapshot.")

    lines.append("")
    lines.append("## Provenance note")
    lines.append(
        "This summary is generated from structured public disclosures (AMFI NAV data and scheme attributes aggregated "
        "from AMC factsheets). Values are point-in-time and should be verified against the AMC's official scheme documents. "
        "Past performance does not guarantee future results."
    )

    return RawDocument(
        document_id=f"{fund.fund_id}-scheme-summary",
        document_type="SchemeSummary",
        title=f"{fund.fund_name} - Scheme Summary",
        source="AMC disclosures via Kuvera scheme API and AMFI",
        source_url=fund.source_url,
        fund_id=fund.fund_id,
        publication_date=fund.expense_ratio_date or fund.data_as_of,
        content="\n".join(lines),
        metadata={"generated": True, "generator": "app.ingestion.pipeline.build_scheme_summary"},
    )


class IngestionPipeline:
    """Coordinates loading, validation, metric computation, document generation and persistence."""

    def __init__(
        self,
        session: Session,
        raw_dir: Path,
        documents_dir: Path,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repo = FundRepository(session)
        self.raw_dir = Path(raw_dir)
        self.documents_dir = Path(documents_dir)
        self.fund_loader = FundDataLoader(self.raw_dir)
        self.document_loader = MarkdownDocumentLoader()
        self.cleaner = TextCleaner()
        self.chunker = SentenceAwareChunker(
            chunk_size=self.settings.chunk_size, chunk_overlap=self.settings.chunk_overlap
        )

    # ------------------------------------------------------------------ steps

    def load_sources(self) -> List[DataSourceRecord]:
        sources_path = self.raw_dir / "sources.json"
        if not sources_path.exists():
            raise IngestionError(f"sources.json missing in {self.raw_dir}; run scripts/fetch_fund_data.py first")
        payload = json.loads(sources_path.read_text(encoding="utf-8"))
        return [DataSourceRecord.model_validate(s) for s in payload["sources"]]

    def ingest_funds(self) -> tuple[List[FundRecord], Dict[str, FundMetricsRecord], int, List[DataQualityIssue]]:
        universe = self.fund_loader.load_universe()
        known_sources = {s.source_id for s in self.repo.list_sources()}
        funds: List[FundRecord] = []
        metrics_by_fund: Dict[str, FundMetricsRecord] = {}
        issues: List[DataQualityIssue] = []
        nav_total = 0

        for entry in universe:
            code = int(entry["amfi_scheme_code"])
            try:
                meta, points, nav_issues = self.fund_loader.load_nav_history(code)
            except IngestionError as exc:
                issues.append(DataQualityIssue(fund_id=entry.get("fund_id"), severity="error", field="nav", message=str(exc)))
                continue
            issues.extend(nav_issues)
            attributes = self.fund_loader.load_attributes(code)
            record, record_issues = self.fund_loader.build_fund_record(entry, meta, attributes, points)
            issues.extend(record_issues)
            if record.source_id and record.source_id not in known_sources:
                issues.append(DataQualityIssue(fund_id=record.fund_id, severity="info", field="source_id", message=f"Source '{record.source_id}' not declared in sources.json; provenance link left empty"))
                record = record.model_copy(update={"source_id": None})

            if points:
                metrics = compute_fund_metrics(record.fund_id, points, risk_free_rate=self.settings.risk_free_rate)
                metrics_by_fund[record.fund_id] = metrics
                for field in ("cagr_3y", "cagr_5y", "volatility_3y", "sharpe_3y"):
                    if getattr(metrics, field) is None:
                        issues.append(DataQualityIssue(fund_id=record.fund_id, severity="info", field=field, message="Insufficient NAV history for this window; metric stored as NULL"))
            else:
                issues.append(DataQualityIssue(fund_id=record.fund_id, severity="error", field="nav", message="No usable NAV points; metrics not computed"))

            self.repo.upsert_fund(record)
            self.repo.session.flush()
            nav_total += self.repo.replace_nav_history(record.fund_id, points)
            if record.fund_id in metrics_by_fund:
                self.repo.upsert_metrics(metrics_by_fund[record.fund_id])
            funds.append(record)

        return funds, metrics_by_fund, nav_total, issues

    def generate_scheme_documents(self, funds: Sequence[FundRecord], metrics: Dict[str, FundMetricsRecord]) -> List[Path]:
        target_dir = self.documents_dir / "funds"
        target_dir.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []
        for fund in funds:
            doc = build_scheme_summary(fund, metrics.get(fund.fund_id))
            front_matter = "\n".join(
                [
                    "---",
                    f"document_id: {doc.document_id}",
                    f"document_type: {doc.document_type}",
                    f"title: {doc.title}",
                    f"source: {doc.source}",
                    f"source_url: {doc.source_url or ''}",
                    f"publication_date: {doc.publication_date.isoformat() if doc.publication_date else ''}",
                    f"fund_id: {doc.fund_id}",
                    "---",
                ]
            )
            path = target_dir / f"{fund.fund_id}.md"
            path.write_text(front_matter + "\n" + doc.content + "\n", encoding="utf-8")
            written.append(path)
        return written

    def ingest_documents(self) -> tuple[int, List[DocumentChunk], List[DataQualityIssue]]:
        issues: List[DataQualityIssue] = []
        documents = self.document_loader.load(self.documents_dir)
        all_chunks: List[DocumentChunk] = []
        known_fund_ids = {f.fund_id for f in self.repo.list_funds()}
        for doc in documents:
            if doc.fund_id and doc.fund_id not in known_fund_ids:
                issues.append(DataQualityIssue(fund_id=doc.fund_id, severity="error", field="document", message=f"Document {doc.document_id} references unknown fund; skipped"))
                continue
            cleaned = doc.model_copy(update={"content": self.cleaner.clean(doc.content)})
            chunks = self.chunker.chunk_document(cleaned)
            if not chunks:
                issues.append(DataQualityIssue(fund_id=doc.fund_id, severity="warning", field="document", message=f"Document {doc.document_id} produced no chunks"))
                continue
            self.repo.replace_document(cleaned, chunks)
            all_chunks.extend(chunks)
        return len(documents), all_chunks, issues

    # -------------------------------------------------------------------- run

    def run(self) -> DataQualityReport:
        """Execute the full ingestion and return a data-quality report."""
        logger.info("Starting mutual fund ingestion from %s", self.raw_dir)
        for source in self.load_sources():
            self.repo.upsert_source(source)
        self.repo.session.flush()

        funds, metrics, nav_total, issues = self.ingest_funds()
        self.repo.commit()

        self.generate_scheme_documents(funds, metrics)
        doc_count, chunks, doc_issues = self.ingest_documents()
        issues.extend(doc_issues)
        self.repo.commit()

        gaps = {i.fund_id for i in issues if i.severity == "warning" and i.fund_id}
        report = DataQualityReport(
            generated_at=date.today(),
            funds_total=len(self.fund_loader.load_universe()),
            funds_loaded=len(funds),
            funds_with_attribute_gaps=len(gaps),
            nav_points_loaded=nav_total,
            documents_loaded=doc_count,
            chunks_created=len(chunks),
            issues=issues,
        )
        logger.info(
            "Ingestion complete: %d funds, %d NAV points, %d documents, %d chunks, %d issues",
            report.funds_loaded, report.nav_points_loaded, report.documents_loaded, report.chunks_created, len(issues),
        )
        return report
