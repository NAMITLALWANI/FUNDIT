"""
EvidenceValidator: turns reranked chunks into verified, fund-bound evidence.

Checks performed
1. Provenance: every chunk must carry document_id, document_type, source and a fund_id that is
   either in the candidate set or None (universe-wide regulatory context). Chunks that reference
   funds outside the candidate set are discarded and counted - they can never become citations.
2. Structured consistency: numeric statements in scheme-summary chunks (expense ratio, AUM,
   minimum SIP, risk level) are compared with the database row. A mismatch marks the chunk as
   CONFLICTING so the generator will not cite it for that fact and confidence is lowered.
3. Freshness: publication_date vs. the configured freshness window.

Document chunks never override structured values; conflicts are surfaced, not resolved by
picking one side.
"""

import re
from datetime import date
from typing import Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.decision.models import EvidenceRef
from app.ingestion.models import FundRecord
from app.reranking.models import RerankedResult

logger = get_logger(__name__)

EvidenceStatus = Literal["VERIFIED", "CONFLICTING", "UNBOUND"]

EXPENSE_RE = re.compile(r"expense ratio[^.]*?was\s+(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
AUM_RE = re.compile(r"Rs\s+([\d,]+)\s+crore", re.IGNORECASE)
SIP_RE = re.compile(r"Minimum SIP instalment:\s*Rs\s+([\d,]+)", re.IGNORECASE)
RISK_RE = re.compile(r"Risk-o-meter level disclosed for the scheme is\s+([A-Za-z ]+?)\.", re.IGNORECASE)


class ChunkCheck(BaseModel):
    chunk_id: str
    status: EvidenceStatus
    fund_id: Optional[str]
    notes: List[str] = Field(default_factory=list)
    is_stale: bool = False


class EvidenceBundle(BaseModel):
    """Verified evidence for one fund."""

    fund_id: str
    evidence: List[EvidenceRef] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.evidence)


class ValidationReport(BaseModel):
    """Summary of a validation pass for the decision trace."""

    chunks_in: int
    verified: int
    conflicting: int
    unbound_discarded: int
    stale: int
    global_context: int
    funds_with_evidence: int
    checks: List[ChunkCheck] = Field(default_factory=list)


class EvidenceValidator:
    """Validates reranked chunks and binds them to candidate funds."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def _consistency_notes(self, text: str, fund: FundRecord) -> List[str]:
        notes: List[str] = []
        m = EXPENSE_RE.search(text)
        if m and fund.expense_ratio is not None and abs(float(m.group(1)) - fund.expense_ratio) > 0.011:
            notes.append(f"expense ratio {m.group(1)}% in document vs {fund.expense_ratio}% in database")
        m = SIP_RE.search(text)
        if m and fund.min_sip_amount is not None and abs(float(m.group(1).replace(',', '')) - fund.min_sip_amount) > 0.5:
            notes.append(f"minimum SIP Rs {m.group(1)} in document vs Rs {fund.min_sip_amount:,.0f} in database")
        m = AUM_RE.search(text)
        if m and fund.aum_crores is not None:
            doc_aum = float(m.group(1).replace(",", ""))
            if abs(doc_aum - fund.aum_crores) / max(fund.aum_crores, 1.0) > 0.02:
                notes.append(f"AUM Rs {m.group(1)} crore in document vs Rs {fund.aum_crores:,.0f} crore in database")
        m = RISK_RE.search(text)
        if m and fund.risk_level is not None and m.group(1).strip().lower() != fund.risk_level.lower():
            notes.append(f"risk level '{m.group(1).strip()}' in document vs '{fund.risk_level}' in database")
        return notes

    def validate(
        self,
        reranked: Sequence[RerankedResult],
        candidates: Sequence[FundRecord],
        today: Optional[date] = None,
    ) -> tuple[Dict[str, EvidenceBundle], List[EvidenceRef], ValidationReport]:
        """Return (bundles by fund_id, global context refs, report)."""
        today = today or date.today()
        by_id = {f.fund_id: f for f in candidates}
        bundles: Dict[str, EvidenceBundle] = {fid: EvidenceBundle(fund_id=fid) for fid in by_id}
        global_context: List[EvidenceRef] = []
        checks: List[ChunkCheck] = []
        counts = {"verified": 0, "conflicting": 0, "unbound": 0, "stale": 0}

        for item in reranked:
            chunk = item.chunk
            meta = chunk.metadata
            fund_id = meta.fund_id
            is_stale = bool(meta.publication_date and (today - meta.publication_date).days > self.settings.data_freshness_days)
            if is_stale:
                counts["stale"] += 1

            if not chunk.chunk_id or not meta.document_id or not meta.source:
                checks.append(ChunkCheck(chunk_id=chunk.chunk_id or "?", status="UNBOUND", fund_id=fund_id, notes=["missing provenance"]))
                counts["unbound"] += 1
                continue
            if fund_id is not None and fund_id not in by_id:
                checks.append(ChunkCheck(chunk_id=chunk.chunk_id, status="UNBOUND", fund_id=fund_id, notes=["fund outside candidate set"], is_stale=is_stale))
                counts["unbound"] += 1
                continue

            ref = EvidenceRef(
                chunk_id=chunk.chunk_id,
                document_id=meta.document_id,
                document_type=meta.document_type,
                title=meta.title,
                source_url=meta.source_url,
                publication_date=meta.publication_date,
                relative_relevance=item.relative_relevance if item.relative_relevance is not None else 0.0,
                reranker_score=item.reranker_score,
                snippet=chunk.text[:400],
            )

            if fund_id is None:
                global_context.append(ref)
                checks.append(ChunkCheck(chunk_id=chunk.chunk_id, status="VERIFIED", fund_id=None, notes=["universe-wide context"], is_stale=is_stale))
                counts["verified"] += 1
                continue

            notes = self._consistency_notes(chunk.text, by_id[fund_id])
            if notes:
                bundles[fund_id].conflicts.extend(notes)
                checks.append(ChunkCheck(chunk_id=chunk.chunk_id, status="CONFLICTING", fund_id=fund_id, notes=notes, is_stale=is_stale))
                counts["conflicting"] += 1
                continue

            bundles[fund_id].evidence.append(ref)
            checks.append(ChunkCheck(chunk_id=chunk.chunk_id, status="VERIFIED", fund_id=fund_id, is_stale=is_stale))
            counts["verified"] += 1

        report = ValidationReport(
            chunks_in=len(reranked),
            verified=counts["verified"],
            conflicting=counts["conflicting"],
            unbound_discarded=counts["unbound"],
            stale=counts["stale"],
            global_context=len(global_context),
            funds_with_evidence=sum(1 for b in bundles.values() if b.evidence),
            checks=checks,
        )
        logger.info(
            "Evidence validation: in=%d verified=%d conflicting=%d unbound=%d funds_with_evidence=%d",
            report.chunks_in, report.verified, report.conflicting, report.unbound_discarded, report.funds_with_evidence,
        )
        return bundles, global_context, report
