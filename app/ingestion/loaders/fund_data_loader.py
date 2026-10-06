"""
Loader for structured fund data snapshots in data/raw/ (NAV history + scheme attributes).

Every transformation from raw provider fields to the domain model is explicit here so the
data lineage can be audited. Missing attributes are kept as ``None`` and reported as data
quality issues; they are never filled with guesses.
"""

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from app.core.exceptions import IngestionError
from app.ingestion.models import DataQualityIssue, FundRecord, NavPoint

CATEGORY_PREFIX_MAP = {
    "equity": "Equity",
    "debt": "Debt",
    "income/debt oriented": "Debt",
    "hybrid": "Hybrid",
    "other": "Other",
    "solution oriented": "Solution Oriented",
}

SUB_CATEGORY_NORMALISATION = {
    "ELSS- Tax Saver Fund": "ELSS",
    "ELSS": "ELSS",
    "Sectoral/ Thematic": "Sectoral/Thematic",
    "Sectoral/Thematic": "Sectoral/Thematic",
    "Short Term Fund": "Short Duration Fund",
    "Short Duration Fund": "Short Duration Fund",
    "Index Funds": "Index Fund",
    "Index Fund": "Index Fund",
}

RISK_LABEL_MAP = {
    "low risk": "Low",
    "low to moderate risk": "Low to Moderate",
    "moderate risk": "Moderate",
    "moderately high risk": "Moderately High",
    "high risk": "High",
    "very high risk": "Very High",
}

REQUIRED_ATTRIBUTE_FIELDS = ("risk_level", "expense_ratio", "aum_crores", "min_sip_amount", "min_lump_sum")


def parse_amfi_category(scheme_category: str) -> Tuple[str, Optional[str]]:
    """Split an AMFI category string like 'Equity Scheme - Flexi Cap Fund' into (category, sub_category)."""
    if not scheme_category:
        return "Unknown", None
    parts = [p.strip() for p in scheme_category.split(" - ", 1)]
    prefix = parts[0].lower()
    category = "Unknown"
    for key, value in CATEGORY_PREFIX_MAP.items():
        if prefix.startswith(key):
            category = value
            break
    sub_category = parts[1] if len(parts) > 1 else None
    if sub_category:
        sub_category = SUB_CATEGORY_NORMALISATION.get(sub_category, sub_category)
    return category, sub_category


def normalise_risk_label(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    return RISK_LABEL_MAP.get(raw.strip().lower())


def _parse_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_date(value: Optional[str], fmt: str) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.strptime(value, fmt).date()
    except ValueError:
        return None


def _plan_and_option(scheme_name: str, attributes: Dict[str, Any]) -> Tuple[str, str]:
    name = scheme_name.lower()
    plan = "Direct" if "direct" in name or attributes.get("direct") == "Y" else "Regular"
    option = "Growth" if re.search(r"growth", name) else "IDCW"
    return plan, option


class FundDataLoader:
    """Loads the raw NAV and attribute snapshots for the curated fund universe."""

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = Path(raw_dir)
        self.universe_path = self.raw_dir / "fund_universe.json"
        self.nav_dir = self.raw_dir / "nav"
        self.attr_dir = self.raw_dir / "scheme_attributes"

    def load_universe(self) -> List[Dict[str, Any]]:
        if not self.universe_path.exists():
            raise IngestionError(f"Fund universe file not found: {self.universe_path}")
        return json.loads(self.universe_path.read_text(encoding="utf-8"))["funds"]

    def load_nav_history(self, scheme_code: int) -> Tuple[Dict[str, Any], List[NavPoint], List[DataQualityIssue]]:
        path = self.nav_dir / f"{scheme_code}.json"
        if not path.exists():
            raise IngestionError(f"NAV snapshot missing for scheme {scheme_code}: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        issues: List[DataQualityIssue] = []
        points: List[NavPoint] = []
        seen: set = set()
        fund_id = f"MF-{scheme_code}"
        for row in payload.get("data", []):
            nav_date = _parse_date(row.get("date"), "%d-%m-%Y")
            nav = _parse_float(row.get("nav"))
            if nav_date is None or nav is None:
                issues.append(DataQualityIssue(fund_id=fund_id, severity="warning", field="nav", message=f"Unparseable NAV row {row}"))
                continue
            if nav <= 0:
                issues.append(DataQualityIssue(fund_id=fund_id, severity="warning", field="nav", message=f"Non-positive NAV on {nav_date}"))
                continue
            if nav_date in seen:
                issues.append(DataQualityIssue(fund_id=fund_id, severity="info", field="nav", message=f"Duplicate NAV date {nav_date} ignored"))
                continue
            seen.add(nav_date)
            points.append(NavPoint(date=nav_date, nav=nav))
        points.sort(key=lambda p: p.date)
        if len(points) < 250:
            issues.append(DataQualityIssue(fund_id=fund_id, severity="warning", field="nav", message=f"Only {len(points)} NAV points available"))
        return payload.get("meta", {}), points, issues

    def load_attributes(self, scheme_code: int) -> Optional[Dict[str, Any]]:
        path = self.attr_dir / f"{scheme_code}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def build_fund_record(
        self,
        universe_entry: Dict[str, Any],
        meta: Dict[str, Any],
        attributes: Optional[Dict[str, Any]],
        nav_points: List[NavPoint],
    ) -> Tuple[FundRecord, List[DataQualityIssue]]:
        """Combine AMFI metadata and scheme attributes into a validated FundRecord."""
        scheme_code = int(universe_entry["amfi_scheme_code"])
        fund_id = universe_entry.get("fund_id") or f"MF-{scheme_code}"
        issues: List[DataQualityIssue] = []
        attributes = attributes or {}

        category, sub_category = parse_amfi_category(meta.get("scheme_category", ""))
        if category == "Unknown":
            issues.append(DataQualityIssue(fund_id=fund_id, severity="warning", field="category", message=f"Unrecognised AMFI category '{meta.get('scheme_category')}'"))

        scheme_name = meta.get("scheme_name") or universe_entry.get("scheme_name") or ""
        plan, option = _plan_and_option(scheme_name, attributes)

        risk_level = normalise_risk_label(attributes.get("crisil_rating"))
        if attributes.get("crisil_rating") and risk_level is None:
            issues.append(DataQualityIssue(fund_id=fund_id, severity="warning", field="risk_level", message=f"Unmapped riskometer label '{attributes.get('crisil_rating')}'"))

        # Kuvera reports AUM in INR million (1 crore = 10 million); verified against AMC-published AUM
        # figures for several schemes (see docs/data.md).
        aum_million = _parse_float(attributes.get("aum"))
        aum_crores = round(aum_million / 10.0, 2) if aum_million is not None else None

        inception = _parse_date(attributes.get("start_date"), "%Y-%m-%d") or (nav_points[0].date if nav_points else None)
        data_as_of = nav_points[-1].date if nav_points else date.today()

        source_url = attributes.get("detail_info") or None
        source_block = attributes.get("_source") or meta.get("_source") or {}
        source_id = source_block.get("source_id") or ("kuvera_scheme_api" if attributes else "amfi_mfapi")

        try:
            record = FundRecord(
                fund_id=fund_id,
                scheme_code=scheme_code,
                isin=meta.get("isin_growth") or universe_entry.get("isin"),
                fund_name=scheme_name,
                amc=meta.get("fund_house") or attributes.get("fund_name") or "Unknown AMC",
                category=category,
                sub_category=sub_category,
                risk_level=risk_level,  # type: ignore[arg-type]
                expense_ratio=_parse_float(attributes.get("expense_ratio")),
                expense_ratio_date=_parse_date(attributes.get("expense_ratio_date"), "%Y-%m-%d"),
                aum_crores=aum_crores,
                min_sip_amount=_parse_float(attributes.get("sip_min")),
                min_lump_sum=_parse_float(attributes.get("lump_min")),
                lock_in_years=_parse_float(attributes.get("lock_in_period")),
                exit_load=None,
                benchmark=None,
                investment_objective=attributes.get("investment_objective") or None,
                fund_manager=attributes.get("fund_manager") or None,
                scheme_type=(attributes.get("maturity_type") or meta.get("scheme_type", "Open Ended")).replace(" Schemes", ""),
                plan_type=plan,  # type: ignore[arg-type]
                option_type=option,
                inception_date=inception,
                data_as_of=data_as_of,
                source_id=source_id,
                source_url=source_url,
            )
        except ValidationError as exc:
            raise IngestionError(f"Invalid fund record for {fund_id}: {exc}") from exc

        for field in REQUIRED_ATTRIBUTE_FIELDS:
            if getattr(record, field) is None:
                issues.append(DataQualityIssue(fund_id=fund_id, severity="warning", field=field, message="Attribute unavailable from public sources; stored as NULL (UNKNOWN)"))
        for field in ("exit_load", "benchmark"):
            issues.append(DataQualityIssue(fund_id=fund_id, severity="info", field=field, message="Not provided by the snapshot sources; constraints on this field evaluate to UNKNOWN"))

        return record, issues
