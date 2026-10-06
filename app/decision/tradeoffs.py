"""
Pairwise trade-off generation from structured facts only.

Every statement cites the structured field it was derived from (``fact_basis``) so the
generator and the UI can show exactly which database values support the comparison. No
statement is produced when either side lacks the value - gaps are reported, not glossed over.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.decision.models import CandidateScore, Tradeoff
from app.ingestion.models import RISK_LEVEL_ORDER

# (field, label, unit formatter, higher_is_better, min relative difference, min absolute difference)
# Both thresholds must be exceeded before a difference is reported, so near-identical values
# (e.g. two liquid funds with ~0% drawdown) are not presented as an advantage.
METRIC_RULES: Sequence[Tuple[str, str, str, bool, float, float]] = (
    ("expense_ratio", "expense ratio", "pct", False, 0.05, 0.02),
    ("sharpe_3y", "3-year Sharpe ratio", "ratio", True, 0.05, 0.10),
    ("cagr_3y", "3-year CAGR", "pct_decimal", True, 0.05, 0.005),
    ("cagr_5y", "5-year CAGR", "pct_decimal", True, 0.05, 0.005),
    ("volatility_3y", "3-year volatility", "pct_decimal", False, 0.05, 0.005),
    ("max_drawdown_3y", "3-year maximum drawdown", "pct_decimal_abs", False, 0.05, 0.005),
    ("rolling_3y_positive_pct", "share of positive 3-year rolling windows", "pct_decimal", True, 0.03, 0.02),
    ("aum_crores", "AUM", "crore", True, 0.25, 100.0),
    ("min_sip_amount", "minimum SIP", "inr", False, 0.01, 1.0),
)


def _fact_map(candidate: CandidateScore) -> Dict[str, Any]:
    return {f.field: f.value for f in candidate.structured_facts}


def _fmt(value: float, unit: str) -> str:
    if unit == "pct":
        return f"{value:.2f}%"
    if unit == "pct_decimal":
        return f"{value * 100:.1f}%"
    if unit == "pct_decimal_abs":
        return f"{abs(value) * 100:.1f}%"
    if unit == "crore":
        return f"Rs {value:,.0f} crore"
    if unit == "inr":
        return f"Rs {value:,.0f}"
    return f"{value:.2f}"


def _rel_diff(a: float, b: float) -> float:
    base = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / base


def compare_pair(a: CandidateScore, b: CandidateScore) -> Tradeoff:
    """Build advantage lists for a vs b from structured facts."""
    fa, fb = _fact_map(a), _fact_map(b)
    adv_a: List[str] = []
    adv_b: List[str] = []
    basis: List[str] = []
    gaps: List[str] = []

    for field, label, unit, higher_better, min_rel, min_abs in METRIC_RULES:
        va, vb = fa.get(field), fb.get(field)
        if va is None or vb is None:
            gaps.append(label)
            continue
        if unit == "pct_decimal_abs":
            va_cmp, vb_cmp = abs(va), abs(vb)
        else:
            va_cmp, vb_cmp = va, vb
        if _rel_diff(va_cmp, vb_cmp) < min_rel or abs(va_cmp - vb_cmp) < min_abs:
            continue
        a_better = (va_cmp > vb_cmp) if higher_better else (va_cmp < vb_cmp)
        statement = f"{label} {_fmt(va, unit)} vs {_fmt(vb, unit)}"
        basis.append(field)
        if a_better:
            adv_a.append(f"{'Higher' if higher_better else 'Lower'} {statement}")
        else:
            adv_b.append(f"{'Higher' if higher_better else 'Lower'} {label} {_fmt(vb, unit)} vs {_fmt(va, unit)}")

    ra, rb = fa.get("risk_level"), fb.get("risk_level")
    if ra and rb and ra != rb:
        basis.append("risk_level")
        if RISK_LEVEL_ORDER[ra] < RISK_LEVEL_ORDER[rb]:
            adv_a.append(f"Lower Risk-o-meter level ({ra} vs {rb})")
        else:
            adv_b.append(f"Lower Risk-o-meter level ({rb} vs {ra})")

    if a.evidence_available and not b.evidence_available:
        adv_a.append("Verified document evidence was retrieved for this fund but not for the alternative")
    elif b.evidence_available and not a.evidence_available:
        adv_b.append("Verified document evidence was retrieved for this fund but not for the winner")

    summary_bits = [f"{a.fund_name} scored {a.final_score:.3f} vs {b.fund_name} at {b.final_score:.3f}."]
    if adv_a:
        summary_bits.append(f"{a.fund_name} leads on: {'; '.join(adv_a[:3])}.")
    if adv_b:
        summary_bits.append(f"{b.fund_name} leads on: {'; '.join(adv_b[:3])}.")
    if gaps:
        summary_bits.append(f"Not comparable (data missing for one side): {', '.join(gaps[:4])}.")

    return Tradeoff(
        fund_a=a.fund_name,
        fund_b=b.fund_name,
        advantages_a=adv_a,
        advantages_b=adv_b,
        summary=" ".join(summary_bits),
        fact_basis=basis,
    )


class TradeoffAnalyzer:
    """Generates winner-vs-alternative trade-offs for the top ranked candidates."""

    def __init__(self, max_alternatives: int = 3) -> None:
        self.max_alternatives = max_alternatives

    def analyze(self, ranked: Sequence[CandidateScore], winner: Optional[CandidateScore] = None) -> List[Tradeoff]:
        if not ranked:
            return []
        head = winner or ranked[0]
        alternatives = [c for c in ranked if c.fund_id != head.fund_id][: self.max_alternatives]
        return [compare_pair(head, alt) for alt in alternatives]
