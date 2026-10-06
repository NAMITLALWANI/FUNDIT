"""
Tri-state hard-constraint evaluation: PASS / FAIL / UNKNOWN.

Rules
* A constraint is evaluated only if the user set it.
* If the fund attribute needed for the check is NULL, the result is UNKNOWN - never PASS.
* Aggregate: any FAIL => FAIL; otherwise any UNKNOWN => UNKNOWN; otherwise PASS.
"""

from typing import Any, Callable, List, Optional

from app.decision.models import ConstraintEvaluation, ConstraintResult
from app.ingestion.models import RISK_LEVEL_ORDER, FundMetricsRecord, FundRecord
from app.query.models import QueryConstraints


def _unknown(field: str, expected: Any, reason: str) -> ConstraintResult:
    return ConstraintResult(field=field, status="UNKNOWN", expected=expected, actual=None, reason=reason)


def _check_upper_bound(field: str, actual: Optional[float], limit: float, label: str, unit: str = "") -> ConstraintResult:
    if actual is None:
        return _unknown(field, limit, f"{label} is not available in the data; cannot verify the limit of {limit:g}{unit}")
    if actual <= limit:
        return ConstraintResult(field=field, status="PASS", expected=limit, actual=actual, reason=f"{label} {actual:g}{unit} is within the limit of {limit:g}{unit}")
    return ConstraintResult(field=field, status="FAIL", expected=limit, actual=actual, reason=f"{label} {actual:g}{unit} exceeds the limit of {limit:g}{unit}")


def _check_lower_bound(field: str, actual: Optional[float], minimum: float, label: str, unit: str = "", fmt: Callable[[float], str] = lambda v: f"{v:g}") -> ConstraintResult:
    if actual is None:
        return _unknown(field, minimum, f"{label} is not available in the data; cannot verify the minimum of {fmt(minimum)}{unit}")
    if actual >= minimum:
        return ConstraintResult(field=field, status="PASS", expected=minimum, actual=actual, reason=f"{label} {fmt(actual)}{unit} meets the minimum of {fmt(minimum)}{unit}")
    return ConstraintResult(field=field, status="FAIL", expected=minimum, actual=actual, reason=f"{label} {fmt(actual)}{unit} is below the minimum of {fmt(minimum)}{unit}")


def evaluate_constraints(
    fund: FundRecord,
    metrics: Optional[FundMetricsRecord],
    constraints: QueryConstraints,
) -> ConstraintEvaluation:
    """Evaluate every user-specified hard constraint against structured fund data."""
    results: List[ConstraintResult] = []

    if constraints.sip_amount is not None:
        results.append(_check_upper_bound("sip_amount", fund.min_sip_amount, constraints.sip_amount, "Minimum SIP amount", " INR"))

    if constraints.lump_sum_amount is not None:
        results.append(_check_upper_bound("lump_sum_amount", fund.min_lump_sum, constraints.lump_sum_amount, "Minimum lump sum", " INR"))

    if constraints.max_expense_ratio is not None:
        results.append(_check_upper_bound("max_expense_ratio", fund.expense_ratio, constraints.max_expense_ratio, "Expense ratio", "%"))

    if constraints.max_risk_level is not None:
        limit_rank = RISK_LEVEL_ORDER[constraints.max_risk_level]
        if fund.risk_level is None:
            results.append(_unknown("max_risk_level", constraints.max_risk_level, "SEBI Risk-o-meter level is not available in the data; cannot verify the risk ceiling"))
        elif RISK_LEVEL_ORDER[fund.risk_level] <= limit_rank:
            results.append(ConstraintResult(field="max_risk_level", status="PASS", expected=constraints.max_risk_level, actual=fund.risk_level, reason=f"Risk-o-meter level '{fund.risk_level}' is within the ceiling '{constraints.max_risk_level}'"))
        else:
            results.append(ConstraintResult(field="max_risk_level", status="FAIL", expected=constraints.max_risk_level, actual=fund.risk_level, reason=f"Risk-o-meter level '{fund.risk_level}' exceeds the ceiling '{constraints.max_risk_level}'"))

    if constraints.categories:
        if fund.category in constraints.categories:
            results.append(ConstraintResult(field="categories", status="PASS", expected=constraints.categories, actual=fund.category, reason=f"Category '{fund.category}' is among the requested categories"))
        else:
            results.append(ConstraintResult(field="categories", status="FAIL", expected=constraints.categories, actual=fund.category, reason=f"Category '{fund.category}' is not among the requested categories {constraints.categories}"))

    if constraints.sub_categories:
        if fund.sub_category is None:
            results.append(_unknown("sub_categories", constraints.sub_categories, "SEBI sub-category is not available in the data"))
        elif fund.sub_category in constraints.sub_categories:
            results.append(ConstraintResult(field="sub_categories", status="PASS", expected=constraints.sub_categories, actual=fund.sub_category, reason=f"Sub-category '{fund.sub_category}' matches the request"))
        else:
            results.append(ConstraintResult(field="sub_categories", status="FAIL", expected=constraints.sub_categories, actual=fund.sub_category, reason=f"Sub-category '{fund.sub_category}' is not among {constraints.sub_categories}"))

    if constraints.min_aum_crores is not None:
        results.append(_check_lower_bound("min_aum_crores", fund.aum_crores, constraints.min_aum_crores, "AUM", " crore", fmt=lambda v: f"{v:,.0f}"))

    if constraints.max_lock_in_years is not None:
        if fund.lock_in_years is None:
            results.append(_unknown("max_lock_in_years", constraints.max_lock_in_years, "Lock-in period is not available in the data; cannot verify it fits the horizon"))
        elif fund.lock_in_years <= constraints.max_lock_in_years:
            results.append(ConstraintResult(field="max_lock_in_years", status="PASS", expected=constraints.max_lock_in_years, actual=fund.lock_in_years, reason=f"Lock-in of {fund.lock_in_years:g} years fits within the {constraints.max_lock_in_years:g}-year horizon"))
        else:
            results.append(ConstraintResult(field="max_lock_in_years", status="FAIL", expected=constraints.max_lock_in_years, actual=fund.lock_in_years, reason=f"Lock-in of {fund.lock_in_years:g} years exceeds the {constraints.max_lock_in_years:g}-year horizon"))

    if constraints.exit_load_free:
        if not fund.exit_load:
            results.append(_unknown("exit_load_free", True, "Exit load terms are not available in the data; cannot verify there is no exit load"))
        elif fund.exit_load.strip().lower() in {"nil", "none", "0", "0%", "no exit load"}:
            results.append(ConstraintResult(field="exit_load_free", status="PASS", expected=True, actual=fund.exit_load, reason="Scheme discloses no exit load"))
        else:
            results.append(ConstraintResult(field="exit_load_free", status="FAIL", expected=True, actual=fund.exit_load, reason=f"Scheme has an exit load: {fund.exit_load}"))

    if constraints.min_return_3y is not None:
        actual = metrics.cagr_3y if metrics else None
        results.append(
            _check_lower_bound(
                "min_return_3y", actual, constraints.min_return_3y, "Historical 3-year CAGR",
                fmt=lambda v: f"{v * 100:.1f}%",
            )
        )

    if constraints.required_amcs:
        amc_tokens = fund.amc.lower()
        if any(req.lower() in amc_tokens for req in constraints.required_amcs):
            results.append(ConstraintResult(field="required_amcs", status="PASS", expected=constraints.required_amcs, actual=fund.amc, reason=f"AMC '{fund.amc}' matches the required fund house"))
        else:
            results.append(ConstraintResult(field="required_amcs", status="FAIL", expected=constraints.required_amcs, actual=fund.amc, reason=f"AMC '{fund.amc}' is not among the required fund houses {constraints.required_amcs}"))

    if constraints.plan_type is not None:
        if fund.plan_type == constraints.plan_type:
            results.append(ConstraintResult(field="plan_type", status="PASS", expected=constraints.plan_type, actual=fund.plan_type, reason=f"{fund.plan_type} plan as requested"))
        else:
            results.append(ConstraintResult(field="plan_type", status="FAIL", expected=constraints.plan_type, actual=fund.plan_type, reason=f"Data refers to the {fund.plan_type} plan, user asked for {constraints.plan_type}"))

    if any(r.status == "FAIL" for r in results):
        status = "FAIL"
    elif any(r.status == "UNKNOWN" for r in results):
        status = "UNKNOWN"
    else:
        status = "PASS"
    return ConstraintEvaluation(status=status, results=results)
