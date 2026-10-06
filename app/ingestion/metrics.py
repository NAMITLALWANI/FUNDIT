"""
Deterministic risk/return metric calculations from NAV history.

Definitions (see docs/data.md for the full data dictionary):

* ``cagr_Ny``  – point-to-point annualised return between the NAV on the as-of date and the
  last available NAV on or before ``as_of - N years``. Reported as a decimal (0.12 = 12%).
  ``cagr_1y`` is the simple 1-year return (identical to CAGR for N=1).
* ``volatility_3y`` – standard deviation of daily log returns over the trailing 3 years,
  annualised with sqrt(252).
* ``sharpe_3y`` – ``(cagr_3y - risk_free_rate) / volatility_3y``.
* ``max_drawdown_3y`` – largest peak-to-trough decline over the trailing 3 years (negative decimal).
* ``rolling_3y_positive_pct`` – fraction of monthly-stepped 3-year rolling windows (within the
  available history) whose annualised return was positive. Requires at least 4 years of history.

Metrics are only produced when the history genuinely covers the window; otherwise ``None`` is
returned so downstream components treat the value as UNKNOWN rather than guessing.
"""

import math
from bisect import bisect_right
from datetime import date, timedelta
from typing import List, Optional, Sequence

from app.ingestion.models import FundMetricsRecord, NavPoint

TRADING_DAYS_PER_YEAR = 252
METHOD_VERSION = "v1"


def _years_ago(as_of: date, years: int) -> date:
    try:
        return as_of.replace(year=as_of.year - years)
    except ValueError:  # 29 Feb edge case
        return as_of.replace(year=as_of.year - years, day=28)


def _nav_on_or_before(points: Sequence[NavPoint], dates: Sequence[date], target: date) -> Optional[NavPoint]:
    idx = bisect_right(dates, target) - 1
    if idx < 0:
        return None
    return points[idx]


def annualised_return(start_nav: float, end_nav: float, years: float) -> Optional[float]:
    """Compound annual growth rate between two NAV observations."""
    if start_nav <= 0 or end_nav <= 0 or years <= 0:
        return None
    return (end_nav / start_nav) ** (1.0 / years) - 1.0


def point_to_point_cagr(points: Sequence[NavPoint], as_of: date, years: int, tolerance_days: int = 10) -> Optional[float]:
    """CAGR over ``years`` ending at ``as_of``; None if history does not reach the start date."""
    if not points:
        return None
    dates = [p.date for p in points]
    end = _nav_on_or_before(points, dates, as_of)
    target_start = _years_ago(as_of, years)
    start = _nav_on_or_before(points, dates, target_start)
    if end is None or start is None:
        return None
    if (target_start - start.date).days > tolerance_days:
        return None
    elapsed_years = (end.date - start.date).days / 365.25
    return annualised_return(start.nav, end.nav, elapsed_years)


def annualised_volatility(points: Sequence[NavPoint], as_of: date, years: int, min_observations: int = 200) -> Optional[float]:
    """Annualised standard deviation of daily log returns over the trailing window."""
    window_start = _years_ago(as_of, years)
    window = [p for p in points if window_start <= p.date <= as_of and p.nav > 0]
    if len(window) < min_observations:
        return None
    if (window[0].date - window_start).days > 10:
        return None
    log_returns: List[float] = []
    for prev, curr in zip(window, window[1:]):
        log_returns.append(math.log(curr.nav / prev.nav))
    if len(log_returns) < 2:
        return None
    mean = sum(log_returns) / len(log_returns)
    variance = sum((r - mean) ** 2 for r in log_returns) / (len(log_returns) - 1)
    return math.sqrt(variance) * math.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(points: Sequence[NavPoint], as_of: date, years: int) -> Optional[float]:
    """Largest peak-to-trough decline over the trailing window (negative decimal)."""
    window_start = _years_ago(as_of, years)
    window = [p for p in points if window_start <= p.date <= as_of and p.nav > 0]
    if len(window) < 2 or (window[0].date - window_start).days > 10:
        return None
    peak = window[0].nav
    worst = 0.0
    for p in window:
        if p.nav > peak:
            peak = p.nav
        drawdown = p.nav / peak - 1.0
        if drawdown < worst:
            worst = drawdown
    return worst


def rolling_positive_share(points: Sequence[NavPoint], as_of: date, window_years: int = 3, min_history_years: float = 4.0) -> Optional[float]:
    """Share of monthly-stepped rolling windows with positive annualised return."""
    if not points:
        return None
    dates = [p.date for p in points]
    history_years = (as_of - points[0].date).days / 365.25
    if history_years < min_history_years:
        return None
    positives = 0
    total = 0
    window_end = as_of
    while True:
        window_start = _years_ago(window_end, window_years)
        if window_start < points[0].date:
            break
        start = _nav_on_or_before(points, dates, window_start)
        end = _nav_on_or_before(points, dates, window_end)
        if start is not None and end is not None and (window_start - start.date).days <= 10:
            total += 1
            if end.nav > start.nav:
                positives += 1
        window_end = window_end - timedelta(days=30)
    if total == 0:
        return None
    return positives / total


def compute_fund_metrics(fund_id: str, points: Sequence[NavPoint], risk_free_rate: float, as_of: Optional[date] = None) -> FundMetricsRecord:
    """Compute the full metric record for a fund from its sorted NAV history."""
    ordered = sorted((p for p in points if p.nav > 0), key=lambda p: p.date)
    if not ordered:
        raise ValueError(f"No NAV observations available for {fund_id}")

    as_of_date = as_of or ordered[-1].date
    cagr_3y = point_to_point_cagr(ordered, as_of_date, 3)
    vol_3y = annualised_volatility(ordered, as_of_date, 3)
    sharpe = None
    if cagr_3y is not None and vol_3y is not None and vol_3y > 0:
        sharpe = (cagr_3y - risk_free_rate) / vol_3y

    return FundMetricsRecord(
        fund_id=fund_id,
        as_of_date=as_of_date,
        history_start=ordered[0].date,
        history_years=round((as_of_date - ordered[0].date).days / 365.25, 2),
        cagr_1y=point_to_point_cagr(ordered, as_of_date, 1),
        cagr_3y=cagr_3y,
        cagr_5y=point_to_point_cagr(ordered, as_of_date, 5),
        volatility_3y=vol_3y,
        sharpe_3y=sharpe,
        max_drawdown_3y=max_drawdown(ordered, as_of_date, 3),
        rolling_3y_positive_pct=rolling_positive_share(ordered, as_of_date),
        risk_free_rate_used=risk_free_rate,
        method_version=METHOD_VERSION,
    )
