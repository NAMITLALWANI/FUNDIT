"""Tri-state hard-constraint evaluation: PASS / FAIL / UNKNOWN."""

from app.decision.constraints import evaluate_constraints
from app.query.models import QueryConstraints
from tests.conftest import make_fund, make_metrics


def _status(results, field):
    return next(r.status for r in results if r.field == field)


class TestMissingDataIsUnknownNeverPass:
    def test_missing_risk_level(self):
        fund = make_fund("SYN-X", "Synthetic X", risk_level=None)
        ev = evaluate_constraints(fund, make_metrics("SYN-X"), QueryConstraints(max_risk_level="Moderate"))
        assert ev.status == "UNKNOWN"
        assert _status(ev.results, "max_risk_level") == "UNKNOWN"
        assert ev.unknowns and not ev.failures

    def test_missing_expense_ratio(self):
        fund = make_fund("SYN-X", "Synthetic X", expense_ratio=None)
        ev = evaluate_constraints(fund, None, QueryConstraints(max_expense_ratio=1.0))
        assert ev.status == "UNKNOWN"

    def test_missing_exit_load_terms(self):
        fund = make_fund("SYN-X", "Synthetic X", exit_load=None)
        ev = evaluate_constraints(fund, None, QueryConstraints(exit_load_free=True))
        assert ev.status == "UNKNOWN"
        assert "not available" in ev.unknowns[0].reason

    def test_missing_metrics_for_min_return(self):
        fund = make_fund("SYN-X", "Synthetic X")
        ev = evaluate_constraints(fund, None, QueryConstraints(min_return_3y=0.10))
        assert _status(ev.results, "min_return_3y") == "UNKNOWN"

    def test_missing_min_sip(self):
        fund = make_fund("SYN-X", "Synthetic X", min_sip=None)
        ev = evaluate_constraints(fund, None, QueryConstraints(sip_amount=5000))
        assert ev.status == "UNKNOWN"


class TestFail:
    def test_sip_too_high(self):
        fund = make_fund("SYN-X", "Synthetic X", min_sip=1000)
        ev = evaluate_constraints(fund, None, QueryConstraints(sip_amount=500))
        assert ev.status == "FAIL"
        assert "exceeds" in ev.failures[0].reason

    def test_risk_above_ceiling(self):
        fund = make_fund("SYN-X", "Synthetic X", risk_level="Very High")
        ev = evaluate_constraints(fund, None, QueryConstraints(max_risk_level="Moderate"))
        assert ev.status == "FAIL"

    def test_lock_in_exceeds_horizon(self):
        fund = make_fund("SYN-X", "Synthetic ELSS", lock_in=3.0)
        ev = evaluate_constraints(fund, None, QueryConstraints(max_lock_in_years=2.0))
        assert ev.status == "FAIL"

    def test_fail_dominates_unknown(self):
        fund = make_fund("SYN-X", "Synthetic X", risk_level=None, min_sip=5000)
        ev = evaluate_constraints(fund, None, QueryConstraints(max_risk_level="Moderate", sip_amount=1000))
        assert ev.status == "FAIL"
        assert len(ev.failures) == 1 and len(ev.unknowns) == 1

    def test_required_amc(self):
        fund = make_fund("SYN-X", "Synthetic X", amc="Synthetic AMC")
        ev = evaluate_constraints(fund, None, QueryConstraints(required_amcs=["hdfc"]))
        assert ev.status == "FAIL"

    def test_min_return_below(self):
        fund = make_fund("SYN-X", "Synthetic X")
        ev = evaluate_constraints(fund, make_metrics("SYN-X", cagr_3y=0.08), QueryConstraints(min_return_3y=0.10))
        assert ev.status == "FAIL"


class TestPass:
    def test_all_pass(self):
        fund = make_fund("SYN-X", "Synthetic X", risk_level="Moderate", expense_ratio=0.4, min_sip=100, lock_in=0.0)
        ev = evaluate_constraints(
            fund, make_metrics("SYN-X", cagr_3y=0.12),
            QueryConstraints(max_risk_level="Moderately High", max_expense_ratio=0.5, sip_amount=1000, max_lock_in_years=5, min_return_3y=0.10),
        )
        assert ev.status == "PASS"
        assert len(ev.passes) == 5

    def test_no_constraints_is_pass_with_no_results(self):
        ev = evaluate_constraints(make_fund("SYN-X", "Synthetic X"), None, QueryConstraints())
        assert ev.status == "PASS" and ev.results == []

    def test_plan_type_only_checked_when_specified(self):
        fund = make_fund("SYN-X", "Synthetic X", plan_type="Regular")
        assert evaluate_constraints(fund, None, QueryConstraints()).status == "PASS"
        assert evaluate_constraints(fund, None, QueryConstraints(plan_type="Direct")).status == "FAIL"
