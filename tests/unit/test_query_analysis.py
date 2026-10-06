"""Unit tests for deterministic query understanding (extractor + analyzer)."""

import pytest

from app.core.exceptions import QueryParsingError
from app.query.analyzer import QueryAnalyzer
from app.query.models import QueryConstraints, UserPreferences

KNOWN_FUNDS = [
    ("SYN-PPFC", "Parag Parikh Flexi Cap Fund - Direct Plan - Growth"),
    ("SYN-HFC", "HDFC Flexi Cap Fund - Direct Plan - Growth Option"),
    ("SYN-SBISC", "SBI SMALL CAP FUND - Direct Plan - Growth"),
    ("SYN-NSC", "Nippon India Small Cap Fund - Direct Plan - Growth Option"),
    ("SYN-UTI50", "UTI Nifty 50 Index Fund - Direct Plan - Growth"),
]


@pytest.fixture
def analyzer() -> QueryAnalyzer:
    return QueryAnalyzer(known_funds=KNOWN_FUNDS)


class TestAmountAndFrequency:
    def test_monthly_sip_amount(self, analyzer):
        q = analyzer.analyze("I have ₹5,000 per month, moderate risk tolerance and want to invest for 5 years.")
        assert q.investment_amount == 5000
        assert q.investment_frequency == "monthly_sip"
        assert q.constraints.sip_amount == 5000
        assert q.constraints.lump_sum_amount is None
        assert q.horizon_years == 5
        assert q.constraints.max_lock_in_years == 5

    def test_lakh_lump_sum(self, analyzer):
        q = analyzer.analyze("Invest 2 lakh lump sum in an ELSS direct plan")
        assert q.investment_amount == 200_000
        assert q.investment_frequency == "lump_sum"
        assert q.constraints.lump_sum_amount == 200_000
        assert q.constraints.sub_categories == ["ELSS"]
        assert q.constraints.plan_type == "Direct"
        assert q.objective == "tax_saving"

    def test_k_suffix(self, analyzer):
        q = analyzer.analyze("10k monthly SIP for 3 years, high risk")
        assert q.investment_amount == 10_000
        assert q.investment_frequency == "monthly_sip"

    def test_amount_without_frequency_is_ambiguous_not_guessed(self, analyzer):
        q = analyzer.analyze("I have ₹50,000 and moderate risk appetite")
        assert q.investment_amount == 50_000
        assert q.investment_frequency is None
        assert q.constraints.sip_amount is None and q.constraints.lump_sum_amount is None
        assert any(a.field == "investment_frequency" for a in q.ambiguities)

    def test_horizon_not_mistaken_for_amount(self, analyzer):
        q = analyzer.analyze("moderate risk for 5 years")
        assert q.investment_amount is None
        assert q.horizon_years == 5


class TestRiskAndPlan:
    def test_colloquial_moderate_maps_to_ceiling_with_warning(self, analyzer):
        q = analyzer.analyze("moderate risk, 5 years")
        assert q.constraints.max_risk_level == "Moderately High"
        assert any(a.field == "risk_tolerance" and a.severity == "warning" for a in q.ambiguities)

    def test_exact_sebi_label_has_no_warning(self, analyzer):
        q = analyzer.analyze("very high risk is fine, 10 years")
        assert q.constraints.max_risk_level == "Very High"
        assert not any(a.field == "risk_tolerance" for a in q.ambiguities)

    def test_no_very_high_volatility(self, analyzer):
        q = analyzer.analyze("I want long-term capital appreciation but don't want very high volatility.")
        assert q.constraints.max_risk_level == "High"
        assert "low_volatility" in q.preferences.weights
        assert q.objective == "capital_appreciation"
        assert any(a.field == "horizon_years" for a in q.ambiguities)  # 'long-term' without years

    def test_plan_type_not_assumed(self, analyzer):
        q = analyzer.analyze("5000 per month for 5 years, moderate risk")
        assert q.constraints.plan_type is None

    def test_plan_type_explicit(self, analyzer):
        q = analyzer.analyze("5000 per month in a regular plan for 5 years")
        assert q.constraints.plan_type == "Regular"


class TestExpenseAndPreferences:
    def test_expense_ratio_limit(self, analyzer):
        q = analyzer.analyze("funds with expense ratio under 0.5% for 3 years")
        assert q.constraints.max_expense_ratio == 0.5
        assert "low_cost" in q.preferences.weights

    def test_expense_ratio_number_first(self, analyzer):
        q = analyzer.analyze("Find a fund with 0.3% expense ratio for 3 years")
        assert q.constraints.max_expense_ratio == 0.3

    def test_low_cost_is_preference_not_constraint(self, analyzer):
        q = analyzer.analyze("I want a lower-cost mutual fund and I am willing to accept moderate volatility.")
        assert q.constraints.max_expense_ratio is None
        assert q.preferences.weights.get("low_cost") == 1.0

    def test_preferred_amc_is_soft_only_amc_is_hard(self, analyzer):
        soft = analyzer.analyze("5000 monthly for 5 years, prefer HDFC funds")
        assert soft.constraints.required_amcs is None
        assert soft.preferences.preferred_amcs == ["hdfc"]
        hard = analyzer.analyze("Only HDFC funds please, 10000 monthly for 5 years")
        assert hard.constraints.required_amcs == ["hdfc"]


class TestTargetsAndIntent:
    def test_comparison_resolves_targets_and_drops_name_derived_categories(self, analyzer):
        q = analyzer.analyze("Compare Parag Parikh Flexi Cap and HDFC Flexi Cap for a 7-year horizon.")
        assert q.intent == "comparison"
        assert set(q.resolved_target_ids) == {"SYN-PPFC", "SYN-HFC"}
        assert q.constraints.sub_categories is None
        assert q.horizon_years == 7

    def test_explanation_intent(self, analyzer):
        q = analyzer.analyze("Why did you rank SBI Small Cap above Nippon India Small Cap?")
        assert q.intent == "explanation"
        assert set(q.resolved_target_ids) == {"SYN-SBISC", "SYN-NSC"}

    def test_generic_comparison_without_names(self, analyzer):
        q = analyzer.analyze("Compare two suitable funds for a 7-year investment horizon and explain the trade-offs.")
        assert q.intent == "comparison"
        assert q.comparison_targets == []
        assert not q.blocking_ambiguities

    def test_unknown_named_fund_is_blocking(self, analyzer):
        q = analyzer.analyze("Compare HDFC Flexi Cap and Zerodha Nifty Fund")
        assert q.resolved_target_ids == ["SYN-HFC"]
        assert any(a.field == "comparison_targets" and a.severity == "blocking" for a in q.ambiguities)

    def test_category_word_alone_does_not_resolve_a_fund(self, analyzer):
        q = analyzer.analyze("I want a nifty 50 index fund for 10 years")
        assert q.comparison_targets == []
        assert q.constraints.sub_categories == ["Index Fund"]


class TestFailureModes:
    def test_guaranteed_return_is_blocking(self, analyzer):
        q = analyzer.analyze("Find a fund with 0.01% expense ratio, zero risk, and 30% guaranteed return.")
        assert q.constraints.max_risk_level == "Low"
        assert q.constraints.max_expense_ratio == 0.01
        assert any(a.field == "guaranteed_return" and a.severity == "blocking" for a in q.ambiguities)

    def test_exit_load_constraint(self, analyzer):
        q = analyzer.analyze("Find funds with exit load waiver after 30 days.")
        assert q.constraints.exit_load_free is True

    def test_vague_query_is_blocking(self, analyzer):
        q = analyzer.analyze("best fund")
        assert q.blocking_ambiguities

    def test_empty_query_raises(self, analyzer):
        with pytest.raises(QueryParsingError):
            analyzer.analyze("  ")

    def test_horizon_range_uses_lower_bound_with_warning(self, analyzer):
        q = analyzer.analyze("10000 monthly for 3 to 5 years")
        assert q.horizon_years == 3
        assert any(a.field == "horizon_years" for a in q.ambiguities)


class TestModelValidation:
    def test_invalid_risk_label_rejected(self):
        with pytest.raises(ValueError):
            QueryConstraints(max_risk_level="Extreme")

    def test_invalid_preference_rejected(self):
        with pytest.raises(ValueError):
            UserPreferences(weights={"shiny": 1.0})

    def test_preference_weight_range(self):
        with pytest.raises(ValueError):
            UserPreferences(weights={"low_cost": 1.5})
