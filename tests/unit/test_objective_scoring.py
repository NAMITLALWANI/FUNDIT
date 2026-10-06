"""
Regression tests for investment_objective → preference_fit wiring.

Confirms:
- capital_preservation objective boosts low-volatility / consistency candidates.
- capital_appreciation objective boosts high-return candidates.
- NULL / unknown objective: preference_fit is None (no fabricated value).
- investment_objective appears in structured_facts (never omitted).
- No double-counting when user also states matching preference.
- _effective_pref_weights is a pure function (original dict not mutated).
"""

from datetime import date
from typing import Optional

import pytest

from app.decision.scoring import (
    ScoringEngine,
    _OBJECTIVE_PREFERENCE_BOOSTS,
    _effective_pref_weights,
    structured_facts,
    weights_from_settings,
)
from app.query.models import DecisionQuery, QueryConstraints, UserPreferences
from tests.conftest import AS_OF, make_evidence, make_fund, make_metrics, make_query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidates(settings):
    """Two synthetic candidates:
    SYN-STAB:  low volatility, high consistency  → good for capital_preservation
    SYN-GROW:  high return, lower consistency     → good for capital_appreciation
    """
    from app.decision.models import CandidateInput

    stab = make_fund("SYN-STAB", "Stable Fund", expense_ratio=0.6, aum_crores=15_000)
    grow = make_fund("SYN-GROW", "Growth Fund", expense_ratio=0.9, aum_crores=8_000)
    return [
        CandidateInput(
            fund=stab,
            metrics=make_metrics(
                "SYN-STAB",
                cagr_3y=0.10,
                sharpe_3y=0.65,
                volatility_3y=0.06,
                max_drawdown_3y=-0.08,
                rolling=0.95,
            ),
            evidence=make_evidence("SYN-STAB", 3),
        ),
        CandidateInput(
            fund=grow,
            metrics=make_metrics(
                "SYN-GROW",
                cagr_3y=0.22,
                sharpe_3y=0.75,
                volatility_3y=0.20,
                max_drawdown_3y=-0.25,
                rolling=0.70,
            ),
            evidence=make_evidence("SYN-GROW", 2),
        ),
    ]


def _make_query_with_objective(objective: Optional[str]) -> DecisionQuery:
    return DecisionQuery(
        raw_query="synthetic query",
        rewritten_query="synthetic query",
        constraints=QueryConstraints(),
        preferences=UserPreferences(),
        objective=objective,
        ambiguities=[],
    )


# ---------------------------------------------------------------------------
# Unit tests for _effective_pref_weights
# ---------------------------------------------------------------------------


class TestEffectivePrefWeights:
    def test_none_objective_returns_original_unchanged(self):
        original = {"low_cost": 0.5}
        result = _effective_pref_weights(original, None)
        assert result is original

    def test_unknown_objective_returns_original_unchanged(self):
        original = {"low_cost": 0.5}
        result = _effective_pref_weights(original, "retirement")
        assert result is original

    def test_capital_preservation_injects_volatility_and_consistency(self):
        result = _effective_pref_weights({}, "capital_preservation")
        assert "low_volatility" in result
        assert "consistency" in result
        assert result["low_volatility"] > 0
        assert result["consistency"] > 0

    def test_capital_appreciation_injects_high_return(self):
        result = _effective_pref_weights({}, "capital_appreciation")
        assert "high_return" in result
        assert result["high_return"] > 0

    def test_does_not_mutate_original_dict(self):
        original = {"low_cost": 0.5}
        _effective_pref_weights(original, "capital_preservation")
        assert set(original.keys()) == {"low_cost"}

    def test_additive_boost_does_not_exceed_1(self):
        result = _effective_pref_weights({"low_volatility": 0.9}, "capital_preservation")
        assert result["low_volatility"] <= 1.0

    def test_no_double_counting_beyond_cap(self):
        result = _effective_pref_weights({"low_volatility": 0.5}, "capital_preservation")
        assert result["low_volatility"] <= 1.0

    def test_all_supported_objectives_in_boosts_table(self):
        assert "capital_preservation" in _OBJECTIVE_PREFERENCE_BOOSTS
        assert "capital_appreciation" in _OBJECTIVE_PREFERENCE_BOOSTS


# ---------------------------------------------------------------------------
# Scoring-level tests
# ---------------------------------------------------------------------------


class TestObjectiveScoringIntegration:
    @pytest.fixture
    def candidates(self, settings):
        return _make_candidates(settings)

    def test_capital_preservation_ranks_stable_fund_higher_on_pref_fit(self, settings, candidates):
        engine = ScoringEngine(settings=settings)
        q = _make_query_with_objective("capital_preservation")
        scored = engine.score(candidates, q)

        stab = next(c for c in scored if c.fund_id == "SYN-STAB")
        grow = next(c for c in scored if c.fund_id == "SYN-GROW")
        pf_stab = stab.component("preference_fit")
        pf_grow = grow.component("preference_fit")

        assert pf_stab is not None and pf_stab.score is not None
        assert pf_grow is not None and pf_grow.score is not None
        assert pf_stab.score > pf_grow.score, (
            f"SYN-STAB pref_fit ({pf_stab.score}) should beat SYN-GROW ({pf_grow.score}) "
            "under capital_preservation"
        )

    def test_capital_appreciation_ranks_growth_fund_higher_on_pref_fit(self, settings, candidates):
        engine = ScoringEngine(settings=settings)
        q = _make_query_with_objective("capital_appreciation")
        scored = engine.score(candidates, q)

        stab = next(c for c in scored if c.fund_id == "SYN-STAB")
        grow = next(c for c in scored if c.fund_id == "SYN-GROW")
        pf_grow = grow.component("preference_fit")
        pf_stab = stab.component("preference_fit")

        assert pf_grow is not None and pf_grow.score is not None
        assert pf_stab is not None
        assert pf_grow.score > pf_stab.score, (
            f"SYN-GROW pref_fit ({pf_grow.score}) should beat SYN-STAB ({pf_stab.score}) "
            "under capital_appreciation"
        )

    def test_null_objective_preference_fit_is_none(self, settings, candidates):
        engine = ScoringEngine(settings=settings)
        q = _make_query_with_objective(None)
        scored = engine.score(candidates, q)
        for c in scored:
            pf = c.component("preference_fit")
            assert pf is not None
            assert pf.score is None, (
                f"preference_fit score must be None for {c.fund_id} with no objective/preferences"
            )

    def test_unknown_objective_does_not_fabricate_preference_fit(self, settings, candidates):
        engine = ScoringEngine(settings=settings)
        q = _make_query_with_objective("retirement")
        scored = engine.score(candidates, q)
        for c in scored:
            pf = c.component("preference_fit")
            assert pf is not None
            assert pf.score is None, (
                f"preference_fit score must be None for {c.fund_id} when objective='retirement' unsupported"
            )

    def test_objective_basis_string_is_auditable(self, settings, candidates):
        engine = ScoringEngine(settings=settings)
        q = _make_query_with_objective("capital_preservation")
        scored = engine.score(candidates, q)
        for c in scored:
            pf = c.component("preference_fit")
            assert pf is not None
            assert "capital_preservation" in pf.basis.lower(), (
                f"basis must mention the objective, got: {pf.basis}"
            )


# ---------------------------------------------------------------------------
# structured_facts tests
# ---------------------------------------------------------------------------


class TestStructuredFacts:
    def test_investment_objective_in_facts_when_set(self):
        fund = make_fund("SYN-T1", "Test Fund")
        fund = fund.model_copy(update={"investment_objective": "To generate long-term capital appreciation"})
        facts = structured_facts(fund, None)
        obj_fact = next((f for f in facts if f.field == "investment_objective"), None)
        assert obj_fact is not None
        assert obj_fact.value == "To generate long-term capital appreciation"
        assert obj_fact.source == "funds"

    def test_investment_objective_in_facts_when_null(self):
        fund = make_fund("SYN-T2", "Test Fund 2")
        facts = structured_facts(fund, None)
        obj_fact = next((f for f in facts if f.field == "investment_objective"), None)
        assert obj_fact is not None, "investment_objective fact must always be present"
        assert obj_fact.value is None, "value must be None, not a fabricated default"

    def test_investment_objective_source_is_funds_table(self):
        fund = make_fund("SYN-T3", "Test Fund 3")
        fund = fund.model_copy(update={"investment_objective": "Income"})
        facts = structured_facts(fund, None)
        obj_fact = next(f for f in facts if f.field == "investment_objective")
        assert obj_fact.source == "funds"

    def test_structured_facts_count_includes_objective(self):
        fund = make_fund("SYN-T4", "Test Fund 4")
        facts_no_metrics = structured_facts(fund, None)
        fund_fields = [f for f in facts_no_metrics if f.source == "funds"]
        assert len(fund_fields) == 10, f"Expected 10 fund-sourced facts, got {len(fund_fields)}"
