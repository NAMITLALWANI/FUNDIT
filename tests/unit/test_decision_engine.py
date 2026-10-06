"""Deterministic decision engine: scoring, eligibility, preferences, trade-offs, abstention, sensitivity."""

from datetime import date

import pytest

from app.decision.engine import DecisionEngine
from app.decision.models import CandidateInput
from app.decision.scoring import ScoringEngine, weights_from_settings
from app.decision.sensitivity import SensitivityAnalyzer, perturbed_weight_sets
from app.decision.tradeoffs import compare_pair
from app.query.models import Ambiguity, QueryConstraints, UserPreferences
from tests.conftest import AS_OF, make_evidence, make_fund, make_metrics, make_query


@pytest.fixture
def engine(settings) -> DecisionEngine:
    return DecisionEngine(settings=settings)


class TestScoringBasics:
    def test_ranking_is_deterministic_and_explained(self, engine, three_candidates):
        result = engine.decide(make_query(), three_candidates, today=AS_OF)
        again = engine.decide(make_query(), three_candidates, today=AS_OF)
        assert [c.fund_id for c in result.ranked] == ["SYN-A", "SYN-B", "SYN-C"]
        assert [c.final_score for c in result.ranked] == [c.final_score for c in again.ranked]
        assert result.winner is not None and result.winner.fund_id == "SYN-A"
        assert all(c.basis for c in result.winner.components)
        assert abs(sum(result.weights_used.values()) - 1.0) < 1e-9

    def test_weights_come_from_settings_and_must_sum_to_one(self, settings):
        assert weights_from_settings(settings)["aum_context"] == settings.weight_aum_context
        with pytest.raises(ValueError):
            ScoringEngine(settings=settings, weights={"risk_adjusted_return": 1.0})

    def test_aum_is_low_weight(self, settings):
        w = weights_from_settings(settings)
        assert w["aum_context"] <= 0.10
        assert w["aum_context"] < min(w["risk_adjusted_return"], w["expense_efficiency"], w["downside_protection"])

    def test_engine_only_scores_supplied_candidates(self, engine, three_candidates):
        result = engine.decide(make_query(), three_candidates[:2], today=AS_OF)
        assert result.candidates_considered == 2
        assert {c.fund_id for c in result.ranked} == {"SYN-A", "SYN-B"}

    def test_structured_facts_are_separate_from_evidence(self, engine, three_candidates):
        winner = engine.decide(make_query(), three_candidates, today=AS_OF).winner
        assert winner is not None
        assert {f.source for f in winner.structured_facts} == {"funds", "fund_metrics"}
        assert all(e.chunk_id.startswith("SYN-A") for e in winner.evidence)


class TestEvidenceHandling:
    def test_no_default_evidence_score_when_evidence_missing(self, engine, three_candidates):
        three_candidates[0] = three_candidates[0].model_copy(update={"evidence": []})
        result = engine.decide(make_query(), three_candidates, today=AS_OF)
        a = next(c for c in result.ranked if c.fund_id == "SYN-A")
        comp = a.component("evidence_quality")
        assert a.evidence_available is False
        assert comp is not None and comp.score is None and comp.contribution == 0.0

    def test_fund_without_evidence_cannot_be_recommended(self, engine, three_candidates):
        three_candidates[0] = three_candidates[0].model_copy(update={"evidence": []})
        result = engine.decide(make_query(), three_candidates, today=AS_OF)
        # SYN-A may still rank first on structured metrics, but the gate refuses to recommend it
        if result.ranked[0].fund_id == "SYN-A":
            assert result.abstained and "evidence" in (result.abstention_reason or "").lower()
        else:
            assert result.winner is not None and result.winner.evidence_available


class TestHardConstraints:
    def test_fail_candidates_are_excluded_with_reasons(self, engine, three_candidates):
        q = make_query(constraints=QueryConstraints(max_expense_ratio=1.0))
        result = engine.decide(q, three_candidates, today=AS_OF)
        assert [c.fund_id for c in result.excluded] == ["SYN-C"]
        assert result.rejection_reasons["SYN-C"][0].startswith("Hard constraint failure")
        assert result.winner is not None and result.winner.fund_id == "SYN-A"

    def test_no_eligible_candidates_abstains(self, engine, three_candidates):
        q = make_query(constraints=QueryConstraints(max_expense_ratio=0.1))
        result = engine.decide(q, three_candidates, today=AS_OF)
        assert result.abstained and result.winner is None
        assert len(result.excluded) == 3
        assert "No fund" in (result.abstention_reason or "")
        assert result.confidence.level == "LOW"

    def test_unknown_never_wins_over_pass(self, engine, three_candidates):
        # Make the best fund's risk level unknown; the verified PASS fund must win.
        a = three_candidates[0].fund.model_copy(update={"risk_level": None})
        three_candidates[0] = three_candidates[0].model_copy(update={"fund": a})
        q = make_query(constraints=QueryConstraints(max_risk_level="Very High"))
        result = engine.decide(q, three_candidates, today=AS_OF)
        statuses = {c.fund_id: c.constraint_status for c in result.ranked}
        assert statuses["SYN-A"] == "UNKNOWN"
        assert result.winner is not None and result.winner.fund_id == "SYN-B"
        assert result.ranked[0].fund_id == "SYN-B"  # PASS ranks above UNKNOWN regardless of score

    def test_all_unknown_abstains(self, engine, three_candidates):
        q = make_query(constraints=QueryConstraints(exit_load_free=True))
        result = engine.decide(q, three_candidates, today=AS_OF)
        assert result.abstained and result.winner is None
        assert all(c.constraint_status == "UNKNOWN" for c in result.ranked)
        assert "missing" in (result.abstention_reason or "")

    def test_unknown_penalty_applied(self, settings, three_candidates):
        a = three_candidates[0].fund.model_copy(update={"expense_ratio": None})
        three_candidates[0] = three_candidates[0].model_copy(update={"fund": a})
        scored = ScoringEngine(settings=settings).score(three_candidates, make_query(constraints=QueryConstraints(max_expense_ratio=1.0)))
        a_score = next(c for c in scored if c.fund_id == "SYN-A")
        assert a_score.unknown_penalty == pytest.approx(settings.unknown_constraint_penalty)
        assert a_score.data_completeness < 1.0


class TestPreferences:
    def test_low_cost_preference_shifts_ranking_without_eliminating(self, settings):
        cheap = CandidateInput(fund=make_fund("SYN-CHEAP", "Cheap", expense_ratio=0.2), metrics=make_metrics("SYN-CHEAP", sharpe_3y=0.5), evidence=make_evidence("SYN-CHEAP"))
        strong = CandidateInput(fund=make_fund("SYN-STRONG", "Strong", expense_ratio=1.2), metrics=make_metrics("SYN-STRONG", sharpe_3y=1.0), evidence=make_evidence("SYN-STRONG"))
        eng = ScoringEngine(settings=settings)
        neutral = eng.score([cheap, strong], make_query())
        prefer_cheap = eng.score([cheap, strong], make_query(preferences=UserPreferences(weights={"low_cost": 1.0})))
        cheap_neutral = next(c.final_score for c in neutral if c.fund_id == "SYN-CHEAP")
        cheap_pref = next(c.final_score for c in prefer_cheap if c.fund_id == "SYN-CHEAP")
        assert cheap_pref > cheap_neutral
        assert len(prefer_cheap) == 2 and all(c.constraint_status == "PASS" for c in prefer_cheap)

    def test_preferred_amc_is_soft(self, settings, three_candidates):
        c = three_candidates[2].fund.model_copy(update={"amc": "Preferred House AMC"})
        three_candidates[2] = three_candidates[2].model_copy(update={"fund": c})
        q = make_query(preferences=UserPreferences(weights={"preferred_amc": 1.0}, preferred_amcs=["preferred house"]))
        scored = ScoringEngine(settings=settings).score(three_candidates, q)
        assert all(s.constraint_status == "PASS" for s in scored)
        gamma = next(s for s in scored if s.fund_id == "SYN-C")
        assert gamma.component("preference_fit").score == 1.0

    def test_no_preferences_means_component_absent_not_zero(self, settings, three_candidates):
        scored = ScoringEngine(settings=settings).score(three_candidates, make_query())
        assert all(s.component("preference_fit").score is None for s in scored)
        assert all(s.data_completeness == 1.0 for s in scored)


class TestTradeoffs:
    def test_tradeoffs_cite_structured_fields(self, engine, three_candidates):
        result = engine.decide(make_query(), three_candidates, today=AS_OF)
        assert result.tradeoffs and result.tradeoffs[0].fund_a == "Synthetic Alpha Fund"
        assert "expense_ratio" in result.tradeoffs[0].fact_basis
        assert any("expense ratio 0.50% vs 0.90%" in s for s in result.tradeoffs[0].advantages_a)

    def test_missing_values_reported_as_gaps(self, settings):
        a = CandidateInput(fund=make_fund("SYN-A", "A"), metrics=make_metrics("SYN-A", cagr_5y=None), evidence=make_evidence("SYN-A"))
        b = CandidateInput(fund=make_fund("SYN-B", "B", expense_ratio=1.0), metrics=make_metrics("SYN-B"), evidence=make_evidence("SYN-B"))
        scored = ScoringEngine(settings=settings).score([a, b], make_query())
        t = compare_pair(scored[0], scored[1])
        assert "Not comparable" in t.summary and "5-year CAGR" in t.summary


class TestConfidenceAndAbstention:
    def test_clear_winner_has_higher_confidence_than_close_race(self, engine):
        base_q = make_query()
        clear = [
            CandidateInput(fund=make_fund("SYN-A", "A", expense_ratio=0.3), metrics=make_metrics("SYN-A", sharpe_3y=1.2, volatility_3y=0.08, max_drawdown_3y=-0.08), evidence=make_evidence("SYN-A", 3)),
            CandidateInput(fund=make_fund("SYN-B", "B", expense_ratio=1.5), metrics=make_metrics("SYN-B", sharpe_3y=0.2, volatility_3y=0.25, max_drawdown_3y=-0.35), evidence=make_evidence("SYN-B", 3)),
        ]
        close = [
            CandidateInput(fund=make_fund("SYN-A", "A", expense_ratio=0.80), metrics=make_metrics("SYN-A", sharpe_3y=0.80, volatility_3y=0.120, max_drawdown_3y=-0.150), evidence=make_evidence("SYN-A", 3)),
            CandidateInput(fund=make_fund("SYN-B", "B", expense_ratio=0.81), metrics=make_metrics("SYN-B", sharpe_3y=0.81, volatility_3y=0.121, max_drawdown_3y=-0.151), evidence=make_evidence("SYN-B", 3)),
        ]
        r_clear = engine.decide(base_q, clear, today=AS_OF)
        r_close = engine.decide(base_q, close, today=AS_OF)
        assert r_clear.confidence.composite > r_close.confidence.composite
        assert r_clear.confidence.signals.score_margin > r_close.confidence.signals.score_margin

    def test_close_and_unstable_candidates_abstain(self, engine):
        # A wins expense (0.20) + downside (0.15) = 0.35; B wins risk-adjusted return (0.35).
        # Evidence and AUM are identical, so the utilities tie exactly and any perturbation of
        # those three weights flips the winner -> close margin + instability -> abstain.
        close = [
            CandidateInput(fund=make_fund("SYN-A", "A", expense_ratio=0.5), metrics=make_metrics("SYN-A", sharpe_3y=0.6, volatility_3y=0.10, max_drawdown_3y=-0.12), evidence=make_evidence("SYN-A", 3)),
            CandidateInput(fund=make_fund("SYN-B", "B", expense_ratio=1.0), metrics=make_metrics("SYN-B", sharpe_3y=1.0, volatility_3y=0.14, max_drawdown_3y=-0.18), evidence=make_evidence("SYN-B", 3)),
        ]
        result = engine.decide(make_query(), close, today=AS_OF)
        assert result.sensitivity is not None
        assert result.confidence.signals.raw_margin < engine.settings.min_score_margin
        assert result.sensitivity.winner_flip_rate > 0
        assert result.abstained and "too close" in (result.abstention_reason or "")
        assert result.winner is None

    def test_close_but_stable_lowers_confidence_without_abstaining(self, engine):
        # Nearly identical funds where A is marginally better on every component: the margin is
        # small but no weight perturbation can flip the order, so the system answers at reduced confidence.
        close = [
            CandidateInput(fund=make_fund("SYN-A", "A", expense_ratio=0.80), metrics=make_metrics("SYN-A", sharpe_3y=0.80, volatility_3y=0.120, max_drawdown_3y=-0.150), evidence=make_evidence("SYN-A", 3, relevance=0.8)),
            CandidateInput(fund=make_fund("SYN-B", "B", expense_ratio=0.81), metrics=make_metrics("SYN-B", sharpe_3y=0.79, volatility_3y=0.121, max_drawdown_3y=-0.151), evidence=make_evidence("SYN-B", 3, relevance=0.79)),
            CandidateInput(fund=make_fund("SYN-C", "C", expense_ratio=1.50), metrics=make_metrics("SYN-C", sharpe_3y=0.10, volatility_3y=0.250, max_drawdown_3y=-0.350), evidence=make_evidence("SYN-C", 3, relevance=0.3)),
        ]
        result = engine.decide(make_query(), close, today=AS_OF)
        assert result.sensitivity is not None and result.sensitivity.winner_flip_rate == 0.0
        assert result.confidence.signals.raw_margin < engine.settings.min_score_margin
        assert not result.abstained and result.winner is not None and result.winner.fund_id == "SYN-A"
        assert any("margin" in r.lower() for r in result.confidence.reasons)

    def test_blocking_ambiguity_abstains_before_recommending(self, engine, three_candidates):
        q = make_query(ambiguities=[Ambiguity(field="guaranteed_return", message="Mutual funds cannot guarantee returns.", severity="blocking")])
        result = engine.decide(q, three_candidates, today=AS_OF)
        assert result.abstained and result.winner is None
        assert "guarantee" in (result.abstention_reason or "")

    def test_stale_data_lowers_confidence(self, engine, three_candidates):
        fresh = engine.decide(make_query(), three_candidates, today=AS_OF)
        stale = engine.decide(make_query(), three_candidates, today=date(2029, 1, 1))
        assert stale.confidence.signals.data_freshness < fresh.confidence.signals.data_freshness
        assert stale.confidence.composite < fresh.confidence.composite

    def test_confidence_is_not_a_return_probability(self, engine, three_candidates):
        result = engine.decide(make_query(), three_candidates, today=AS_OF)
        assert "not a probability of future investment returns" in result.confidence.description
        assert result.confidence.level in {"LOW", "MEDIUM", "HIGH"}

    def test_abstain_on_low_confidence_policy(self, settings, three_candidates):
        strict = settings.model_copy(update={"abstain_on_low_confidence": True, "confidence_medium_threshold": 0.98, "confidence_high_threshold": 0.99})
        result = DecisionEngine(settings=strict).decide(make_query(), three_candidates, today=AS_OF)
        assert result.confidence.level == "LOW" and result.abstained


class TestSensitivity:
    def test_perturbation_grid_is_deterministic_and_normalised(self, settings):
        base = weights_from_settings(settings)
        sets = perturbed_weight_sets(base, 0.2)
        assert len(sets) == 12
        assert all(abs(sum(s.values()) - 1.0) < 1e-9 for s in sets)
        assert sets == perturbed_weight_sets(base, 0.2)

    def test_dominant_winner_is_stable(self, settings, three_candidates):
        scored = ScoringEngine(settings=settings).score(three_candidates, make_query())
        report = SensitivityAnalyzer(settings=settings).analyze(three_candidates, make_query(), scored, weights_from_settings(settings))
        assert report.trials == 12
        assert report.winner_flip_rate == 0.0
        assert report.baseline_winner == "SYN-A"
        assert report.top3_jaccard_mean == 1.0

    def test_competing_profiles_produce_flips(self, settings):
        cands = [
            CandidateInput(fund=make_fund("SYN-A", "A", expense_ratio=0.3), metrics=make_metrics("SYN-A", sharpe_3y=0.4, volatility_3y=0.15, max_drawdown_3y=-0.20), evidence=make_evidence("SYN-A", 3)),
            CandidateInput(fund=make_fund("SYN-B", "B", expense_ratio=1.2), metrics=make_metrics("SYN-B", sharpe_3y=1.0, volatility_3y=0.10, max_drawdown_3y=-0.10), evidence=make_evidence("SYN-B", 3)),
        ]
        scored = ScoringEngine(settings=settings).score(cands, make_query())
        report = SensitivityAnalyzer(settings=settings, perturbation=0.9).analyze(cands, make_query(), scored, weights_from_settings(settings))
        assert 0.0 <= report.winner_flip_rate <= 1.0
        assert report.trials == 12
        if report.winner_flip_rate > 0:
            assert report.alternative_winners
