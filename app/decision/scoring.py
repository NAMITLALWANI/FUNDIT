"""
Deterministic multi-attribute utility scoring over the bounded candidate set.

Components (weights come from ``Settings``; rationale in docs/decision_engine.md):

    risk_adjusted_return   sharpe_3y, min-max normalised within the candidate set
    expense_efficiency     1 - minmax(expense_ratio)
    downside_protection    mean of (1 - minmax(volatility_3y)) and (1 - minmax(|max_drawdown_3y|))
    preference_fit         weighted mean of the user's stated soft preferences (None if none stated)
    evidence_quality       coverage x relevance of *verified* document chunks; 0 contribution when
                           no evidence exists (absence is a negative signal, never a default score)
    aum_context            minmax(aum_crores); deliberately low weight - size is contextual only

Normalisation is always relative to the candidate set produced by SQL filtering + retrieval,
so the engine never scores the whole catalogue.

Missing structured data: the component is ``None`` and its weight is dropped from the
denominator (renormalisation). This neither rewards nor punishes a fund for a gap; the gap is
reported through ``data_completeness`` and lowers confidence instead. Missing *evidence* is
treated differently (contribution 0, weight kept) because the system must not recommend a fund
it has no verified textual evidence for.

UNKNOWN hard constraints subtract ``unknown_constraint_penalty`` each, so verified PASS funds
outrank funds whose compliance could not be established at equal utility.
"""

from typing import Dict, List, Optional, Sequence

from app.core.config import Settings, get_settings
from app.decision.constraints import evaluate_constraints
from app.decision.models import (
    STATUS_RANK,
    CandidateInput,
    CandidateScore,
    ComponentScore,
    StructuredFact,
)
from app.ingestion.models import FundMetricsRecord, FundRecord
from app.query.models import DecisionQuery

COMPONENT_NAMES = (
    "risk_adjusted_return",
    "expense_efficiency",
    "downside_protection",
    "preference_fit",
    "evidence_quality",
    "aum_context",
)


def weights_from_settings(settings: Settings) -> Dict[str, float]:
    return {
        "risk_adjusted_return": settings.weight_risk_adjusted_return,
        "expense_efficiency": settings.weight_expense_efficiency,
        "downside_protection": settings.weight_downside_protection,
        "preference_fit": settings.weight_preference_fit,
        "evidence_quality": settings.weight_evidence_quality,
        "aum_context": settings.weight_aum_context,
    }


def _minmax(values: Dict[str, Optional[float]], higher_is_better: bool = True) -> Dict[str, Optional[float]]:
    """Min-max normalise within the candidate set; None stays None; constant sets map to 0.5."""
    present = [v for v in values.values() if v is not None]
    if not present:
        return {k: None for k in values}
    lo, hi = min(present), max(present)
    out: Dict[str, Optional[float]] = {}
    for key, value in values.items():
        if value is None:
            out[key] = None
        elif hi == lo:
            out[key] = 0.5
        else:
            norm = (value - lo) / (hi - lo)
            out[key] = norm if higher_is_better else 1.0 - norm
    return out


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


# Objective → preference-key boosts injected into preference_fit when query.objective is set.
# Only the two objectives with sufficient data evidence are supported.
# Values are additive preference weights (not absolute; see _effective_pref_weights).
_OBJECTIVE_PREFERENCE_BOOSTS: Dict[str, Dict[str, float]] = {
    "capital_preservation": {"low_volatility": 0.6, "consistency": 0.4},
    "capital_appreciation": {"high_return": 0.8},
}


def _effective_pref_weights(prefs_weights: Dict[str, float], objective: Optional[str]) -> Dict[str, float]:
    """Return effective preference weights after applying objective boosts.

    Boosts are additive: if the user already stated a preference the boost increases its weight;
    if they did not, the preference is introduced at the boost value.  Weights are NOT re-normalised
    here; normalisation happens inside the weighted-mean computation which divides by the sum of
    available weights.  This prevents double-counting: the objective never creates a weight beyond
    its own boost share.
    """
    if not objective or objective not in _OBJECTIVE_PREFERENCE_BOOSTS:
        return prefs_weights
    effective = dict(prefs_weights)  # copy, never mutate the original
    for key, boost in _OBJECTIVE_PREFERENCE_BOOSTS[objective].items():
        # Cap at 1.0 to keep the weight in a sensible range.
        effective[key] = min(1.0, effective.get(key, 0.0) + boost)
    return effective


def structured_facts(fund: FundRecord, metrics: Optional[FundMetricsRecord]) -> List[StructuredFact]:
    """Collect the structured values a reader may need to audit the score.

    ``investment_objective`` is included so the generator and auditors can trace the fund's
    stated objective.  The value is ``None`` when the database field is unset; the system
    never substitutes a default for a missing objective.
    """
    facts: List[StructuredFact] = []
    for field, unit in (
        ("expense_ratio", "%"), ("aum_crores", "INR crore"), ("min_sip_amount", "INR"),
        ("min_lump_sum", "INR"), ("lock_in_years", "years"), ("risk_level", None),
        ("category", None), ("sub_category", None), ("exit_load", None),
        ("investment_objective", None),
    ):
        facts.append(StructuredFact(field=field, value=getattr(fund, field), unit=unit, as_of=fund.expense_ratio_date or fund.data_as_of, source="funds"))
    if metrics:
        for field in ("cagr_1y", "cagr_3y", "cagr_5y", "volatility_3y", "sharpe_3y", "max_drawdown_3y", "rolling_3y_positive_pct"):
            facts.append(StructuredFact(field=field, value=getattr(metrics, field), unit="decimal", as_of=metrics.as_of_date, source="fund_metrics"))
    return facts


class ScoringEngine:
    """Scores a bounded candidate set with configurable, documented weights."""

    def __init__(self, settings: Optional[Settings] = None, weights: Optional[Dict[str, float]] = None) -> None:
        self.settings = settings or get_settings()
        self.weights = dict(weights) if weights else weights_from_settings(self.settings)
        missing = set(COMPONENT_NAMES) - set(self.weights)
        if missing:
            raise ValueError(f"Decision weights missing components: {sorted(missing)}")
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Decision weights must sum to 1.0, got {total:.4f}")
        self.unknown_penalty = self.settings.unknown_constraint_penalty
        self.evidence_chunks_per_fund = self.settings.evidence_chunks_per_fund

    # ------------------------------------------------------------ components

    def _component_scores(self, candidates: Sequence[CandidateInput], query: DecisionQuery) -> Dict[str, Dict[str, Optional[float]]]:
        ids = [c.fund.fund_id for c in candidates]
        m = {c.fund.fund_id: c.metrics for c in candidates}
        f = {c.fund.fund_id: c.fund for c in candidates}

        sharpe = _minmax({i: (m[i].sharpe_3y if m[i] else None) for i in ids})
        expense = _minmax({i: f[i].expense_ratio for i in ids}, higher_is_better=False)
        vol = _minmax({i: (m[i].volatility_3y if m[i] else None) for i in ids}, higher_is_better=False)
        dd = _minmax({i: (abs(m[i].max_drawdown_3y) if m[i] and m[i].max_drawdown_3y is not None else None) for i in ids}, higher_is_better=False)
        downside = {i: _mean([vol[i], dd[i]]) for i in ids}
        aum = _minmax({i: f[i].aum_crores for i in ids})

        horizon = query.horizon_years or 0
        return_field = "cagr_5y" if horizon >= 5 else "cagr_3y"
        returns = _minmax({i: (getattr(m[i], return_field) if m[i] else None) for i in ids})
        consistency = _minmax({i: (m[i].rolling_3y_positive_pct if m[i] else None) for i in ids})

        preference_fit: Dict[str, Optional[float]] = {}
        prefs = query.preferences
        # Build effective weights: user preferences + objective-driven boosts (additive, no mutation).
        effective_weights = _effective_pref_weights(prefs.weights, query.objective)
        _has_any_preference = bool(effective_weights) or bool(prefs.preferred_amcs)
        for i in ids:
            parts: List[Optional[float]] = []
            weights: List[float] = []
            for key, w in effective_weights.items():
                if key == "low_cost":
                    parts.append(expense[i])
                elif key == "low_volatility":
                    parts.append(downside[i])
                elif key == "high_return":
                    parts.append(returns[i])
                elif key == "consistency":
                    parts.append(consistency[i])
                elif key == "fund_size":
                    parts.append(aum[i])
                elif key == "preferred_amc":
                    amc = f[i].amc.lower()
                    parts.append(1.0 if any(p.lower() in amc for p in prefs.preferred_amcs) else 0.0)
                else:
                    continue
                weights.append(w)
            available = [(p, w) for p, w in zip(parts, weights) if p is not None]
            if not _has_any_preference:
                preference_fit[i] = None
            elif not available:
                preference_fit[i] = None
            else:
                preference_fit[i] = sum(p * w for p, w in available) / sum(w for _, w in available)

        evidence: Dict[str, Optional[float]] = {}
        for c in candidates:
            if not c.evidence:
                evidence[c.fund.fund_id] = None
            else:
                top = sorted(c.evidence, key=lambda e: e.relative_relevance, reverse=True)[: self.evidence_chunks_per_fund]
                coverage = min(1.0, len(top) / self.evidence_chunks_per_fund)
                relevance = sum(e.relative_relevance for e in top) / len(top)
                evidence[c.fund.fund_id] = round(coverage * relevance, 4)

        return {
            "risk_adjusted_return": sharpe,
            "expense_efficiency": expense,
            "downside_protection": downside,
            "preference_fit": preference_fit,
            "evidence_quality": evidence,
            "aum_context": aum,
        }

    # ------------------------------------------------------------------ score

    def score(self, candidates: Sequence[CandidateInput], query: DecisionQuery) -> List[CandidateScore]:
        """Score and rank the given candidates (PASS before UNKNOWN before FAIL, then by score)."""
        if not candidates:
            return []
        components = self._component_scores(candidates, query)
        effective_weights_for_basis = _effective_pref_weights(query.preferences.weights, query.objective)
        _pref_active = bool(effective_weights_for_basis) or bool(query.preferences.preferred_amcs)
        basis = {
            "risk_adjusted_return": "fund_metrics.sharpe_3y",
            "expense_efficiency": "funds.expense_ratio",
            "downside_protection": "fund_metrics.volatility_3y, fund_metrics.max_drawdown_3y",
            "preference_fit": (
                "objective: " + (query.objective or "") + "; user preferences: " + ", ".join(effective_weights_for_basis)
                if _pref_active else "no preferences or objective stated"
            ),
            "evidence_quality": "verified document chunks (coverage x relevance)",
            "aum_context": "funds.aum_crores",
        }

        # preference_fit is only expected when user stated preferences OR objective is active
        expected_components = len(COMPONENT_NAMES) - (0 if _pref_active else 1)

        scored: List[CandidateScore] = []
        for c in candidates:
            fid = c.fund.fund_id
            evaluation = evaluate_constraints(c.fund, c.metrics, query.constraints)
            comps: List[ComponentScore] = []
            weighted_sum = 0.0
            weight_denominator = 0.0
            available = 0
            for name in COMPONENT_NAMES:
                w = self.weights[name]
                s = components[name][fid]
                if name == "evidence_quality":
                    # absence of verified evidence contributes 0 but keeps its weight (penalty, not default)
                    contribution = w * (s or 0.0)
                    weight_denominator += w
                    if s is not None:
                        available += 1
                elif s is None:
                    contribution = 0.0
                else:
                    contribution = w * s
                    weight_denominator += w
                    available += 1
                weighted_sum += contribution
                comps.append(ComponentScore(name=name, weight=w, score=None if s is None else round(s, 4), contribution=round(contribution, 4), basis=basis[name]))

            utility = weighted_sum / weight_denominator if weight_denominator > 0 else 0.0
            penalty = self.unknown_penalty * len(evaluation.unknowns)
            final = max(0.0, utility - penalty)

            scored.append(
                CandidateScore(
                    fund_id=fid,
                    fund_name=c.fund.fund_name,
                    amc=c.fund.amc,
                    category=c.fund.category,
                    sub_category=c.fund.sub_category,
                    constraint_status=evaluation.status,
                    constraint_results=evaluation.results,
                    violations=[r.reason for r in evaluation.failures],
                    unknowns=[r.reason for r in evaluation.unknowns],
                    components=comps,
                    unknown_penalty=round(penalty, 4),
                    final_score=round(final, 4),
                    data_completeness=round(min(1.0, available / expected_components), 3),
                    evidence_available=bool(c.evidence),
                    evidence=sorted(c.evidence, key=lambda e: e.relative_relevance, reverse=True)[: self.evidence_chunks_per_fund],
                    structured_facts=structured_facts(c.fund, c.metrics),
                )
            )

        scored.sort(key=lambda s: (STATUS_RANK[s.constraint_status], -s.final_score, s.fund_name))
        for position, s in enumerate(scored, start=1):
            s.rank = position
        return scored
