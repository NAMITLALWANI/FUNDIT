"""
Ranking-stability (sensitivity) analysis under small, deterministic weight perturbations.

For each weight w_i the engine is re-run with w_i scaled by (1 +/- perturbation) and the other
weights renormalised so the vector still sums to one. With six components this yields 12
trials. The winner-flip rate (share of trials with a different top candidate) feeds the
``ranking_stability`` confidence signal; the mean Jaccard similarity of the top-3 set describes
how stable the shortlist is. The grid is deterministic so results are reproducible.
"""

from typing import Dict, List, Optional, Sequence, Set

from app.core.config import Settings, get_settings
from app.decision.models import STATUS_RANK, CandidateInput, CandidateScore, SensitivityReport
from app.decision.scoring import COMPONENT_NAMES, ScoringEngine
from app.query.models import DecisionQuery


def perturbed_weight_sets(base: Dict[str, float], perturbation: float) -> List[Dict[str, float]]:
    """Deterministic +/- perturbation grid, renormalised to sum to 1.0."""
    sets: List[Dict[str, float]] = []
    for name in COMPONENT_NAMES:
        for direction in (1.0, -1.0):
            trial = dict(base)
            trial[name] = max(0.0, base[name] * (1.0 + direction * perturbation))
            total = sum(trial.values())
            if total <= 0:
                continue
            sets.append({k: v / total for k, v in trial.items()})
    return sets


def _eligible_top(ranked: Sequence[CandidateScore], n: int) -> List[str]:
    """Top-n fund IDs among non-FAIL candidates (same selection rule as the engine)."""
    return [c.fund_id for c in ranked if STATUS_RANK[c.constraint_status] < STATUS_RANK["FAIL"]][:n]


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class SensitivityAnalyzer:
    """Measures how often the winner changes when decision weights are nudged."""

    def __init__(self, settings: Optional[Settings] = None, perturbation: float = 0.20) -> None:
        self.settings = settings or get_settings()
        self.perturbation = perturbation

    def analyze(
        self,
        candidates: Sequence[CandidateInput],
        query: DecisionQuery,
        baseline_ranked: Sequence[CandidateScore],
        base_weights: Dict[str, float],
    ) -> SensitivityReport:
        baseline_top3 = _eligible_top(baseline_ranked, 3)
        baseline_winner = baseline_top3[0] if baseline_top3 else None
        trials = perturbed_weight_sets(base_weights, self.perturbation)
        if baseline_winner is None or len(baseline_ranked) < 2 or not trials:
            return SensitivityReport(
                perturbation=self.perturbation, trials=0, winner_flip_rate=0.0, top3_jaccard_mean=1.0,
                baseline_winner=baseline_winner,
            )

        flips = 0
        jaccards: List[float] = []
        alternatives: Dict[str, int] = {}
        for weights in trials:
            ranked = ScoringEngine(settings=self.settings, weights=weights).score(candidates, query)
            top3 = _eligible_top(ranked, 3)
            winner = top3[0] if top3 else None
            if winner != baseline_winner:
                flips += 1
                if winner:
                    alternatives[winner] = alternatives.get(winner, 0) + 1
            jaccards.append(_jaccard(set(baseline_top3), set(top3)))

        return SensitivityReport(
            perturbation=self.perturbation,
            trials=len(trials),
            winner_flip_rate=round(flips / len(trials), 3),
            top3_jaccard_mean=round(sum(jaccards) / len(jaccards), 3),
            alternative_winners=alternatives,
            baseline_winner=baseline_winner,
        )
