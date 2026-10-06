"""
Multi-signal system confidence and the abstention gate.

Confidence is a *decision/evidence* confidence signal: how well the system could verify
constraints, how much verified evidence backs the winner, how clearly it separates from the
runner-up, how complete and fresh the structured data is, and how stable the ranking is under
weight perturbation. It is explicitly not a probability of future returns.

Signals (each in [0,1], equal-weighted mean -> composite; thresholds from Settings):
    constraint_completeness  share of the winner's user-set constraints that are PASS (1 if none)
    evidence_coverage        verified chunks for the winner / evidence_chunks_per_fund
    evidence_quality         winner's evidence_quality component (0 if none)
    score_margin             min(1, raw_margin / (3 * min_score_margin))
    data_completeness        winner's share of scoring components with data
    data_freshness           1 if data age <= data_freshness_days, linear decay to 0 at 2x
    ranking_stability        1 - winner_flip_rate from the sensitivity analysis

Equal weighting is a deliberate, documented simplification: there is no empirical basis in
this project for ranking one signal above another, and equal weights keep the composite
interpretable. Blocking ambiguities and FAIL-only candidate sets bypass the composite entirely
(hard abstention).
"""

from datetime import date
from typing import List, Optional, Sequence, Tuple

from app.core.config import Settings, get_settings
from app.decision.models import (
    CandidateScore,
    ConfidenceAssessment,
    ConfidenceSignals,
    SensitivityReport,
)
from app.query.models import DecisionQuery


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


class ConfidenceCalculator:
    """Computes confidence signals, a level, and whether the system should abstain."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def freshness_signal(self, as_of: Optional[date], today: Optional[date] = None) -> float:
        if as_of is None:
            return 0.0
        age_days = ((today or date.today()) - as_of).days
        limit = self.settings.data_freshness_days
        if age_days <= limit:
            return 1.0
        return _clamp(1.0 - (age_days - limit) / max(1, limit))

    def assess(
        self,
        query: DecisionQuery,
        winner: Optional[CandidateScore],
        ranked: Sequence[CandidateScore],
        sensitivity: Optional[SensitivityReport],
        data_as_of: Optional[date],
        today: Optional[date] = None,
    ) -> Tuple[ConfidenceAssessment, bool, Optional[str]]:
        """Return (assessment, abstain?, abstention_reason)."""
        reasons: List[str] = []
        blocking = query.blocking_ambiguities

        if winner is None:
            signals = ConfidenceSignals(
                constraint_completeness=0.0, evidence_coverage=0.0, evidence_quality=0.0, score_margin=0.0,
                data_completeness=0.0, data_freshness=self.freshness_signal(data_as_of, today),
                ranking_stability=0.0, raw_margin=0.0, unknown_constraint_count=0,
                ambiguity_count=len(query.ambiguities),
            )
            if blocking:
                reason = "; ".join(a.message for a in blocking)
            elif any(c.constraint_status == "UNKNOWN" for c in ranked):
                reason = "No candidate could be verified against all hard constraints; the required data is missing for every remaining fund."
            else:
                reason = "No fund in the database satisfies all stated hard constraints."
            return ConfidenceAssessment(level="LOW", composite=0.0, signals=signals, reasons=[reason]), True, reason

        active = [r for r in winner.constraint_results]
        constraint_completeness = (
            len([r for r in active if r.status == "PASS"]) / len(active) if active else 1.0
        )
        evidence_coverage = _clamp(len(winner.evidence) / max(1, self.settings.evidence_chunks_per_fund))
        ev_component = winner.component("evidence_quality")
        evidence_quality = ev_component.score if ev_component and ev_component.score is not None else 0.0

        eligible = [c for c in ranked if c.constraint_status != "FAIL"]
        runner_up = next((c for c in eligible if c.fund_id != winner.fund_id), None)
        raw_margin = winner.final_score - runner_up.final_score if runner_up else winner.final_score
        if runner_up is None:
            score_margin = 1.0
        else:
            score_margin = _clamp(raw_margin / (3.0 * self.settings.min_score_margin))

        freshness = self.freshness_signal(data_as_of, today)
        stability = 1.0 - (sensitivity.winner_flip_rate if sensitivity and sensitivity.trials else 0.0)

        signals = ConfidenceSignals(
            constraint_completeness=round(constraint_completeness, 3),
            evidence_coverage=round(evidence_coverage, 3),
            evidence_quality=round(evidence_quality, 3),
            score_margin=round(score_margin, 3),
            data_completeness=round(winner.data_completeness, 3),
            data_freshness=round(freshness, 3),
            ranking_stability=round(stability, 3),
            raw_margin=round(raw_margin, 4),
            unknown_constraint_count=len(winner.unknowns),
            ambiguity_count=len(query.ambiguities),
        )
        composite = round(
            sum([
                signals.constraint_completeness, signals.evidence_coverage, signals.evidence_quality,
                signals.score_margin, signals.data_completeness, signals.data_freshness, signals.ranking_stability,
            ]) / 7.0,
            3,
        )

        if winner.unknowns:
            reasons.append(f"{len(winner.unknowns)} hard constraint(s) could not be verified for the winner (UNKNOWN).")
        if not winner.evidence_available:
            reasons.append("No verified document evidence was retrieved for the winner.")
        if runner_up is not None and raw_margin < self.settings.min_score_margin:
            reasons.append(f"Winner margin over {runner_up.fund_name} is only {raw_margin:.3f} (threshold {self.settings.min_score_margin}).")
        if sensitivity and sensitivity.trials and sensitivity.winner_flip_rate > 0:
            reasons.append(f"Winner changed in {sensitivity.winner_flip_rate:.0%} of weight-perturbation trials.")
        if freshness < 1.0:
            reasons.append("Structured data is older than the configured freshness window.")
        if winner.data_completeness < 1.0:
            reasons.append(f"Only {winner.data_completeness:.0%} of scoring components had data for the winner.")
        if query.ambiguities:
            reasons.append(f"{len(query.ambiguities)} query interpretation note(s) were recorded.")

        if composite >= self.settings.confidence_high_threshold:
            level = "HIGH"
        elif composite >= self.settings.confidence_medium_threshold:
            level = "MEDIUM"
        else:
            level = "LOW"

        # ----- abstention gate (hard rules first, then policy) -----
        abstain = False
        abstention_reason: Optional[str] = None
        if blocking:
            abstain, abstention_reason = True, "; ".join(a.message for a in blocking)
        elif winner.constraint_status != "PASS":
            abstain, abstention_reason = True, (
                "The best-ranked fund could not be verified against all hard constraints: " + "; ".join(winner.unknowns)
            )
        elif len(winner.evidence) < self.settings.min_evidence_chunks_for_generation:
            abstain, abstention_reason = True, "Insufficient verified document evidence for the leading candidate."
        elif runner_up is not None and raw_margin < self.settings.min_score_margin and stability < 1.0:
            # Close margin alone lowers confidence; close margin plus an observed winner flip under
            # small weight perturbations means the ranking is not defensible -> abstain.
            abstain, abstention_reason = True, (
                f"Candidates are too close to separate reliably: {winner.fund_name} and {runner_up.fund_name} differ by "
                f"{raw_margin:.3f} and the winner flips in {1 - stability:.0%} of sensitivity trials."
            )
        elif self.settings.abstain_on_low_confidence and level == "LOW":
            abstain, abstention_reason = True, "System confidence is LOW and the abstain-on-low-confidence policy is enabled."

        if not reasons:
            reasons.append("All stated hard constraints verified, evidence retrieved, and ranking stable under perturbation.")

        return ConfidenceAssessment(level=level, composite=composite, signals=signals, reasons=reasons), abstain, abstention_reason
