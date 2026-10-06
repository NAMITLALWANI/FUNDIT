"""
DecisionEngine: deterministic coordinator over a bounded candidate set.

Input: candidates produced by SQL filtering + evidence retrieval (never the whole catalogue),
each carrying structured rows and *verified* evidence references. Output: ranked candidates,
winner, trade-offs, confidence signals, sensitivity report and the abstention decision.

The engine always runs before any LLM call; the LLM later explains this result and cannot
change the winner, the scores or the constraint outcomes.
"""

from datetime import date
from typing import Dict, List, Optional, Sequence

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.decision.confidence import ConfidenceCalculator
from app.decision.models import CandidateInput, CandidateScore, DecisionResult
from app.decision.scoring import ScoringEngine
from app.decision.sensitivity import SensitivityAnalyzer
from app.decision.tradeoffs import TradeoffAnalyzer
from app.query.models import DecisionQuery

logger = get_logger(__name__)


class DecisionEngine:
    """Scores, ranks, explains and gates a candidate set deterministically."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        scoring: Optional[ScoringEngine] = None,
        tradeoffs: Optional[TradeoffAnalyzer] = None,
        confidence: Optional[ConfidenceCalculator] = None,
        sensitivity: Optional[SensitivityAnalyzer] = None,
        run_sensitivity: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.scoring = scoring or ScoringEngine(settings=self.settings)
        self.tradeoffs = tradeoffs or TradeoffAnalyzer()
        self.confidence = confidence or ConfidenceCalculator(settings=self.settings)
        self.sensitivity = sensitivity or SensitivityAnalyzer(settings=self.settings)
        self.run_sensitivity = run_sensitivity

    def decide(
        self,
        query: DecisionQuery,
        candidates: Sequence[CandidateInput],
        data_as_of: Optional[date] = None,
        today: Optional[date] = None,
    ) -> DecisionResult:
        """Run the full deterministic decision over the supplied candidates."""
        scored = self.scoring.score(candidates, query)
        ranked = [c for c in scored if c.constraint_status != "FAIL"]
        excluded = [c for c in scored if c.constraint_status == "FAIL"]

        # Winner must be a verified PASS candidate; UNKNOWN candidates are listed but never win.
        winner: Optional[CandidateScore] = next((c for c in ranked if c.constraint_status == "PASS"), None)

        sensitivity = None
        if self.run_sensitivity and winner is not None and len(ranked) >= 2:
            sensitivity = self.sensitivity.analyze(candidates, query, scored, self.scoring.weights)

        if data_as_of is None:
            dates = [c.fund.data_as_of for c in candidates]
            data_as_of = max(dates) if dates else None

        assessment, abstain, reason = self.confidence.assess(
            query=query, winner=winner, ranked=ranked, sensitivity=sensitivity, data_as_of=data_as_of, today=today
        )

        tradeoffs = self.tradeoffs.analyze(ranked, winner) if winner is not None else []

        selection_reasons: List[str] = []
        rejection_reasons: Dict[str, List[str]] = {}
        if winner is not None:
            selection_reasons.append(f"Highest utility score ({winner.final_score:.3f}) among {len(ranked)} eligible candidate(s).")
            if winner.constraint_results:
                selection_reasons.append(f"All {len(winner.constraint_results)} stated hard constraint(s) verified as PASS from structured data.")
            top_components = sorted((c for c in winner.components if c.score is not None), key=lambda c: c.contribution, reverse=True)[:3]
            for comp in top_components:
                selection_reasons.append(f"{comp.name.replace('_', ' ').capitalize()}: {comp.score:.2f} relative to the candidate set (basis: {comp.basis}).")
            if winner.evidence_available:
                selection_reasons.append(f"{len(winner.evidence)} verified document chunk(s) retrieved for this fund.")
            for alt in ranked:
                if alt.fund_id == winner.fund_id:
                    continue
                reasons = [f"Lower utility score {alt.final_score:.3f} vs {winner.final_score:.3f}."]
                if alt.constraint_status == "UNKNOWN":
                    reasons.append("Could not verify: " + "; ".join(alt.unknowns))
                if not alt.evidence_available:
                    reasons.append("No verified document evidence retrieved.")
                rejection_reasons[alt.fund_id] = reasons
        for ex in excluded:
            rejection_reasons[ex.fund_id] = ["Hard constraint failure: " + v for v in ex.violations]

        result = DecisionResult(
            candidates_considered=len(candidates),
            winner=None if abstain else winner,
            ranked=ranked,
            excluded=excluded,
            tradeoffs=tradeoffs,
            confidence=assessment,
            sensitivity=sensitivity,
            abstained=abstain,
            abstention_reason=reason,
            selection_reasons=selection_reasons if not abstain else [],
            rejection_reasons=rejection_reasons,
            weights_used=dict(self.scoring.weights),
        )
        logger.info(
            "Decision: candidates=%d eligible=%d excluded=%d winner=%s confidence=%s(%.2f) abstained=%s",
            len(candidates), len(ranked), len(excluded), winner.fund_id if winner else None,
            assessment.level, assessment.composite, abstain,
        )
        return result
