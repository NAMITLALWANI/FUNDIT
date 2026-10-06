"""
GenerationService: coordinates grounded LLM synthesis, citation management,
and citation integrity validation.

Execution order:
1. Build numbered Citations from EvidenceRef objects attached to the winner.
2. Format the structured decision + evidence context into the LLM user prompt.
3. Call the configured LLM provider (Gemini or Mock).
4. Parse and validate the JSON output against GeneratedDecision.
5. Validate cited_chunk_ids against the available citation set (programmatic).
6. Strip unverified citations from the final response.
7. Assemble DecisionResponse including the decision trace.
"""

import json
from typing import List, Optional

from app.core.config import Settings, get_settings
from app.core.exceptions import GenerationError
from app.core.logging import get_logger
from app.decision.models import DecisionResult, EvidenceRef
from app.generation.citations import CitationTracker
from app.generation.models import (
    Citation,
    DecisionResponse,
    DecisionTraceStep,
    GeneratedDecision,
)
from app.generation.prompts import DECISION_SYSTEM_PROMPT, format_user_prompt
from app.generation.providers.base import LLMProvider
from app.generation.providers import get_llm_provider
from app.query.models import DecisionQuery

logger = get_logger(__name__)


def _collect_evidence_refs(decision: DecisionResult) -> List[EvidenceRef]:
    """Gather evidence refs: winner first, then runner-ups (de-duplicated by chunk_id)."""
    refs: List[EvidenceRef] = []
    seen: set = set()
    # Winner evidence is the primary pool
    if decision.winner:
        for ref in decision.winner.evidence:
            if ref.chunk_id not in seen:
                refs.append(ref)
                seen.add(ref.chunk_id)
    # Add runner-up evidence for context (up to 3 extra chunks)
    for candidate in decision.ranked:
        if decision.winner and candidate.fund_id == decision.winner.fund_id:
            continue
        for ref in candidate.evidence[:2]:
            if ref.chunk_id not in seen and len(refs) < 15:
                refs.append(ref)
                seen.add(ref.chunk_id)
    return sorted(refs, key=lambda r: r.relative_relevance, reverse=True)


class GenerationService:
    """Coordinates prompt formatting, LLM execution, citation validation, and response assembly."""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        citation_tracker: Optional[CitationTracker] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm_provider = llm_provider or get_llm_provider(self.settings)
        self.citation_tracker = citation_tracker or CitationTracker()

    def generate(
        self,
        query: DecisionQuery,
        decision: DecisionResult,
        extra_trace: Optional[List[DecisionTraceStep]] = None,
    ) -> DecisionResponse:
        """Generate the complete DecisionResponse from a finalized DecisionResult.

        If the decision is an abstention, returns an abstention response without calling the LLM.
        """
        trace: List[DecisionTraceStep] = list(extra_trace or [])

        # -------- Abstention path --------
        if decision.abstained:
            trace.append(DecisionTraceStep(
                stage="generation",
                summary="Abstention: LLM not called.",
                details={"reason": decision.abstention_reason},
            ))
            return DecisionResponse(
                query=query.raw_query,
                parsed_query=query,
                abstained=True,
                abstention_reason=decision.abstention_reason,
                winner=None,
                top_candidates=decision.ranked,
                excluded_candidates=decision.excluded,
                recommendation_text="",
                summary=f"Abstained: {decision.abstention_reason}",
                reasons=[decision.abstention_reason or "Abstention condition met."],
                tradeoffs=[],
                citations=[],
                confidence=decision.confidence,
                sensitivity=decision.sensitivity,
                decision_trace=trace,
                weights_used=decision.weights_used,
            )

        # -------- Evidence → Citations --------
        evidence_refs = _collect_evidence_refs(decision)
        citations = self.citation_tracker.build_citations(evidence_refs)

        trace.append(DecisionTraceStep(
            stage="citation_building",
            summary=f"Built {len(citations)} numbered citations from {len(evidence_refs)} evidence refs.",
            details={"citation_ids": [c.citation_id for c in citations]},
        ))

        # -------- LLM generation --------
        recommendation_text = ""
        summary = ""
        reasons: List[str] = list(decision.selection_reasons)
        tradeoff_notes: List[str] = []
        verified_citations = citations  # default: keep all if LLM fails

        try:
            user_prompt = format_user_prompt(query, decision, citations)
            raw_output = self.llm_provider.generate(
                prompt=user_prompt,
                system_prompt=DECISION_SYSTEM_PROMPT,
            )
            parsed = self._parse_llm_output(raw_output)

            recommendation_text = parsed.recommendation_text
            summary = parsed.summary
            if parsed.reasons:
                reasons = parsed.reasons
            tradeoff_notes = parsed.tradeoff_notes

            # Citation integrity validation
            verified_ids, unverified_ids = self.citation_tracker.validate_cited_ids(
                parsed.cited_chunk_ids, citations
            )
            if unverified_ids:
                logger.warning(
                    "LLM cited %d unverified chunk ID(s) — stripped: %s",
                    len(unverified_ids), unverified_ids,
                )
            verified_citations = self.citation_tracker.filter_citations(citations, verified_ids)

            trace.append(DecisionTraceStep(
                stage="generation",
                summary=f"LLM generation succeeded ({len(verified_citations)} citations verified).",
                details={
                    "cited_count": len(parsed.cited_chunk_ids),
                    "verified": len(verified_ids),
                    "unverified_stripped": len(unverified_ids),
                    "provider": type(self.llm_provider).__name__,
                },
            ))

        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM generation failed (%s); falling back to deterministic synthesis.", exc)
            trace.append(DecisionTraceStep(
                stage="generation",
                summary="LLM generation failed; deterministic fallback used.",
                details={"error": str(exc)},
            ))
            recommendation_text, summary = self._deterministic_fallback(decision)
            reasons = list(decision.selection_reasons)

        return DecisionResponse(
            query=query.raw_query,
            parsed_query=query,
            abstained=False,
            winner=decision.winner,
            top_candidates=decision.ranked,
            excluded_candidates=decision.excluded,
            recommendation_text=recommendation_text,
            summary=summary,
            reasons=reasons,
            tradeoffs=decision.tradeoffs,
            citations=verified_citations,
            confidence=decision.confidence,
            sensitivity=decision.sensitivity,
            decision_trace=trace,
            weights_used=decision.weights_used,
        )

    @staticmethod
    def _parse_llm_output(raw: str) -> GeneratedDecision:
        """Parse and validate LLM JSON output. Raises GenerationError on failure."""
        text = raw.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if "```" in text:
                text = text.rsplit("```", 1)[0]
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise GenerationError(f"LLM output is not valid JSON: {exc}") from exc
        try:
            return GeneratedDecision.model_validate(data)
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"LLM output failed GeneratedDecision validation: {exc}") from exc

    @staticmethod
    def _deterministic_fallback(decision: DecisionResult) -> tuple:
        """Produce a minimal but accurate recommendation without LLM."""
        if not decision.winner:
            return "", "No recommendation available."
        w = decision.winner
        facts = {f.field: f.value for f in w.structured_facts}
        exp = facts.get("expense_ratio")
        sharpe = facts.get("sharpe_3y")
        exp_str = f"{exp:.2f}%" if exp is not None else "N/A"
        sharpe_str = f"{sharpe:.2f}" if sharpe is not None else "N/A"
        text = (
            f"The deterministic scoring engine recommends {w.fund_name} (AMC: {w.amc}) "
            f"with a final utility score of {w.final_score:.4f} "
            f"(confidence: {decision.confidence.level}). "
            f"Expense ratio: {exp_str}. 3-year Sharpe: {sharpe_str}. "
            "This analysis is based on historical structured data. "
            "Past performance does not guarantee future results."
        )
        summary = (
            f"{w.fund_name} ranked first among {len(decision.ranked)} eligible candidate(s) "
            f"with {decision.confidence.level} system confidence."
        )
        return text, summary
