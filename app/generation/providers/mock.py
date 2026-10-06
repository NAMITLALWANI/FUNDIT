"""
DeterministicMockProvider: offline LLM provider for tests and environments without
a live Gemini API key.

The mock produces structurally valid JSON matching GeneratedDecision. It does not
fabricate financial data; it synthesises a response from the decision context already
embedded in the prompt (winner name, score, constraint status). This is ONLY for
offline/test execution and must never be claimed to demonstrate real LLM quality.
"""

import json
import re
from typing import Optional

from app.generation.providers.base import LLMProvider


class DeterministicMockProvider(LLMProvider):
    """Produces valid GeneratedDecision JSON from the user prompt without calling any API.

    IMPORTANT: This provider is for testing and offline execution ONLY. Its output
    quality does not represent the GeminiProvider.
    """

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Extract key facts from the prompt and return a deterministic JSON response."""
        winner_name = self._extract_winner(prompt)
        confidence_level = self._extract_confidence(prompt)
        fund_names = self._extract_funds(prompt)
        has_evidence = "Evidence [" in prompt

        reasons = [
            f"The deterministic scoring engine selected {winner_name} as the highest-ranked eligible fund.",
            "All stated hard constraints were evaluated using tri-state logic (PASS / FAIL / UNKNOWN).",
        ]
        if has_evidence:
            reasons.append("Verified document evidence was retrieved and assessed for evidence quality scoring.")

        tradeoff_notes: list = []
        if len(fund_names) > 1:
            alternative = next((f for f in fund_names if f != winner_name), None)
            if alternative:
                tradeoff_notes.append(
                    f"{winner_name} achieved a higher utility score than {alternative} "
                    f"based on structured metrics (Sharpe ratio, expense ratio, downside protection)."
                )

        cited_chunk_ids: list = []
        for m in re.finditer(r"Chunk ID:\s*([^\n]+)", prompt):
            cid = m.group(1).strip()
            if cid:
                cited_chunk_ids.append(cid)

        result = {
            "recommendation_text": (
                f"Based on the deterministic multi-attribute scoring, {winner_name} is recommended "
                f"as the top-ranked eligible fund (system confidence: {confidence_level}). "
                "This analysis is based on historical data and publicly available scheme documents. "
                "Past performance does not guarantee future results."
            ),
            "summary": (
                f"{winner_name} was selected by the deterministic decision engine with {confidence_level} "
                f"confidence after evaluating {len(fund_names)} candidate(s)."
            ),
            "reasons": reasons,
            "tradeoff_notes": tradeoff_notes,
            "cited_chunk_ids": cited_chunk_ids[:5],
        }
        return json.dumps(result)

    @staticmethod
    def _extract_winner(prompt: str) -> str:
        m = re.search(r"Fund:\s*([^\n(]+)", prompt)
        if m:
            return m.group(1).strip()
        m = re.search(r"WINNER:\s*\n.*?Fund:\s*([^\n]+)", prompt, re.DOTALL)
        if m:
            return m.group(1).strip()
        return "the recommended fund"

    @staticmethod
    def _extract_confidence(prompt: str) -> str:
        m = re.search(r"Confidence Level:\s*(HIGH|MEDIUM|LOW)", prompt)
        return m.group(1) if m else "MEDIUM"

    @staticmethod
    def _extract_funds(prompt: str) -> list:
        names: list = []
        for m in re.finditer(r"Fund:\s*([^\n(]+)", prompt):
            name = m.group(1).strip()
            if name and name not in names:
                names.append(name)
        return names
