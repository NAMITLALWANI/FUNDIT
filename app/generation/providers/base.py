"""
Abstract LLM provider interface for the mutual fund decision engine.

Providers must be stateless across calls. All configuration (model name, temperature,
timeout) is passed via the Settings object at construction time.
"""

from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    """Abstract interface for LLM backends (Gemini, Mock)."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate completion text given prompt and optional system instructions.

        Returns raw string output which is expected to be valid JSON matching
        the GeneratedDecision schema.
        """
