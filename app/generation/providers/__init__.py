"""
LLM Provider package: factory function and public exports.
"""

from app.core.config import Settings
from app.generation.providers.base import LLMProvider
from app.generation.providers.mock import DeterministicMockProvider


def get_llm_provider(settings: Settings) -> LLMProvider:
    """Instantiate appropriate LLM provider based on application configuration."""
    if settings.llm_provider == "gemini" and settings.gemini_api_key:
        from app.generation.providers.gemini import GeminiProvider
        return GeminiProvider(settings=settings)
    # Default: offline mock (safe for tests, CI, and environments without credentials)
    return DeterministicMockProvider()


__all__ = [
    "LLMProvider",
    "DeterministicMockProvider",
    "GeminiProvider",
    "get_llm_provider",
]
