"""
Google Gemini LLM provider via the google-generativeai SDK.

Configuration is sourced entirely from Settings; no hardcoded model names or keys.
The provider calls Gemini with a system instruction + user prompt and returns the
raw text response for JSON parsing and citation validation upstream.

Retry logic: up to settings.llm_max_retries on transient errors (5xx / network issues).
"""

import json
import time
from typing import Optional

from app.core.config import Settings
from app.core.exceptions import LLMProviderError
from app.core.logging import get_logger
from app.generation.providers.base import LLMProvider

logger = get_logger(__name__)


class GeminiProvider(LLMProvider):
    """Google Gemini API provider with configurable model, temperature, and retry."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None  # lazy-loaded on first call

    def _get_model(self):  # type: ignore[return]
        if self._model is not None:
            return self._model
        try:
            import google.generativeai as genai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise LLMProviderError(
                "google-generativeai package is not installed. "
                "Run: pip install google-generativeai"
            ) from exc

        if not self.settings.gemini_api_key:
            raise LLMProviderError(
                "GEMINI_API_KEY is not configured. "
                "Set it in your .env file or as an environment variable."
            )

        genai.configure(api_key=self.settings.gemini_api_key)
        generation_config = genai.GenerationConfig(
            temperature=self.settings.llm_temperature,
            response_mime_type="application/json",
        )
        self._model = genai.GenerativeModel(
            model_name=self.settings.gemini_model,
            generation_config=generation_config,
            system_instruction=None,  # passed per-call via start_chat
        )
        logger.info("Gemini provider initialized: model=%s", self.settings.gemini_model)
        return self._model

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Send prompt to Gemini, retry on transient failures, return raw text."""
        model = self._get_model()
        last_exc: Optional[Exception] = None
        for attempt in range(1 + self.settings.llm_max_retries):
            try:
                if system_prompt:
                    # Use start_chat with a history entry to simulate system instructions
                    # for models that accept system_instruction via GenerativeModel init
                    import google.generativeai as genai  # type: ignore[import-untyped]

                    model_with_system = genai.GenerativeModel(
                        model_name=self.settings.gemini_model,
                        generation_config=genai.GenerationConfig(
                            temperature=self.settings.llm_temperature,
                            response_mime_type="application/json",
                        ),
                        system_instruction=system_prompt,
                    )
                    response = model_with_system.generate_content(prompt)
                else:
                    response = model.generate_content(prompt)

                text = response.text
                logger.info(
                    "Gemini generation successful: attempt=%d tokens_approx=%d",
                    attempt + 1,
                    len(text) // 4,
                )
                return text
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self.settings.llm_max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "Gemini call failed (attempt %d/%d): %s — retrying in %ds",
                        attempt + 1, 1 + self.settings.llm_max_retries, exc, wait,
                    )
                    time.sleep(wait)
        raise LLMProviderError(
            f"Gemini generation failed after {1 + self.settings.llm_max_retries} attempts",
            details={"last_error": str(last_exc)},
        )
