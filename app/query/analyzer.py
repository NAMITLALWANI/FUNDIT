"""
Query analyzer: turns a natural-language mutual fund request into a validated ``DecisionQuery``.

The analyzer is deterministic. An optional LLM extraction pass (Step 4) may *propose* values,
but every proposal is validated through the same Pydantic models and merged conservatively here:
deterministic extractions always win, and LLM-only values are recorded as ambiguities until the
user confirms them.
"""

from typing import Optional, Sequence, Tuple

from pydantic import ValidationError

from app.core.exceptions import QueryParsingError
from app.core.logging import get_logger
from app.query.extractor import ConstraintExtractor
from app.query.models import Ambiguity, DecisionQuery
from app.query.rewriter import QueryRewriter

logger = get_logger(__name__)

MIN_QUERY_LENGTH = 3


class QueryAnalyzer:
    """Analyzes natural language queries into typed intent, constraints, preferences and targets."""

    def __init__(
        self,
        known_funds: Optional[Sequence[Tuple[str, str]]] = None,
        extractor: Optional[ConstraintExtractor] = None,
        rewriter: Optional[QueryRewriter] = None,
    ) -> None:
        self.extractor = extractor or ConstraintExtractor(known_funds=known_funds)
        self.rewriter = rewriter or QueryRewriter()

    def analyze(self, query: str) -> DecisionQuery:
        """Parse the raw query. Raises ``QueryParsingError`` on empty or unusable input."""
        raw_query = (query or "").strip()
        if len(raw_query) < MIN_QUERY_LENGTH:
            raise QueryParsingError("Query is empty or too short to analyse")

        extraction = self.extractor.extract(raw_query)
        ambiguities = list(extraction.ambiguities)

        if (
            extraction.intent == "recommendation"
            and not extraction.constraints.active_fields()
            and extraction.preferences.is_empty
            and extraction.objective is None
            and not extraction.comparison_targets
        ):
            ambiguities.append(
                Ambiguity(
                    field="query",
                    message="No investment amount, horizon, risk tolerance, category, cost limit or preference could be identified; the request is too vague to rank funds meaningfully.",
                    severity="blocking",
                )
            )

        rewritten = self.rewriter.rewrite(raw_query, extraction.constraints, extraction.preferences)

        try:
            decision_query = DecisionQuery(
                raw_query=raw_query,
                rewritten_query=rewritten,
                intent=extraction.intent,  # type: ignore[arg-type]
                investment_amount=extraction.investment_amount,
                investment_frequency=extraction.investment_frequency,  # type: ignore[arg-type]
                horizon_years=extraction.horizon_years,
                risk_tolerance=extraction.risk_tolerance,
                objective=extraction.objective,
                constraints=extraction.constraints,
                preferences=extraction.preferences,
                comparison_targets=extraction.comparison_targets,
                ambiguities=ambiguities,
                extraction_method="deterministic",
            )
        except ValidationError as exc:
            raise QueryParsingError(f"Extracted query failed validation: {exc}") from exc

        logger.info(
            "Analyzed query intent=%s constraints=%s preferences=%s targets=%d ambiguities=%d",
            decision_query.intent,
            decision_query.constraints.model_dump(exclude_none=True),
            list(decision_query.preferences.weights),
            len(decision_query.comparison_targets),
            len(decision_query.ambiguities),
        )
        return decision_query
