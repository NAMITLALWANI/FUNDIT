"""
Deterministic retrieval query rewriter for mutual fund evidence search.

The rewritten string is used only for retrieving document chunks. The original query is always
preserved in ``DecisionQuery.raw_query`` and hard constraints are never removed - they are
appended as retrieval terms so BM25 and dense search both see them.
"""

import re
from typing import List

from app.query.models import QueryConstraints, UserPreferences

DOMAIN_EXPANSIONS = {
    r"\bsip\b|\bper month\b|\bmonthly\b": "systematic investment plan minimum SIP instalment",
    r"\blump[\s-]?sum\b": "minimum lump sum investment",
    r"\blow[\s-]?cost\b|\bexpense\b|\bcheap": "total expense ratio direct plan",
    r"\bvolatil|\bstable\b|\bsafe\b|\bdownside\b": "risk-o-meter volatility drawdown risk level",
    r"\bretire|\blong[\s-]?term\b|\bwealth\b|\bcapital appreciation\b": "long-term capital appreciation investment objective equity",
    r"\btax\b|\belss\b|\b80c\b": "ELSS tax saver lock-in 3 years section 80C",
    r"\bemergency\b|\bpark": "liquid fund short-term parking low interest rate risk",
    r"\bindex\b|\bnifty\b|\bpassive\b": "index fund passive tracking error benchmark",
    r"\bexit load\b": "exit load redemption holding period",
    r"\brisk\b": "risk-o-meter principal at risk investment objective",
}


class QueryRewriter:
    """Expands the user query with domain terminology and extracted constraint terms."""

    def rewrite(self, raw_query: str, constraints: QueryConstraints, preferences: UserPreferences) -> str:
        parts: List[str] = [raw_query.strip()]
        ql = raw_query.lower()

        for pattern, expansion in DOMAIN_EXPANSIONS.items():
            if re.search(pattern, ql):
                parts.append(expansion)

        if constraints.sub_categories:
            parts.extend(constraints.sub_categories)
        if constraints.categories:
            parts.extend(f"{c} scheme" for c in constraints.categories)
        if constraints.max_risk_level:
            parts.append(f"{constraints.max_risk_level} risk")
        if constraints.max_expense_ratio is not None:
            parts.append("expense ratio")
        if constraints.exit_load_free:
            parts.append("exit load")
        if constraints.max_lock_in_years is not None:
            parts.append("lock-in period")
        if "low_volatility" in preferences.weights:
            parts.append("volatility drawdown")
        if "consistency" in preferences.weights:
            parts.append("consistent returns rolling")

        seen: set = set()
        tokens: List[str] = []
        for token in " ".join(parts).split():
            key = token.lower()
            if key not in seen:
                seen.add(key)
                tokens.append(token)
        return " ".join(tokens)
