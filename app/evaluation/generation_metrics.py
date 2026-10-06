"""
Generation grounding and citation coverage evaluation metrics.
"""

import re
from typing import Any, Dict, List
from app.generation.models import Citation


def calculate_citation_coverage(response_text: str, citations: List[Citation]) -> float:
    """Calculate percentage of provided citations referenced in generated response text."""
    if not citations:
        return 1.0

    referenced_count = 0
    for c in citations:
        # Check if citation marker (e.g. "[1]") appears in text
        escaped_marker = re.escape(c.citation_id)
        if re.search(escaped_marker, response_text):
            referenced_count += 1

    return referenced_count / float(len(citations))


def calculate_constraint_faithfulness(
    extracted_constraints: Dict[str, Any], expected_constraints: Dict[str, Any]
) -> float:
    """Calculate accuracy of extracted constraints relative to expected ground truth."""
    if not expected_constraints:
        return 1.0

    matching_keys = 0
    for k, expected_val in expected_constraints.items():
        actual_val = extracted_constraints.get(k)
        if actual_val is not None:
            if isinstance(expected_val, float) and isinstance(actual_val, (int, float)):
                if abs(expected_val - actual_val) < 1.0:
                    matching_keys += 1
            elif isinstance(expected_val, list) and isinstance(actual_val, list):
                if set(expected_val).issubset(set(actual_val)):
                    matching_keys += 1
            elif actual_val == expected_val:
                matching_keys += 1

    return matching_keys / float(len(expected_constraints))
