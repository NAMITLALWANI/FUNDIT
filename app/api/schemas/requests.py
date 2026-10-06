"""
Mutual fund decision request schema.
"""

from typing import Optional

from pydantic import BaseModel, Field


class DecisionRequest(BaseModel):
    """Payload for POST /api/v1/decisions."""

    query: str = Field(
        ...,
        min_length=3,
        description="Natural language investment query",
        examples=[
            "I have Rs 5,000 per month, moderate risk tolerance and want to invest for 5 years. What funds should I consider?",
            "Compare HDFC Flexi Cap Fund and Mirae Asset Large Cap Fund for a 7-year horizon.",
            "Suggest a low-cost index fund with expense ratio below 0.3% and minimum SIP of Rs 500.",
        ],
    )
    debug: Optional[bool] = Field(
        default=False,
        description="Include retrieval, reranking, and scoring diagnostic payloads in the response",
    )
