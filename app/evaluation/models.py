"""
Evaluation data models for IR benchmarks, generation metrics, and reporting.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EvaluationQuery(BaseModel):
    """Ground-truth evaluation benchmark item."""

    query_id: str
    query: str
    expected_intent: str = "recommendation"
    expected_constraints: Dict[str, Any] = Field(default_factory=dict)
    expected_fund_ids: List[str] = Field(default_factory=list)
    relevant_chunk_ids: List[str] = Field(default_factory=list)
    description: Optional[str] = None


class RetrievalMetrics(BaseModel):
    """Standard Information Retrieval evaluation metrics."""

    recall_at_k: Dict[int, float] = Field(default_factory=dict)
    precision_at_k: Dict[int, float] = Field(default_factory=dict)
    mrr: float = Field(default=0.0, description="Mean Reciprocal Rank")
    ndcg_at_k: Dict[int, float] = Field(default_factory=dict, description="Normalized Discounted Cumulative Gain")


class GenerationMetrics(BaseModel):
    """Faithfulness, constraint precision, and citation coverage metrics."""

    citation_coverage: float = Field(default=0.0, description="Percentage of claims backed by valid citations")
    constraint_faithfulness: float = Field(
        default=0.0, description="Rate at which hard constraints are accurately preserved"
    )
    answer_relevance: float = Field(default=0.0, description="Relevance score of generated answer to query")


class EvaluationReport(BaseModel):
    """Aggregated evaluation summary report."""

    total_queries: int
    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics
    query_details: List[Dict[str, Any]] = Field(default_factory=list)
