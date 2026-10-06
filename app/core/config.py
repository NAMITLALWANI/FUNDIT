"""
Central configuration management using Pydantic Settings.

Every tunable parameter of the mutual fund decision engine (database, retrieval sizes,
decision-engine weights, confidence thresholds, Gemini model) is declared here and can be
overridden through environment variables or a local `.env` file.
"""

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration parameters."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application
    app_name: str = Field(default="AI-Decision-Engine-V2", description="Application Name")
    app_env: Literal["development", "testing", "production"] = Field(
        default="development", description="Environment mode"
    )
    debug: bool = Field(default=True, description="Enable debug diagnostics in API responses")
    log_level: str = Field(default="INFO", description="Logging level")
    log_json: bool = Field(default=False, description="Emit JSON-lines structured logs")
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    preload_models_on_startup: bool = Field(
        default=False,
        description="Load embedding and reranker models during startup instead of on first request",
    )

    # Relational database (SQLite locally, PostgreSQL in deployment)
    database_url: str = Field(
        default="sqlite:///./data/processed/funds.db",
        description="SQLAlchemy database URL (sqlite:///... or postgresql+psycopg://...)",
    )
    database_echo: bool = Field(default=False, description="Echo SQL statements (debugging only)")

    # Qdrant Vector Store
    qdrant_url: Optional[str] = Field(default="http://localhost:6333", description="Qdrant URL")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API Key if enabled")
    qdrant_collection: str = Field(
        default="fund_documents", description="Target Qdrant collection name"
    )
    qdrant_prefer_grpc: bool = Field(default=False, description="Use gRPC transport for Qdrant")
    qdrant_fallback_in_memory: bool = Field(
        default=True,
        description="Fall back to an in-memory Qdrant instance when the server is unreachable",
    )

    # Embedding Model
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5", description="Sentence Transformers embedding model"
    )
    embedding_device: str = Field(default="cpu", description="Embedding inference device (cpu, cuda, mps)")
    embedding_batch_size: int = Field(default=32, description="Batch size for embedding operations")
    embedding_normalize: bool = Field(default=True, description="L2 normalize embedding vectors")
    embedding_cache_path: Optional[str] = Field(
        default="data/processed/embeddings_cache.npz",
        description="Cache of chunk embeddings to avoid re-embedding unchanged text (empty to disable)",
    )

    # Cross-Encoder Reranker Model
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="Cross-encoder model name"
    )
    reranker_device: str = Field(default="cpu", description="Reranker inference device (cpu, cuda, mps)")
    reranker_batch_size: int = Field(default=16, description="Batch size for reranker operations")

    # Chunking & Retrieval
    chunk_size: int = Field(default=500, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=80, description="Chunk overlap in characters")
    retrieval_dense_top_k: int = Field(default=30, description="Dense candidates fetched per query")
    retrieval_sparse_top_k: int = Field(default=30, description="Sparse BM25 candidates fetched per query")
    retrieval_candidate_k: int = Field(default=30, description="Fused candidate pool size passed to the reranker")
    retrieval_rrf_k: int = Field(default=60, description="Reciprocal Rank Fusion smoothing parameter k")
    rerank_top_k: int = Field(default=12, description="Evidence chunks kept after reranking")
    evidence_chunks_per_fund: int = Field(default=3, description="Max evidence chunks attached to a single fund")
    max_structured_candidates: int = Field(
        default=25, description="Upper bound on SQL candidates passed to retrieval and scoring"
    )

    # Decision engine weights (see docs/decision_engine.md for rationale)
    weight_risk_adjusted_return: float = Field(default=0.35, description="Sharpe / CAGR contribution")
    weight_expense_efficiency: float = Field(default=0.20, description="Expense ratio contribution")
    weight_downside_protection: float = Field(default=0.15, description="Volatility / drawdown contribution")
    weight_preference_fit: float = Field(default=0.15, description="Soft preference alignment contribution")
    weight_evidence_quality: float = Field(default=0.10, description="Verified evidence contribution")
    weight_aum_context: float = Field(default=0.05, description="Low-weight AUM contextual signal")
    unknown_constraint_penalty: float = Field(
        default=0.10, description="Score penalty applied per UNKNOWN hard constraint"
    )
    risk_free_rate: float = Field(
        default=0.065,
        description="Annual risk-free rate used for Sharpe ratio (approx. Indian 1Y T-bill yield)",
    )

    # Confidence & abstention thresholds
    confidence_high_threshold: float = Field(default=0.70, description="Composite signal >= this => HIGH")
    confidence_medium_threshold: float = Field(default=0.45, description="Composite signal >= this => MEDIUM")
    min_score_margin: float = Field(default=0.03, description="Winner margin below this is flagged as unstable")
    min_evidence_chunks_for_generation: int = Field(
        default=1, description="Minimum verified evidence chunks required before calling the LLM"
    )
    data_freshness_days: int = Field(default=400, description="Data older than this is flagged stale")
    abstain_on_low_confidence: bool = Field(
        default=False, description="Abstain whenever confidence is LOW (otherwise only on hard failures)"
    )

    # LLM provider configuration
    llm_provider: Literal["gemini", "mock"] = Field(
        default="mock", description="LLM backend: 'gemini' for Google Gemini, 'mock' for offline tests"
    )
    gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini API key")
    gemini_model: str = Field(
        default="gemini-3.8-flash", description="Gemini model identifier (configurable, not pinned)"
    )
    llm_temperature: float = Field(default=0.1, description="LLM sampling temperature")
    llm_timeout_seconds: float = Field(default=30.0, description="Timeout for LLM provider calls")
    llm_max_retries: int = Field(default=2, description="Retries on transient LLM failures")
    llm_use_for_query_analysis: bool = Field(
        default=False, description="Augment deterministic query parsing with an LLM extraction pass"
    )
    gemini_input_cost_per_million: float = Field(
        default=0.0, description="USD per 1M input tokens for cost estimates (0 on free tier)"
    )
    gemini_output_cost_per_million: float = Field(
        default=0.0, description="USD per 1M output tokens for cost estimates (0 on free tier)"
    )

    @model_validator(mode="after")
    def _validate_weights(self) -> "Settings":
        total = (
            self.weight_risk_adjusted_return
            + self.weight_expense_efficiency
            + self.weight_downside_protection
            + self.weight_preference_fit
            + self.weight_evidence_quality
            + self.weight_aum_context
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Decision weights must sum to 1.0, got {total:.4f}")
        if self.confidence_medium_threshold >= self.confidence_high_threshold:
            raise ValueError("confidence_medium_threshold must be below confidence_high_threshold")
        return self

    @property
    def llm_enabled(self) -> bool:
        """True when a real LLM provider is configured with credentials."""
        return self.llm_provider == "gemini" and bool(self.gemini_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton getter for application settings."""
    return Settings()
