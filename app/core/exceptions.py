"""
Domain-specific exception hierarchy for AI Decision Engine.
"""

from typing import Any, Dict, Optional


class AppError(Exception):
    """Base exception for all application-specific errors."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


class ConfigurationError(AppError):
    """Raised when configuration parameters are missing or invalid."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="CONFIGURATION_ERROR", details=details)


class IngestionError(AppError):
    """Raised during document loading, cleaning, or chunking errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="INGESTION_ERROR", details=details)


class VectorStoreError(AppError):
    """Raised when interactions with the vector database fail."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="VECTOR_STORE_ERROR", details=details)


class RetrievalError(AppError):
    """Raised when dense, sparse, or hybrid retrieval operations fail."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="RETRIEVAL_ERROR", details=details)


class RerankingError(AppError):
    """Raised during cross-encoder reranking operations."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="RERANKING_ERROR", details=details)


class DecisionError(AppError):
    """Raised when multi-attribute decision scoring fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="DECISION_ERROR", details=details)


class LLMProviderError(AppError):
    """Raised when upstream LLM service calls fail or return invalid responses."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="LLM_PROVIDER_ERROR", details=details)


class GenerationError(AppError):
    """Raised when grounded generation or citation formatting fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="GENERATION_ERROR", details=details)


class QueryParsingError(AppError):
    """Raised when query normalization or constraint extraction fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="QUERY_PARSING_ERROR", details=details)
