"""
Core dependency lifecycle management and container for singleton models/services.
"""

from typing import Any, Callable, Optional

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ServiceContainer:
    """Singleton container managing life-cycle of heavyweight AI models and indexes."""

    def __init__(self) -> None:
        self.settings: Settings = get_settings()
        self.embedding_service: Optional[Any] = None
        self.vector_store: Optional[Any] = None
        self.sparse_retriever: Optional[Any] = None
        self.workflow: Optional[Any] = None
        self.session_factory: Optional[Callable[[], Session]] = None
        self.initialized: bool = False

    def is_ready(self) -> bool:
        """Check if essential services are initialized and ready."""
        return self.initialized and self.workflow is not None


_container: Optional[ServiceContainer] = None


def get_container() -> ServiceContainer:
    """Access the global service container singleton."""
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container
