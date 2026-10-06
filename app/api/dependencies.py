"""
FastAPI route dependency injectors.
"""

from fastapi import Depends
from app.core.config import Settings, get_settings
from app.core.dependencies import ServiceContainer, get_container


def get_app_settings() -> Settings:
    """Inject application settings."""
    return get_settings()


def get_service_container() -> ServiceContainer:
    """Inject singleton ServiceContainer instance."""
    return get_container()
