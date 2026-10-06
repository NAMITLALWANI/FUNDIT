"""
API routes module.
"""

from app.api.routes.decisions import router as decisions_router
from app.api.routes.health import router as health_router

__all__ = ["health_router", "decisions_router"]
