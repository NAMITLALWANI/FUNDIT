"""
FastAPI Response Pydantic schemas.
"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Payload for GET /health."""

    status: str = "ok"
    version: str = "0.1.0"
    app_name: str = "AI-Decision-Engine-V2"


class ReadyResponse(BaseModel):
    """Payload for GET /ready."""

    ready: bool
    vector_store_healthy: bool
    models_loaded: bool
    indexed_chunks_count: int = 0
