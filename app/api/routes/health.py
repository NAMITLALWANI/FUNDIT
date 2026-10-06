"""
Health and readiness check endpoints.
"""

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from app.api.dependencies import get_service_container
from app.api.schemas.responses import HealthResponse, ReadyResponse
from app.core.dependencies import ServiceContainer

router = APIRouter(tags=["Health & Monitoring"])


@router.get("/health", response_model=HealthResponse, summary="Liveness Probe")
def health_check() -> HealthResponse:
    """Returns basic liveness status."""
    return HealthResponse(status="ok", version="0.1.0", app_name="AI-Decision-Engine-V2")


@router.get("/ready", response_model=ReadyResponse, summary="Readiness Probe")
def readiness_check(
    container: ServiceContainer = Depends(get_service_container),
) -> ReadyResponse:
    """Checks whether models, vector store, and indexes are loaded and operational."""
    vector_ok = False
    if container.vector_store is not None:
        vector_ok = container.vector_store.health_check()

    is_ready = container.is_ready() and vector_ok

    chunks_count = 0
    if container.sparse_retriever is not None:
        chunks_count = len(container.sparse_retriever.chunks)

    resp = ReadyResponse(
        ready=is_ready,
        vector_store_healthy=vector_ok,
        models_loaded=container.embedding_service is not None,
        indexed_chunks_count=chunks_count,
    )

    if not is_ready:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=resp.model_dump())

    return resp
