"""
Decision recommendation endpoint.
"""

import time

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_service_container
from app.api.schemas.requests import DecisionRequest
from app.core.dependencies import ServiceContainer
from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.generation.models import DecisionResponse

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Decisions"])


@router.post(
    "/decisions",
    response_model=DecisionResponse,
    summary="Compute grounded mutual fund decision with explainable trade-offs and citations",
)
def compute_decision(
    request: DecisionRequest,
    container: ServiceContainer = Depends(get_service_container),
) -> DecisionResponse:
    """
    Execute full Decision-Augmented Pipeline:
    1. Query Analysis & Intent Extraction
    2. SQL Database Retrieval (Filtering candidates by structured logic)
    3. Hybrid Retrieval (Dense Semantic + BM25 Lexical with RRF)
    4. Cross-Encoder Candidate Reranking
    5. Evidence Validation (Detecting conflicts across chunk sources)
    6. Deterministic Multi-Attribute Utility Scoring & Confidence Gating
    7. Grounded LLM Response Synthesis with Chunk Citations
    """
    t0 = time.perf_counter()
    try:
        if container.workflow is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Workflow service is not initialized",
            )

        response = container.workflow.run(request.query, debug=request.debug)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Processed decision query in %.0fms [Winner: %s]",
            elapsed_ms,
            response.winner.fund_name if response.winner else "None",
        )
        return response

    except AppError as e:
        logger.warning("Application error processing decision: %s", e.message, extra=e.details)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.to_dict(),
        ) from e
    except Exception as e:
        logger.exception("Unexpected server error during decision execution: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"code": "INTERNAL_SERVER_ERROR", "message": str(e)}},
        ) from e
