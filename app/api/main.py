"""
FastAPI application factory with lifespan model management.

Startup sequence:
  1. Configure settings and logging.
  2. Initialize and validate the database (create tables if needed).
  3. Load all fund chunks from the database.
  4. Build the Qdrant dense index and BM25 sparse index from database chunks.
  5. Pre-warm embedding and reranking models (if configured).
  6. Instantiate DecisionWorkflow and register it in ServiceContainer.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes.decisions import router as decisions_router
from app.api.routes.health import router as health_router
from app.core.config import Settings, get_settings
from app.core.dependencies import ServiceContainer, get_container
from app.core.exceptions import AppError
from app.core.logging import get_logger, setup_logging
from app.db.session import create_db_engine, get_session_factory, init_database
from app.db.repository import FundRepository
from app.decision.engine import DecisionEngine
from app.evidence.validator import EvidenceValidator
from app.generation.generator import GenerationService
from app.generation.providers import get_llm_provider
from app.query.analyzer import QueryAnalyzer
from app.reranking.reranker import CrossEncoderReranker
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.retriever import EvidenceRetriever
from app.retrieval.sparse_retriever import BM25Retriever
from app.retrieval.vector_store import QdrantVectorStore
from app.workflow.graph import DecisionWorkflow

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle."""
    settings = get_settings()
    setup_logging(log_level=settings.log_level)
    logger.info("=== AI Decision Engine V2 — starting up ===")

    container = get_container()
    container.settings = settings

    # 1. Database
    engine = create_db_engine(settings=settings)
    init_database(engine)
    session_factory = get_session_factory(engine)
    container.session_factory = session_factory

    # Load chunks from DB (used to build indexes)
    with session_factory() as session:
        repo = FundRepository(session)
        chunks = repo.list_chunks()
        fund_count = repo.count_funds()

    logger.info("Database: %d funds, %d chunks loaded", fund_count, len(chunks))

    # 2. Retrieval infrastructure
    embedding_service = EmbeddingService(settings=settings)
    vector_store = QdrantVectorStore(settings=settings)
    sparse_retriever = BM25Retriever()
    retriever = EvidenceRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        sparse_retriever=sparse_retriever,
        settings=settings,
    )
    container.embedding_service = embedding_service
    container.vector_store = vector_store
    container.sparse_retriever = sparse_retriever

    if chunks:
        logger.info("Building search index from %d chunks ...", len(chunks))
        try:
            report = retriever.build_index(chunks)
            logger.info(
                "Index built: %d chunks, %s backend, %d cache hits, embed=%.0fms upsert=%.0fms",
                report.chunks_indexed, report.embedding_backend,
                report.embedding_cache_hits, report.embed_time_ms, report.upsert_time_ms,
            )
            if report.warnings:
                for w in report.warnings:
                    logger.warning("Index build warning: %s", w)
        except Exception as exc:
            logger.error("Index build failed: %s — retrieval will be unavailable", exc)
            retriever = None  # type: ignore[assignment]
    else:
        logger.warning(
            "No chunks in database; run scripts/ingest_funds.py then scripts/build_index.py."
        )

    # 3. Pre-warm models if configured
    if settings.preload_models_on_startup:
        try:
            embedding_service.warm_up()
            logger.info("Embedding model pre-warmed")
        except Exception as exc:
            logger.warning("Embedding model warm-up failed: %s", exc)

    # 4. Query analysis
    # Seed the QueryAnalyzer with known fund names so it can resolve named fund mentions
    known_funds: list = []
    if chunks:
        with session_factory() as session:
            repo2 = FundRepository(session)
            known_funds = [(r.fund_id, r.fund_name) for r in repo2.list_funds()]
    query_analyzer = QueryAnalyzer(known_funds=known_funds)

    # 5. Build DecisionWorkflow
    with session_factory() as session:
        repo3 = FundRepository(session)
        workflow = DecisionWorkflow(
            settings=settings,
            query_analyzer=query_analyzer,
            repository=repo3,
            retriever=retriever,
            reranker=CrossEncoderReranker(settings=settings),
            evidence_validator=EvidenceValidator(settings=settings),
            decision_engine=DecisionEngine(settings=settings),
            generation_service=GenerationService(
                llm_provider=get_llm_provider(settings),
                settings=settings,
            ),
        )
        container.workflow = workflow
        container.initialized = True

    logger.info("=== AI Decision Engine V2 — ready (provider=%s) ===", settings.llm_provider)
    yield

    logger.info("=== AI Decision Engine V2 — shutting down ===")


def create_app() -> FastAPI:
    """Create and configure FastAPI application instance."""
    settings = get_settings()

    app = FastAPI(
        title="FUNDIT — AI Mutual Fund Decision Engine",
        description=(
            "FUNDIT: evidence-driven mutual fund decision support. "
            "Natural-language queries → SQL candidate filtering → Hybrid RAG retrieval → "
            "Cross-encoder reranking → Deterministic scoring → Confidence gating → "
            "Grounded Gemini explanation. Gemini explains — never ranks."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )

    # ── CORS ────────────────────────────────────────────────────────────
    # In production set ALLOWED_ORIGINS env var to your actual frontend origin.
    # Falls back to same-origin only when ALLOWED_ORIGINS is not set.
    raw_origins = settings.allowed_origins
    allow_origins: list[str] = (
        [o.strip() for o in raw_origins.split(",") if o.strip()]
        if raw_origins
        else []
    )
    # If the static frontend is bundled in-process (same origin) we still
    # allow cross-origin requests from localhost ports for local dev.
    if not allow_origins:
        allow_origins = [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:3000",
        ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ── Request size guard (32 KB) ───────────────────────────────────
    @app.middleware("http")
    async def limit_request_size(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 32_768:
            return JSONResponse(
                status_code=413,
                content={"error": {"code": "REQUEST_TOO_LARGE", "message": "Request body exceeds 32 KB limit."}},
            )
        return await call_next(request)

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=exc.to_dict(),
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"code": "INTERNAL_SERVER_ERROR", "message": str(exc)}},
        )

    app.include_router(health_router)
    app.include_router(decisions_router)

    # ── Static UI ───────────────────────────────────────────────────
    static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
    if os.path.isdir(static_dir):
        # Serve the SPA root
        @app.get("/", include_in_schema=False)
        async def serve_ui():
            return FileResponse(os.path.join(static_dir, "index.html"))

        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        logger.info("Static UI mounted from %s", static_dir)

    return app


app = create_app()
