"""
GeoAI Agent Backend — FastAPI Application Entry Point.

Run with:
    uvicorn app.main:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agents.intent_agent import get_intent_graph
from app.agents.orchestrator import get_orchestrator_graph
from app.core.config import get_settings
from app.core.database import engine
from app.core.logging import get_logger, setup_logging
from app.models.database import Base
from app.services.llm import get_llm_service

logger = get_logger(__name__)


# ── Lifespan ─────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.

    Startup:
      - Configure logging
      - Create database tables (dev only — use Alembic migrations in production)

    Shutdown:
      - Dispose of the database engine connection pool
    """
    # Startup
    setup_logging()
    logger = get_logger(__name__)
    settings = get_settings()

    logger.info("Starting %s v%s [%s]", settings.app_name, settings.app_version, settings.app_env)

    # Warm up singletons at startup so the first request isn't slower
    # and misconfigured env vars fail fast here rather than mid-request.
    get_llm_service()
    get_intent_graph()
    get_orchestrator_graph()
    logger.info("LLM service, intent graph, and orchestrator graph initialised")

    # Initialize Google Earth Engine
    from app.analysis.base import init_gee
    try:
        init_gee()
        logger.info("Google Earth Engine initialised")
    except Exception as e:
        logger.error("Failed to initialise Google Earth Engine: %s", str(e))

    # Create tables — safe for dev. In production, use Alembic.
    if settings.is_development:
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                logger.info("Database tables created/verified")
        except Exception as e:
            logger.warning("Database connection failed — server will start without DB: %s", str(e))

    yield

    # Shutdown
    await engine.dispose()
    logger.info("Database connections closed")


# ── App Factory ──────────────────────────────────


def create_app() -> FastAPI:
    """
    Application factory.

    Constructs the FastAPI app with all middleware, routes,
    and configuration. Using a factory makes testing easier.
    """
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "GeoAI Agent Backend — An AI-powered geospatial analysis engine. "
            "Accepts natural language prompts and returns structured analysis plans, "
            "maps, and reports."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ─────────────────────────────────────
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Static Maps Server ───────────────────────
    # We host the output_maps directory so the frontend can fetch GeoJSON/PNG files
    from fastapi.staticfiles import StaticFiles
    maps_dir = os.path.abspath(settings.static_maps_dir)
    os.makedirs(maps_dir, exist_ok=True)
    
    application.mount("/api/v1/static/maps", StaticFiles(directory=maps_dir), name="maps")
    logger.info("Static maps served from: %s", maps_dir)

    # ── Routes ───────────────────────────────────
    from app.api.v1.router import router as v1_router
    from app.models.schemas import HealthResponse

    application.include_router(v1_router)

    @application.get(
        "/",
        response_model=HealthResponse,
        summary="Health check",
        tags=["system"],
        methods=["GET", "HEAD"],
    )
    async def health_check() -> HealthResponse:
        return HealthResponse(
            app_name=settings.app_name,
            version=settings.app_version,
            environment=settings.app_env,
        )

    return application


app = create_app()
