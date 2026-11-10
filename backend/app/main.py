"""FastAPI application entrypoint for the Agentic AutoML backend.

Wires together configuration, logging, CORS, the event bus (bound to the running
loop so worker-thread graph nodes can stream SSE), and the API routers. Domain
exceptions are translated to HTTP status codes here so the routes stay thin.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .agents.llm import llm_available
from .api import datasets_router, experiments_router
from .config import get_settings
from .logging_config import configure_logging, get_logger
from .services.csv_validation import DatasetValidationError
from .services.event_bus import get_event_bus
from .services.experiment_manager import (
    ExperimentNotFound,
    ExperimentStateError,
    get_experiment_manager,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_dirs()

    # Bind the event bus to the running loop so nodes (in worker threads) can
    # deliver SSE events thread-safely.
    get_event_bus().bind_loop(asyncio.get_running_loop())

    # Eagerly build the manager so the graph compiles at startup, not on the
    # first request.
    get_experiment_manager()

    logger.info(
        "%s started (env=%s, llm=%s/%s, key=%s)",
        settings.app_name,
        settings.environment,
        settings.llm_provider,
        settings.llm_model,
        "present" if llm_available() else "absent (heuristic fallback)",
    )
    yield
    logger.info("%s shutting down", settings.app_name)


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Agentic AutoML — an LLM-planned, deterministically-executed ML pipeline.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- exception handlers (domain errors -> HTTP) ------------------------
@app.exception_handler(ExperimentNotFound)
async def _handle_not_found(_: Request, exc: ExperimentNotFound) -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})


@app.exception_handler(ExperimentStateError)
async def _handle_state_error(_: Request, exc: ExperimentStateError) -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": str(exc)})


@app.exception_handler(FileNotFoundError)
async def _handle_missing_file(_: Request, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})


@app.exception_handler(DatasetValidationError)
async def _handle_validation(_: Request, exc: DatasetValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


# ---- routes ------------------------------------------------------------
@app.get("/api/health", tags=["meta"])
async def health() -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "llm_available": llm_available(),
        "llm_provider": settings.llm_provider,
    }


app.include_router(datasets_router)
app.include_router(experiments_router)
