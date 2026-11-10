"""Experiment endpoints: create, inspect, chat, stream (SSE), human-in-the-loop,
results, and artifact downloads.

Routes stay thin — all orchestration lives in :class:`ExperimentManager`.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from ..logging_config import get_logger
from ..schemas.evaluation import ResultsResponse
from ..schemas.experiment import (
    ChatRequest,
    ChatResponse,
    ExperimentCreateRequest,
    ExperimentSnapshot,
    HumanResponseRequest,
)
from ..services.experiment_manager import ExperimentManager
from .deps import get_manager

logger = get_logger(__name__)

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.post(
    "",
    response_model=ExperimentSnapshot,
    status_code=status.HTTP_201_CREATED,
    summary="Create an experiment and start the AutoML workflow",
)
async def create_experiment(
    req: ExperimentCreateRequest,
    manager: ExperimentManager = Depends(get_manager),
) -> ExperimentSnapshot:
    return manager.create_experiment(req)


@router.get("", response_model=list[ExperimentSnapshot], summary="List experiments")
async def list_experiments(
    manager: ExperimentManager = Depends(get_manager),
) -> list[ExperimentSnapshot]:
    return manager.list_experiments()


@router.get(
    "/{experiment_id}",
    response_model=ExperimentSnapshot,
    summary="Get an experiment snapshot",
)
async def get_experiment(
    experiment_id: str,
    manager: ExperimentManager = Depends(get_manager),
) -> ExperimentSnapshot:
    return manager.snapshot(experiment_id)


@router.post(
    "/{experiment_id}/chat",
    response_model=ChatResponse,
    summary="Send a chat message to the experiment agent",
)
async def chat(
    experiment_id: str,
    req: ChatRequest,
    manager: ExperimentManager = Depends(get_manager),
) -> ChatResponse:
    return await manager.chat(experiment_id, req.message)


@router.get(
    "/{experiment_id}/events",
    summary="Server-Sent Events stream of agent activity",
)
async def stream_events(
    experiment_id: str,
    request: Request,
    manager: ExperimentManager = Depends(get_manager),
) -> EventSourceResponse:
    manager.snapshot(experiment_id)  # raises -> 404 if unknown

    async def event_generator():
        async for event in manager.event_stream(experiment_id):
            if await request.is_disconnected():
                break
            yield {"event": event.type.value, "data": event.model_dump_json()}

    return EventSourceResponse(event_generator())


@router.post(
    "/{experiment_id}/human-response",
    response_model=ExperimentSnapshot,
    summary="Resume a paused experiment with the user's target confirmation",
)
async def human_response(
    experiment_id: str,
    req: HumanResponseRequest,
    manager: ExperimentManager = Depends(get_manager),
) -> ExperimentSnapshot:
    return await manager.submit_human_response(experiment_id, req)


@router.get(
    "/{experiment_id}/results",
    response_model=ResultsResponse,
    summary="Get the evaluation results and review for a completed experiment",
)
async def get_results(
    experiment_id: str,
    manager: ExperimentManager = Depends(get_manager),
) -> ResultsResponse:
    return manager.results(experiment_id)


@router.get(
    "/{experiment_id}/download-model",
    summary="Download the trained model pipeline (.pkl)",
)
async def download_model(
    experiment_id: str,
    manager: ExperimentManager = Depends(get_manager),
) -> FileResponse:
    path = manager.artifact_path(experiment_id, "model")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=f"best_model_{experiment_id}.pkl",
    )


@router.get(
    "/{experiment_id}/download-report",
    summary="Download the experiment report (JSON)",
)
async def download_report(
    experiment_id: str,
    manager: ExperimentManager = Depends(get_manager),
) -> FileResponse:
    path = manager.artifact_path(experiment_id, "report")
    return FileResponse(
        path,
        media_type="application/json",
        filename=f"report_{experiment_id}.json",
    )
