"""Training node — runs the deterministic TrainingService and streams per-model
progress events (model_training / model_completed) to the frontend."""
from __future__ import annotations

from ..config import get_settings
from ..ml.training import TrainingService
from ..schemas.common import EventType
from ..schemas.plan import PreprocessingConfig
from ..services.event_bus import get_event_bus
from ..services.storage import load_dataframe
from .base import ai_message, graph_node

AGENT = "training"


@graph_node(AGENT)
def training_node(state: dict) -> dict:
    bus = get_event_bus()
    settings = get_settings()
    experiment_id = state.get("experiment_id")

    target = state["target_column"]
    problem_type = state["problem_type"]
    metric = state["metric"]
    candidates = state["candidate_models"]
    cfg = PreprocessingConfig.model_validate(state["preprocessing_config"])

    bus.emit_event(
        experiment_id, EventType.training_started, agent=AGENT, status="running",
        message=f"Training {len(candidates)} models (optimising {metric})…",
        data={"candidate_models": candidates, "metric": metric},
    )

    df = load_dataframe(state["dataset_path"])
    service = TrainingService(
        random_state=settings.random_state,
        test_size=settings.test_size,
        cv_folds=settings.cv_folds,
        max_training_rows=settings.max_training_rows,
    )

    def on_model(kind: str, payload: dict) -> None:
        event_type = EventType.model_training if kind == "model_training" else EventType.model_completed
        status = payload.get("status")
        name = payload.get("model_name")
        if event_type == EventType.model_training:
            message = f"Training {name}…"
        elif status == "failed":
            message = f"{name} failed: {payload.get('error')}"
        else:
            score = payload.get("primary_score")
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
            message = f"{name} done — {payload.get('primary_metric')}={score_str} ({payload.get('training_time_sec')}s)"
        bus.emit_event(experiment_id, event_type, agent=AGENT, status=status, message=message, data=payload)

    outcome = service.train_candidates(
        df, target, problem_type, metric, candidates, cfg, on_model=on_model
    )

    trained = [r.model_dump(mode="json") for r in outcome.results]
    n_ok = sum(1 for r in outcome.results if r.status == "success")
    return {
        "trained_models": trained,
        "training_meta": {
            "n_train": outcome.split.n_train,
            "n_test": outcome.split.n_test,
            "cv_folds": outcome.split.cv_folds,
            "n_classes": outcome.split.n_classes,
            "stratified": outcome.split.stratified,
            "subsampled_from": outcome.split.subsampled_from,
        },
        "messages": [ai_message(f"🏋️ Trained {n_ok}/{len(candidates)} models successfully.")],
    }
