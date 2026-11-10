"""Artifact-generation node.

Refits the winning model on ALL available data (a final, deployment-ready fit),
then persists the complete sklearn pipeline (preprocessing + model) plus JSON
reports under ``artifacts/<experiment_id>/``. Emits ``artifact_created`` per file
and finally ``experiment_completed``.
"""
from __future__ import annotations

import joblib

from ..config import get_settings
from ..ml.training import TrainingService
from ..schemas.common import EventType
from ..schemas.plan import PreprocessingConfig
from ..services.event_bus import get_event_bus
from ..services.storage import artifacts_dir, load_dataframe, write_json
from .base import ai_message, graph_node

AGENT = "artifact_generation"


@graph_node(AGENT)
def artifact_generation_node(state: dict) -> dict:
    bus = get_event_bus()
    settings = get_settings()
    experiment_id = state["experiment_id"]

    best = state.get("best_model") or {}
    best_key = best.get("model_key")
    if not best_key:
        return {"error": "artifact_generation: no best model to persist."}

    target = state["target_column"]
    problem_type = state["problem_type"]
    cfg = PreprocessingConfig.model_validate(state["preprocessing_config"])

    directory = artifacts_dir(experiment_id)

    # Final refit on the full dataset for the deployable artifact.
    df = load_dataframe(state["dataset_path"])
    service = TrainingService(random_state=settings.random_state)
    pipeline = service.fit_final(df, target, problem_type, cfg, best_key)

    model_path = directory / "best_model.pkl"
    joblib.dump(pipeline, model_path)
    bus.emit_event(experiment_id, EventType.artifact_created, agent=AGENT,
                   message="Saved best_model.pkl (full sklearn pipeline).",
                   data={"file": "best_model.pkl"})

    # JSON artifacts.
    metrics_path = directory / "metrics.json"
    write_json(metrics_path, state.get("evaluation_results"))
    write_json(directory / "preprocessing.json", cfg.model_dump(mode="json"))
    write_json(directory / "dataset_info.json", state.get("dataset_info"))
    write_json(directory / "ml_plan.json", state.get("ml_plan"))

    report = {
        "experiment_id": experiment_id,
        "user_query": state.get("user_query"),
        "problem_type": problem_type,
        "target_column": target,
        "metric": state.get("metric"),
        "best_model": best,
        "ml_plan": state.get("ml_plan"),
        "training_meta": state.get("training_meta"),
        "evaluation": state.get("evaluation_results"),
        "review": state.get("model_review"),
        "eda": {
            "interpretation": (state.get("eda_report") or {}).get("interpretation"),
            "warnings": (state.get("eda_report") or {}).get("warnings"),
        },
        "artifacts": ["best_model.pkl", "metrics.json", "preprocessing.json",
                      "dataset_info.json", "ml_plan.json", "report.json"],
    }
    report_path = directory / "report.json"
    write_json(report_path, report)

    for name in ("metrics.json", "preprocessing.json", "dataset_info.json", "ml_plan.json", "report.json"):
        bus.emit_event(experiment_id, EventType.artifact_created, agent=AGENT,
                       message=f"Saved {name}.", data={"file": name})

    artifacts = {
        "best_model": str(model_path),
        "metrics": str(metrics_path),
        "preprocessing": str(directory / "preprocessing.json"),
        "dataset_info": str(directory / "dataset_info.json"),
        "ml_plan": str(directory / "ml_plan.json"),
        "report": str(report_path),
    }

    bus.emit_event(
        experiment_id, EventType.experiment_completed, agent=AGENT, status="completed",
        message=f"Experiment complete. Best model: {best.get('model_name')} "
                f"({best.get('metric')}={best.get('score')}).",
        data={"best_model": best, "artifacts_dir": str(directory)},
    )

    return {
        "artifacts_dir": str(directory),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "report_path": str(report_path),
        "artifacts": artifacts,
        "messages": [ai_message(f"💾 Saved deployable pipeline and reports to artifacts/{experiment_id}/.")],
    }
