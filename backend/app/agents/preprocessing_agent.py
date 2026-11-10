"""Preprocessing node — builds the leakage-safe preprocessing configuration."""
from __future__ import annotations

from ..ml.preprocessing import build_preprocessing_config
from ..schemas.common import EventType
from ..schemas.dataset import DatasetInfo
from ..services.event_bus import get_event_bus
from .base import ai_message, graph_node

AGENT = "preprocessing"


@graph_node(AGENT)
def preprocessing_node(state: dict) -> dict:
    bus = get_event_bus()
    experiment_id = state.get("experiment_id")
    bus.emit_event(experiment_id, EventType.preprocessing_started, agent=AGENT,
                   status="running", message="Designing the preprocessing pipeline…")

    info = DatasetInfo.model_validate(state["dataset_info"])
    target = state["target_column"]
    cfg = build_preprocessing_config(info, target)

    msg = (
        f"Preprocessing: {len(cfg.numeric_features)} numeric "
        f"(impute median{'+scale' if cfg.scale_numeric else ''}), "
        f"{len(cfg.categorical_features)} categorical (impute mode + one-hot), "
        f"{len(cfg.dropped_features)} dropped."
    )
    bus.emit_event(
        experiment_id, EventType.preprocessing_completed, agent=AGENT, status="completed",
        message=msg, data={"preprocessing_config": cfg.model_dump(mode="json")},
    )
    return {"preprocessing_config": cfg.model_dump(mode="json"), "messages": [ai_message(f"🧪 {msg}")]}
