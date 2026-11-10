"""Dataset analyzer node.

The full deterministic profile was computed by the planner; this node formalises
it, computes the target distribution now that a target is known, and emits the
analysis to the frontend. All numbers come from Python tools.
"""
from __future__ import annotations

from ..schemas.common import EventType
from ..services.event_bus import get_event_bus
from ..services.storage import load_dataframe
from ..tools.dataset_tools import profile_dataframe, target_distribution
from .base import ai_message, graph_node

AGENT = "dataset_analyzer"


@graph_node(AGENT)
def dataset_analyzer_node(state: dict) -> dict:
    bus = get_event_bus()
    experiment_id = state.get("experiment_id")
    bus.emit_event(experiment_id, EventType.dataset_analysis_started, agent=AGENT,
                   status="running", message="Profiling the dataset…")

    info = state.get("dataset_info")
    df = load_dataframe(state["dataset_path"])
    if not info:
        info = profile_dataframe(df).model_dump(mode="json")

    target = state.get("target_column")
    dist = target_distribution(df, target) if target else None

    msg = (
        f"Analysed {info['n_rows']} rows × {info['n_columns']} columns — "
        f"{len(info['numeric_columns'])} numeric, {len(info['categorical_columns'])} categorical, "
        f"{info['duplicate_rows']} duplicate rows, {info['total_missing']} missing values."
    )
    bus.emit_event(
        experiment_id, EventType.dataset_analysis_completed, agent=AGENT, status="completed",
        message=msg,
        data={"dataset_info": info, "target_distribution": dist, "target_column": target},
    )
    return {"dataset_info": info, "messages": [ai_message(f"🔍 {msg}")]}
