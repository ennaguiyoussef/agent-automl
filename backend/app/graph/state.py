"""The typed shared state that flows through the LangGraph workflow.

We deliberately store *JSON-serialisable dicts* (produced via ``model_dump()``)
rather than Pydantic instances in the state. This keeps LangGraph checkpointing
simple and lets every field be streamed to the frontend without custom encoders.
Pydantic models are still used at the boundaries (inside each agent) for
validation — they are dumped to dicts before being written back to state.
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AutoMLState(TypedDict, total=False):
    # ---- Inputs ---------------------------------------------------------
    experiment_id: str
    user_query: str
    dataset_path: str
    dataset_id: str
    optimize_metric: str | None

    # ---- Profiling / analysis ------------------------------------------
    dataset_info: dict[str, Any] | None
    eda_report: dict[str, Any] | None

    # ---- Plan -----------------------------------------------------------
    ml_plan: dict[str, Any] | None
    target_column: str | None
    problem_type: str | None
    metric: str | None
    candidate_models: list[str]

    # ---- Human-in-the-loop ---------------------------------------------
    human_approval_required: bool
    human_feedback: str | None
    target_confirmed: bool

    # ---- Preprocessing / training / evaluation -------------------------
    preprocessing_config: dict[str, Any] | None
    trained_models: list[dict[str, Any]]
    training_meta: dict[str, Any] | None
    evaluation_results: dict[str, Any] | None
    best_model: dict[str, Any] | None
    model_review: dict[str, Any] | None

    # ---- Artifacts ------------------------------------------------------
    artifacts_dir: str | None
    model_path: str | None
    metrics_path: str | None
    report_path: str | None
    artifacts: dict[str, str]

    # ---- Control / misc -------------------------------------------------
    error: str | None
    messages: Annotated[list, add_messages]
