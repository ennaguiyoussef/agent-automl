"""Planner agent.

Inspects dataset metadata (deterministically profiled by Python) and asks the
LLM to produce a structured :class:`MLPlan`. If the LLM is unavailable it builds
an equivalent plan from heuristics. The plan is validated against the model
registry so the LLM can only choose real, supported models.
"""
from __future__ import annotations

from ..ml.metrics import normalize_metric
from ..ml.registry import available_model_keys, default_candidates, sanitize_candidates
from ..schemas.common import EventType, ProblemType
from ..schemas.dataset import DatasetInfo
from ..schemas.plan import MLPlan
from ..services.event_bus import get_event_bus
from ..services.storage import load_dataframe
from ..tools.dataset_tools import profile_dataframe
from .base import ai_message, graph_node
from .heuristics import default_metric, guess_problem_type, guess_target
from .llm import structured_invoke

AGENT = "planner"

CONFIDENCE_THRESHOLD = 0.7  # below this, ask the human to confirm the target

_SYSTEM = """You are the planning agent of an AutoML system. Given a dataset \
profile and the user's objective, decide the machine-learning plan.

Rules:
- Choose problem_type as exactly "classification" or "regression".
- target_column MUST be one of the dataset's column names.
- Set target_confidence in [0,1] reflecting how sure you are about the target.
- recommended_metric must suit the problem (classification: f1, accuracy, \
precision, recall, roc_auc; regression: r2, rmse, mae, mse).
- candidate_models MUST be chosen ONLY from the provided allowed lists (use the \
exact keys). Pick 3-4 models that fit the data size and type; do not pick every \
model. Prefer simpler models for small/simple data.
- Explain your reasoning concisely.
You do not run code — you only produce the plan."""


def _summarize_columns(info: DatasetInfo, limit: int = 60) -> str:
    lines = []
    for col in info.columns[:limit]:
        lines.append(
            f"- {col.name}: type={col.inferred_type.value}, dtype={col.dtype}, "
            f"unique={col.unique}, missing={col.missing_pct}%, "
            f"samples={col.sample_values[:4]}"
        )
    if len(info.columns) > limit:
        lines.append(f"... and {len(info.columns) - limit} more columns")
    return "\n".join(lines)


def _build_prompt(info: DatasetInfo, user_query: str, optimize_metric: str | None) -> str:
    return (
        f"USER OBJECTIVE:\n{user_query}\n\n"
        f"DATASET PROFILE:\n"
        f"- rows: {info.n_rows}, columns: {info.n_columns}\n"
        f"- duplicate rows: {info.duplicate_rows}, total missing: {info.total_missing}\n"
        f"- numeric columns: {info.numeric_columns}\n"
        f"- categorical columns: {info.categorical_columns}\n"
        f"- constant columns: {info.constant_columns}\n"
        f"- high-cardinality columns: {info.high_cardinality_columns}\n\n"
        f"COLUMNS:\n{_summarize_columns(info)}\n\n"
        f"ALLOWED classification models: {available_model_keys('classification')}\n"
        f"ALLOWED regression models: {available_model_keys('regression')}\n\n"
        f"{('User metric preference: ' + optimize_metric) if optimize_metric else ''}\n"
        f"Return the ML plan."
    )


def _heuristic_plan(info: DatasetInfo, user_query: str, optimize_metric: str | None) -> MLPlan:
    target, confidence = guess_target(info, user_query)
    problem = guess_problem_type(info, target)
    metric = normalize_metric(optimize_metric, problem) if optimize_metric else default_metric(problem)
    return MLPlan(
        problem_type=problem,
        target_column=target or (info.column_names[-1] if info.column_names else ""),
        target_confidence=confidence,
        recommended_metric=metric,
        candidate_models=default_candidates(problem),
        reasoning=(
            "[Heuristic plan — no LLM configured] Selected the target and problem "
            f"type from column names/types: predicting '{target}' looks like a "
            f"{problem.value} task, optimising {metric}."
        ),
    )


@graph_node(AGENT)
def planner_node(state: dict) -> dict:
    bus = get_event_bus()
    experiment_id = state.get("experiment_id")
    user_query = state.get("user_query", "")
    explicit_target = state.get("target_column")
    optimize_metric = state.get("optimize_metric")

    bus.emit_event(experiment_id, EventType.planner_started, agent=AGENT, status="running",
                   message="Analysing the request and dataset metadata…")

    df = load_dataframe(state["dataset_path"])
    info = profile_dataframe(df)
    bus.emit_event(
        experiment_id, EventType.dataset_loaded, agent=AGENT,
        message=f"Loaded dataset: {info.n_rows} rows × {info.n_columns} columns.",
        data={"n_rows": info.n_rows, "n_columns": info.n_columns},
    )

    # Ask the LLM; fall back to heuristics.
    plan = structured_invoke(MLPlan, _SYSTEM, _build_prompt(info, user_query, optimize_metric))
    used_llm = plan is not None
    if plan is None:
        plan = _heuristic_plan(info, user_query, optimize_metric)

    # ---- validate / repair the plan against ground truth ----------------
    if plan.target_column not in info.column_names:
        h_target, h_conf = guess_target(info, user_query)
        plan.target_column = h_target or info.column_names[-1]
        plan.target_confidence = min(plan.target_confidence, h_conf)

    if explicit_target and explicit_target in info.column_names:
        plan.target_column = explicit_target
        plan.target_confidence = 1.0

    problem_type = plan.problem_type
    metric = normalize_metric(optimize_metric or plan.recommended_metric, problem_type)
    plan.recommended_metric = metric
    candidates = sanitize_candidates(problem_type, plan.candidate_models)
    plan.candidate_models = candidates

    needs_human = plan.target_confidence < CONFIDENCE_THRESHOLD and not explicit_target

    bus.emit_event(
        experiment_id, EventType.planner_completed, agent=AGENT, status="completed",
        message=(
            f"Plan ready: {problem_type.value} predicting '{plan.target_column}', "
            f"optimising {metric} across {len(candidates)} models."
        ),
        data={
            "plan": plan.model_dump(mode="json"),
            "used_llm": used_llm,
            "needs_human": needs_human,
        },
    )

    return {
        "dataset_info": info.model_dump(mode="json"),
        "ml_plan": plan.model_dump(mode="json"),
        "target_column": plan.target_column,
        "problem_type": problem_type.value,
        "metric": metric,
        "candidate_models": candidates,
        "human_approval_required": needs_human,
        "target_confirmed": not needs_human,
        "messages": [ai_message(f"📋 Plan: {plan.reasoning}")],
    }
