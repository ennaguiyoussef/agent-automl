"""Human target-confirmation node (human-in-the-loop).

If the planner was confident about the target column, this node is a no-op. If
not, it calls LangGraph's :func:`interrupt`, which pauses the whole graph and
persists a checkpoint. The API layer detects the pause, streams a
``human_input_required`` event, and later resumes the graph with the user's
answer via ``Command(resume=...)`` — at which point ``interrupt`` returns that
answer and execution continues.
"""
from __future__ import annotations

from langgraph.types import interrupt

from ..ml.metrics import normalize_metric
from ..ml.registry import default_candidates
from ..schemas.common import EventType
from ..schemas.dataset import DatasetInfo
from ..services.event_bus import get_event_bus
from .base import ai_message, graph_node
from .heuristics import guess_problem_type

AGENT = "target_confirmation"


def _apply_answer(state: dict, answer) -> dict:
    bus = get_event_bus()
    experiment_id = state.get("experiment_id")
    info_dict = state.get("dataset_info") or {}
    columns = info_dict.get("column_names", [])

    chosen: str | None = None
    feedback: str | None = None
    if isinstance(answer, dict):
        chosen = answer.get("target_column") or answer.get("target")
        feedback = answer.get("message")
    elif isinstance(answer, str):
        chosen = answer

    updates: dict = {"target_confirmed": True, "human_approval_required": False}
    if feedback:
        updates["human_feedback"] = feedback

    current = state.get("target_column")
    if chosen and chosen in columns and chosen != current:
        info_model = DatasetInfo.model_validate(info_dict)
        new_problem = guess_problem_type(info_model, chosen)
        new_metric = normalize_metric(state.get("optimize_metric") or state.get("metric"), new_problem)
        updates.update(
            {
                "target_column": chosen,
                "problem_type": new_problem.value,
                "metric": new_metric,
                "candidate_models": default_candidates(new_problem),
            }
        )
        message = (
            f"Target set to '{chosen}'. Re-evaluated as a {new_problem.value} "
            f"problem optimising {new_metric}."
        )
    else:
        chosen = chosen or current
        message = f"Target column '{chosen}' confirmed."

    bus.emit_event(
        experiment_id, EventType.human_input_received, agent=AGENT, status="completed",
        message=message, data={"target_column": chosen},
    )
    updates["messages"] = [ai_message(f"✅ {message}")]
    return updates


@graph_node(AGENT)
def target_confirmation_node(state: dict) -> dict:
    # Confident target -> proceed without human interaction.
    if state.get("target_confirmed") or not state.get("human_approval_required"):
        return {"target_confirmed": True, "human_approval_required": False}

    info = state.get("dataset_info") or {}
    suggested = state.get("target_column")
    payload = {
        "reason": "target_uncertain",
        "question": (
            f"I'm not fully confident which column to predict — my best guess is "
            f"'{suggested}'. Please confirm or choose the correct target column."
        ),
        "options": info.get("column_names", []),
        "suggested": suggested,
    }
    # Pauses the graph. The API layer emits the SSE `human_input_required` event
    # when it observes this interrupt, so we don't emit it here (avoids a
    # duplicate emission when the node re-runs on resume).
    answer = interrupt(payload)
    return _apply_answer(state, answer)
