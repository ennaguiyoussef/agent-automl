"""Model-selection node — validates the planner's candidate list against the
registry (the planner never trains everything blindly; it chooses, and this node
guarantees the choices are real, supported models)."""
from __future__ import annotations

from ..ml.registry import get_model_spec, sanitize_candidates
from ..schemas.common import EventType
from ..services.event_bus import get_event_bus
from .base import ai_message, graph_node

AGENT = "model_selection"


@graph_node(AGENT)
def model_selection_node(state: dict) -> dict:
    bus = get_event_bus()
    experiment_id = state.get("experiment_id")
    problem_type = state["problem_type"]

    candidates = sanitize_candidates(problem_type, state.get("candidate_models", []))
    names = [(get_model_spec(problem_type, k).name if get_model_spec(problem_type, k) else k) for k in candidates]

    msg = f"Selected {len(candidates)} candidate models: {', '.join(names)}."
    bus.emit_event(
        experiment_id, EventType.model_selection_completed, agent=AGENT, status="completed",
        message=msg, data={"candidate_models": candidates, "model_names": names},
    )
    return {"candidate_models": candidates, "messages": [ai_message(f"🤖 {msg}")]}
