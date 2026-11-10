"""Shared plumbing for graph nodes: an error-isolating decorator and helpers.

The decorator makes every node resilient — an exception in one node is caught,
turned into an ``error`` on the state and an ``error`` SSE event, and downstream
nodes short-circuit. The one exception we must re-raise is ``GraphInterrupt``,
which LangGraph uses internally to pause for human input.
"""
from __future__ import annotations

from functools import wraps
from typing import Callable

from langchain_core.messages import AIMessage
from langgraph.errors import GraphInterrupt

from ..logging_config import get_logger
from ..schemas.common import EventType
from ..services.event_bus import get_event_bus

logger = get_logger(__name__)


def ai_message(text: str) -> AIMessage:
    return AIMessage(content=text)


def graph_node(agent_name: str) -> Callable:
    """Wrap a node function with error isolation + skip-on-prior-error."""

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(state: dict) -> dict:
            bus = get_event_bus()
            experiment_id = state.get("experiment_id")
            if state.get("error"):
                # A previous node already failed — skip quietly.
                return {}
            try:
                return fn(state)
            except GraphInterrupt:
                raise  # human-in-the-loop pause — must propagate
            except Exception as exc:  # noqa: BLE001
                logger.exception("Node '%s' raised", agent_name)
                bus.emit_event(
                    experiment_id,
                    EventType.error,
                    agent=agent_name,
                    status="failed",
                    message=f"{agent_name} failed: {exc}",
                )
                return {"error": f"{agent_name}: {exc}"}

        return wrapper

    return decorator
