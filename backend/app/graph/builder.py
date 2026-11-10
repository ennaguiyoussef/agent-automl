"""Assemble the LangGraph AutoML workflow.

The graph is linear and matches the specified pipeline:

    START → planner → dataset_analyzer → eda → target_confirmation →
    preprocessing → model_selection → training → evaluation →
    model_reviewer → artifact_generation → END

``target_confirmation`` may pause the graph via ``interrupt`` (human-in-the-loop);
a checkpointer is therefore required so execution can resume from that point.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ..agents import (
    artifact_generation_node,
    dataset_analyzer_node,
    eda_node,
    evaluation_node,
    model_reviewer_node,
    model_selection_node,
    planner_node,
    preprocessing_node,
    target_confirmation_node,
    training_node,
)
from .state import AutoMLState

# Node order for the linear pipeline.
_PIPELINE = [
    ("planner", planner_node),
    ("dataset_analyzer", dataset_analyzer_node),
    ("eda", eda_node),
    ("target_confirmation", target_confirmation_node),
    ("preprocessing", preprocessing_node),
    ("model_selection", model_selection_node),
    ("training", training_node),
    ("evaluation", evaluation_node),
    ("model_reviewer", model_reviewer_node),
    ("artifact_generation", artifact_generation_node),
]


def build_graph(checkpointer=None):
    """Build and compile the workflow graph."""
    graph = StateGraph(AutoMLState)
    for name, fn in _PIPELINE:
        graph.add_node(name, fn)

    graph.add_edge(START, _PIPELINE[0][0])
    for (prev, _), (nxt, _) in zip(_PIPELINE, _PIPELINE[1:]):
        graph.add_edge(prev, nxt)
    graph.add_edge(_PIPELINE[-1][0], END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
