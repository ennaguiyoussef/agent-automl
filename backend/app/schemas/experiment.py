"""Experiment lifecycle and API request/response schemas."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .common import ExperimentStatus, ProblemType


class ExperimentCreateRequest(BaseModel):
    dataset_id: str = Field(description="ID returned by the dataset upload endpoint.")
    user_query: str = Field(
        min_length=1,
        description="Natural-language description of the ML objective.",
    )
    target_column: str | None = Field(
        default=None, description="Optional explicit target column."
    )
    optimize_metric: str | None = Field(
        default=None, description="Optional metric preference (e.g. recall)."
    )


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    reply: str
    intent: str = "chat"
    actions: list[str] = Field(default_factory=list)


class HumanResponseRequest(BaseModel):
    """Payload the frontend sends to resume an interrupted graph."""

    target_column: str | None = None
    approve: bool | None = None
    message: str | None = None


class PendingHumanRequest(BaseModel):
    """Describes what the graph is waiting for while paused."""

    reason: str
    question: str
    options: list[str] = Field(default_factory=list)
    suggested: str | None = None


class ExperimentSnapshot(BaseModel):
    """Public snapshot of an experiment's state."""

    id: str
    status: ExperimentStatus
    user_query: str
    dataset_id: str | None = None
    target_column: str | None = None
    problem_type: ProblemType | None = None
    metric: str | None = None
    created_at: str
    updated_at: str
    error: str | None = None
    has_results: bool = False
    pending_human: PendingHumanRequest | None = None
    plan: dict[str, Any] | None = None
