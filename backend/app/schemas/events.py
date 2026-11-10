"""Streaming event schema (Server-Sent Events)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .common import EventType


class AgentEvent(BaseModel):
    """A single structured event streamed to the frontend."""

    type: EventType
    experiment_id: str
    timestamp: str
    agent: str | None = None
    status: str | None = None  # running | completed | failed | waiting
    message: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
