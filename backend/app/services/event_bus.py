"""Event bus that bridges the (synchronous, thread-run) LangGraph nodes to the
async Server-Sent-Events endpoints.

Design notes:
* Graph nodes run in a worker thread (scikit-learn training is blocking), so
  ``emit`` must be thread-safe. It simply schedules delivery onto the FastAPI
  event loop via ``call_soon_threadsafe``.
* All mutation of channel state happens *on the loop thread* (inside
  ``_deliver`` / ``subscribe``), so no locks are needed and there are no
  lost/duplicated events for late subscribers (SSE reconnects replay history).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import AsyncIterator

from ..logging_config import get_logger
from ..schemas.common import EventType
from ..schemas.events import AgentEvent

logger = get_logger(__name__)

# Cap replay history so long experiments do not grow without bound.
MAX_HISTORY = 1000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class _Channel:
    def __init__(self) -> None:
        self.history: list[AgentEvent] = []
        self.subscribers: list[asyncio.Queue] = []


class EventBus:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._channels: dict[str, _Channel] = {}

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # ---- producer side (any thread) -------------------------------------
    def emit(self, event: AgentEvent) -> None:
        if self._loop is None:
            return  # no consumer loop (e.g. unit tests) -> silently drop
        try:
            self._loop.call_soon_threadsafe(self._deliver, event)
        except RuntimeError:
            # Loop is closed/closing — nothing to deliver to.
            pass

    def emit_event(
        self,
        experiment_id: str | None,
        event_type: EventType,
        *,
        agent: str | None = None,
        status: str | None = None,
        message: str | None = None,
        data: dict | None = None,
    ) -> None:
        if not experiment_id:
            return
        self.emit(
            AgentEvent(
                type=event_type,
                experiment_id=experiment_id,
                timestamp=_now_iso(),
                agent=agent,
                status=status,
                message=message,
                data=data or {},
            )
        )

    def _deliver(self, event: AgentEvent) -> None:  # runs on the loop thread
        channel = self._channels.setdefault(event.experiment_id, _Channel())
        channel.history.append(event)
        if len(channel.history) > MAX_HISTORY:
            channel.history = channel.history[-MAX_HISTORY:]
        for queue in channel.subscribers:
            queue.put_nowait(event)

    # ---- consumer side (loop thread) ------------------------------------
    async def subscribe(self, experiment_id: str) -> AsyncIterator[AgentEvent]:
        channel = self._channels.setdefault(experiment_id, _Channel())
        queue: asyncio.Queue = asyncio.Queue()
        backlog = list(channel.history)  # snapshot — no await before append == atomic
        channel.subscribers.append(queue)
        try:
            for event in backlog:
                yield event
            while True:
                yield await queue.get()
        finally:
            if queue in channel.subscribers:
                channel.subscribers.remove(queue)

    def history(self, experiment_id: str) -> list[AgentEvent]:
        channel = self._channels.get(experiment_id)
        return list(channel.history) if channel else []


# Process-wide singleton — nodes reach it via ``get_event_bus()``.
_BUS: EventBus | None = None


def get_event_bus() -> EventBus:
    global _BUS
    if _BUS is None:
        _BUS = EventBus()
    return _BUS
