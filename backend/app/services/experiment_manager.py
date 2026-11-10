"""Experiment orchestration.

Owns the lifecycle of every experiment: it runs the compiled LangGraph workflow
in a worker thread (scikit-learn training is blocking), detects the
human-in-the-loop pause, resumes on the user's answer, and exposes read-only
snapshots / results to the API layer.

Design:
* One compiled graph per process, sharing a single ``MemorySaver`` checkpointer.
  Each experiment uses its own ``thread_id`` (== experiment id), so concurrent
  experiments never collide and a paused graph can be resumed later.
* The graph runs inside ``asyncio.to_thread`` so the event loop stays free to
  serve SSE subscribers. Nodes stream progress through the event bus, which is
  bound to the loop at startup.
* Business logic lives here, never in the FastAPI routes.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from langgraph.types import Command

from ..agents.llm import text_invoke
from ..graph import build_graph
from ..logging_config import get_logger
from ..schemas.common import EventType, ExperimentStatus
from ..schemas.evaluation import EvaluationReport, ModelReview, ResultsResponse
from ..schemas.events import AgentEvent
from ..schemas.experiment import (
    ChatResponse,
    ExperimentCreateRequest,
    ExperimentSnapshot,
    HumanResponseRequest,
    PendingHumanRequest,
)
from .event_bus import get_event_bus
from .storage import dataset_exists, dataset_path, new_id

logger = get_logger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExperimentNotFound(KeyError):
    """Raised when an experiment id is unknown."""


class ExperimentStateError(RuntimeError):
    """Raised when an operation is invalid for the experiment's current status."""


@dataclass
class ExperimentRecord:
    """In-memory record of a single experiment run."""

    id: str
    user_query: str
    dataset_id: str
    dataset_path: str
    optimize_metric: str | None = None
    explicit_target: str | None = None

    status: ExperimentStatus = ExperimentStatus.created
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    target_column: str | None = None
    problem_type: str | None = None
    metric: str | None = None
    plan: dict[str, Any] | None = None
    pending_human: PendingHumanRequest | None = None
    error: str | None = None
    final_state: dict[str, Any] | None = None

    task: asyncio.Task | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_AFFIRMATIONS = (
    "yes", "yep", "yeah", "correct", "confirm", "confirmed", "approve",
    "approved", "proceed", "go ahead", "sounds good", "okay", "sure",
    "use it", "that's right", "looks good", "confirmed.",
)


class ExperimentManager:
    """Runs and tracks AutoML experiments."""

    def __init__(self) -> None:
        # Single compiled graph + MemorySaver shared across the process.
        self._graph = build_graph()
        self._experiments: dict[str, ExperimentRecord] = {}
        self._bus = get_event_bus()

    # ---- registry --------------------------------------------------------
    def get(self, experiment_id: str) -> ExperimentRecord | None:
        return self._experiments.get(experiment_id)

    def _require(self, experiment_id: str) -> ExperimentRecord:
        record = self._experiments.get(experiment_id)
        if record is None:
            raise ExperimentNotFound(experiment_id)
        return record

    def list_experiments(self) -> list[ExperimentSnapshot]:
        return [self.snapshot(rec.id) for rec in self._experiments.values()]

    # ---- creation / execution -------------------------------------------
    def create_experiment(self, req: ExperimentCreateRequest) -> ExperimentSnapshot:
        """Register an experiment and kick off its run in the background."""
        if not dataset_exists(req.dataset_id):
            raise ExperimentNotFound(f"dataset {req.dataset_id} not found")

        experiment_id = new_id()
        record = ExperimentRecord(
            id=experiment_id,
            user_query=req.user_query,
            dataset_id=req.dataset_id,
            dataset_path=str(dataset_path(req.dataset_id)),
            optimize_metric=req.optimize_metric,
            explicit_target=req.target_column,
        )
        self._experiments[experiment_id] = record

        record.task = asyncio.create_task(self._start(record))
        record.task.add_done_callback(self._on_task_done)
        return self.snapshot(experiment_id)

    def _on_task_done(self, task: asyncio.Task) -> None:
        # _drive already catches node errors; this only surfaces truly
        # unexpected failures (e.g. cancellation) without crashing the loop.
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Experiment task ended with an unhandled exception: %s", exc)

    def _initial_input(self, record: ExperimentRecord) -> dict[str, Any]:
        graph_input: dict[str, Any] = {
            "experiment_id": record.id,
            "user_query": record.user_query,
            "dataset_path": record.dataset_path,
            "dataset_id": record.dataset_id,
        }
        if record.explicit_target:
            graph_input["target_column"] = record.explicit_target
        if record.optimize_metric:
            graph_input["optimize_metric"] = record.optimize_metric
        return graph_input

    async def _start(self, record: ExperimentRecord) -> None:
        self._bus.emit_event(
            record.id, EventType.experiment_started, agent="orchestrator", status="running",
            message=f"Starting experiment: {record.user_query}",
            data={"dataset_id": record.dataset_id},
        )
        await self._drive(record, self._initial_input(record))

    async def _drive(self, record: ExperimentRecord, graph_input: Any) -> None:
        """Invoke (or resume) the graph in a worker thread and absorb the result."""
        async with record.lock:
            record.status = ExperimentStatus.running
            record.updated_at = _now()
            config = {"configurable": {"thread_id": record.id}}
            try:
                result = await asyncio.to_thread(self._graph.invoke, graph_input, config)
            except Exception as exc:  # noqa: BLE001 — surface any orchestration failure
                logger.exception("Experiment %s crashed", record.id)
                record.status = ExperimentStatus.failed
                record.error = str(exc)
                record.updated_at = _now()
                self._bus.emit_event(
                    record.id, EventType.error, agent="orchestrator", status="failed",
                    message=f"Experiment failed: {exc}",
                )
                return

            self._absorb(record, result)
            snapshot = self._graph.get_state(config)
            pending = self._extract_interrupt(result, snapshot)

            if getattr(snapshot, "next", None) and pending is not None:
                self._enter_awaiting(record, pending)
            elif record.error:
                record.status = ExperimentStatus.failed
                record.pending_human = None
            else:
                record.status = ExperimentStatus.completed
                record.pending_human = None
            record.updated_at = _now()

    def _absorb(self, record: ExperimentRecord, state: Any) -> None:
        """Copy the interesting fields out of the (JSON-serialisable) graph state."""
        if not isinstance(state, dict):
            return
        record.final_state = state
        record.target_column = state.get("target_column") or record.target_column
        record.problem_type = state.get("problem_type") or record.problem_type
        record.metric = state.get("metric") or record.metric
        if state.get("ml_plan"):
            record.plan = state["ml_plan"]
        if state.get("error"):
            record.error = state["error"]

    def _enter_awaiting(self, record: ExperimentRecord, pending: PendingHumanRequest) -> None:
        record.status = ExperimentStatus.awaiting_human
        record.pending_human = pending
        # The target_confirmation node deliberately does NOT emit this event
        # (it would re-fire on resume); the orchestrator emits it once here.
        self._bus.emit_event(
            record.id, EventType.human_input_required, agent="target_confirmation",
            status="waiting", message=pending.question, data=pending.model_dump(),
        )

    @staticmethod
    def _extract_interrupt(result: Any, snapshot: Any) -> PendingHumanRequest | None:
        payload: Any = None
        if isinstance(result, dict):
            interrupts = result.get("__interrupt__")
            if interrupts:
                first = interrupts[0]
                payload = getattr(first, "value", first)
        if payload is None:
            for task in getattr(snapshot, "tasks", ()) or ():
                for intr in getattr(task, "interrupts", ()) or ():
                    payload = getattr(intr, "value", intr)
                    break
                if payload is not None:
                    break
        if payload is None:
            return None
        if isinstance(payload, dict):
            return PendingHumanRequest(
                reason=payload.get("reason", "input_required"),
                question=payload.get("question", "Please provide input to continue."),
                options=list(payload.get("options", []) or []),
                suggested=payload.get("suggested"),
            )
        return PendingHumanRequest(reason="input_required", question=str(payload))

    # ---- human-in-the-loop ----------------------------------------------
    async def submit_human_response(
        self, experiment_id: str, req: HumanResponseRequest
    ) -> ExperimentSnapshot:
        record = self._require(experiment_id)
        if record.status != ExperimentStatus.awaiting_human:
            raise ExperimentStateError(
                f"Experiment is not awaiting human input (status={record.status.value})."
            )
        answer: dict[str, Any] = {}
        if req.target_column:
            answer["target_column"] = req.target_column
        if req.message:
            answer["message"] = req.message

        record.pending_human = None
        await self._drive(record, Command(resume=answer))
        return self.snapshot(experiment_id)

    # ---- chat ------------------------------------------------------------
    async def chat(self, experiment_id: str, message: str) -> ChatResponse:
        record = self._require(experiment_id)
        self._bus.emit_event(
            record.id, EventType.chat_message, agent="user", message=message,
            data={"role": "user"},
        )

        # If we're waiting on a target choice, let the user answer via chat.
        if record.status == ExperimentStatus.awaiting_human and record.pending_human:
            chosen = self._match_option(message, record.pending_human.options)
            if chosen or self._is_affirmation(message):
                await self.submit_human_response(
                    experiment_id,
                    HumanResponseRequest(target_column=chosen, approve=True, message=message),
                )
                target = record.target_column
                return self._reply(
                    record,
                    f"Great — using '{target}' as the target column. Resuming the pipeline now.",
                    intent="human_response", actions=["resume"],
                )
            question = record.pending_human.question
            options = ", ".join(record.pending_human.options)
            return self._reply(
                record,
                f"I still need to confirm the target column first. {question} "
                f"Available columns: {options}.",
                intent="awaiting_human",
            )

        if record.status == ExperimentStatus.running:
            return self._reply(
                record,
                "The pipeline is still running — watch the activity panel for live "
                "updates as each agent finishes.",
                intent="running",
            )
        if record.status == ExperimentStatus.failed:
            return self._reply(
                record, f"This experiment failed: {record.error or 'unknown error'}.",
                intent="error",
            )

        return self._reply(record, self._grounded_reply(record, message), intent="chat")

    def _reply(
        self, record: ExperimentRecord, text: str, *, intent: str = "chat",
        actions: list[str] | None = None,
    ) -> ChatResponse:
        self._bus.emit_event(
            record.id, EventType.chat_message, agent="assistant", message=text,
            data={"role": "assistant", "intent": intent},
        )
        return ChatResponse(reply=text, intent=intent, actions=actions or [])

    def _grounded_reply(self, record: ExperimentRecord, message: str) -> str:
        system = (
            "You are the assistant of an AutoML system. Answer the user's question "
            "about their experiment using ONLY the facts provided below. Never invent "
            "metrics, numbers, or model names. If the answer is not in the facts, say "
            "you don't have that information yet. Be concise (2-4 sentences). No markdown headers."
        )
        reply = text_invoke(system, f"FACTS:\n{self._chat_context(record)}\n\nUSER QUESTION:\n{message}")
        return reply.strip() if reply else self._fallback_reply(record)

    def _chat_context(self, record: ExperimentRecord) -> str:
        state = record.final_state
        if not state:
            return (
                f"Experiment status: {record.status.value}. "
                f"User objective: {record.user_query}. No results computed yet."
            )
        best = state.get("best_model") or {}
        review = state.get("model_review") or {}
        evaluation = state.get("evaluation_results") or {}
        lines = [
            f"Objective: {record.user_query}",
            f"Problem type: {state.get('problem_type')}",
            f"Target column: {state.get('target_column')}",
            f"Primary metric: {state.get('metric')}",
            f"Best model: {best.get('model_name')} "
            f"({best.get('metric')}={best.get('score')})",
            f"Best model test metrics: {best.get('metrics')}",
        ]
        leaderboard = evaluation.get("leaderboard") or []
        if leaderboard:
            lines.append(
                "Leaderboard: "
                + "; ".join(
                    f"{e.get('rank')}. {e.get('model_name')} {e.get('primary_score')}"
                    for e in leaderboard
                )
            )
        if review:
            lines.append(f"Review recommendation: {review.get('recommendation')}")
            if review.get("warnings"):
                lines.append(f"Warnings: {review.get('warnings')}")
            if review.get("limitations"):
                lines.append(f"Limitations: {review.get('limitations')}")
        meta = state.get("training_meta") or {}
        if meta:
            lines.append(f"Training split: {meta}")
        return "\n".join(str(line) for line in lines)

    def _fallback_reply(self, record: ExperimentRecord) -> str:
        if record.status == ExperimentStatus.completed and record.final_state:
            best = record.final_state.get("best_model") or {}
            review = record.final_state.get("model_review") or {}
            return (
                f"The best model is {best.get('model_name')} with "
                f"{best.get('metric')}={best.get('score')}. "
                f"{review.get('recommendation', '')}"
            ).strip()
        return (
            f"This experiment is currently '{record.status.value}'. "
            "Ask me again once it finishes for details on the results."
        )

    @staticmethod
    def _match_option(message: str, options: list[str]) -> str | None:
        """Find a column name referenced in a free-text message."""
        text = message.lower()
        tokens = set(re.findall(r"[a-z0-9_]+", text))
        for opt in options:  # exact token match wins
            if opt.lower() in tokens:
                return opt
        for opt in sorted(options, key=len, reverse=True):  # bounded phrase match
            if re.search(rf"(?<![a-z0-9_]){re.escape(opt.lower())}(?![a-z0-9_])", text):
                return opt
        return None

    @staticmethod
    def _is_affirmation(message: str) -> bool:
        text = message.strip().lower()
        return any(word in text for word in _AFFIRMATIONS)

    # ---- read-only views -------------------------------------------------
    def snapshot(self, experiment_id: str) -> ExperimentSnapshot:
        record = self._require(experiment_id)
        has_results = (
            record.status == ExperimentStatus.completed
            and bool((record.final_state or {}).get("best_model"))
        )
        return ExperimentSnapshot(
            id=record.id,
            status=record.status,
            user_query=record.user_query,
            dataset_id=record.dataset_id,
            target_column=record.target_column,
            problem_type=record.problem_type,
            metric=record.metric,
            created_at=record.created_at,
            updated_at=record.updated_at,
            error=record.error,
            has_results=has_results,
            pending_human=record.pending_human,
            plan=record.plan,
        )

    def results(self, experiment_id: str) -> ResultsResponse:
        record = self._require(experiment_id)
        state = record.final_state or {}
        evaluation = state.get("evaluation_results")
        review = state.get("model_review")
        return ResultsResponse(
            experiment_id=record.id,
            problem_type=record.problem_type,
            target_column=record.target_column,
            metric=record.metric,
            evaluation=EvaluationReport.model_validate(evaluation) if evaluation else None,
            review=ModelReview.model_validate(review) if review else None,
            best_model=state.get("best_model"),
            artifacts=state.get("artifacts") or {},
        )

    def artifact_path(self, experiment_id: str, kind: str) -> Path:
        """Return the on-disk path for a downloadable artifact ('model' | 'report')."""
        record = self._require(experiment_id)
        artifacts = (record.final_state or {}).get("artifacts") or {}
        key = {"model": "best_model", "report": "report"}.get(kind)
        if key is None:
            raise ValueError(f"Unknown artifact kind: {kind}")
        raw = artifacts.get(key)
        if not raw:
            raise FileNotFoundError(f"No {kind} artifact available for experiment {experiment_id}.")
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"Artifact file missing on disk: {path}")
        return path

    # ---- streaming -------------------------------------------------------
    def event_stream(self, experiment_id: str) -> AsyncIterator[AgentEvent]:
        """Async iterator of events for SSE (replays history, then live)."""
        return self._bus.subscribe(experiment_id)


# Process-wide singleton (lazy so settings/env are loaded first).
_MANAGER: ExperimentManager | None = None


def get_experiment_manager() -> ExperimentManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = ExperimentManager()
    return _MANAGER
