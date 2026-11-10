"""Evaluation node.

Python ranks the models deterministically by the chosen metric (respecting
higher-is-better vs lower-is-better). The LLM then explains *why* the winner won,
using only the computed numbers.
"""
from __future__ import annotations

from ..ml.metrics import metric_is_higher_better
from ..schemas.common import EventType, ProblemType
from ..schemas.evaluation import EvaluationReport, LeaderboardEntry, ModelResult
from ..services.event_bus import get_event_bus
from .base import ai_message, graph_node
from .llm import text_invoke

AGENT = "evaluation"

_SYSTEM = """You are the evaluation agent of an AutoML system. You are given a \
leaderboard of models with their metrics. In 3-5 sentences explain which model \
is best for the chosen metric and why, comparing it to the runners-up. Only use \
the numbers provided; never invent metrics. No markdown headers."""


def _leaderboard_text(report: EvaluationReport, results: list[ModelResult]) -> str:
    lines = [f"Chosen metric: {report.metric} (higher is better: {metric_is_higher_better(report.metric)})"]
    for r in results:
        if r.status != "success":
            lines.append(f"- {r.model_name}: FAILED ({r.error})")
        else:
            lines.append(
                f"- {r.model_name}: {report.metric}={r.primary_score}, "
                f"all={r.metrics}, cv_mean={r.cv_mean}, time={r.training_time_sec}s"
            )
    return "\n".join(lines)


@graph_node(AGENT)
def evaluation_node(state: dict) -> dict:
    bus = get_event_bus()
    experiment_id = state.get("experiment_id")
    metric = state["metric"]
    problem_type = state["problem_type"]

    results = [ModelResult.model_validate(d) for d in state.get("trained_models", [])]
    successful = [r for r in results if r.status == "success" and r.primary_score is not None]

    if not successful:
        bus.emit_event(experiment_id, EventType.error, agent=AGENT, status="failed",
                       message="All candidate models failed to train.")
        return {"error": "evaluation: all candidate models failed to train."}

    higher = metric_is_higher_better(metric)
    ranked = sorted(successful, key=lambda r: r.primary_score, reverse=higher)
    best = ranked[0]

    leaderboard = [
        LeaderboardEntry(
            model_key=r.model_key, model_name=r.model_name,
            primary_score=r.primary_score, training_time_sec=r.training_time_sec, rank=i + 1,
        )
        for i, r in enumerate(ranked)
    ]

    report = EvaluationReport(
        problem_type=ProblemType(problem_type),
        metric=metric,
        best_model_key=best.model_key,
        best_model_name=best.model_name,
        best_score=best.primary_score,
        results=results,
        leaderboard=leaderboard,
    )

    explanation = text_invoke(_SYSTEM, _leaderboard_text(report, results))
    if not explanation:
        runner = ranked[1] if len(ranked) > 1 else None
        explanation = (
            f"{best.model_name} achieved the best {metric} of {best.primary_score} "
            + (
                f"vs {runner.model_name} at {runner.primary_score}. "
                if runner else "among the trained models. "
            )
            + "It was selected as the best model."
        )
    report.explanation = explanation.strip()

    best_model = {
        "model_key": best.model_key,
        "model_name": best.model_name,
        "metric": metric,
        "score": best.primary_score,
        "metrics": best.metrics,
        "train_metrics": best.train_metrics,
        "cv_mean": best.cv_mean,
        "cv_std": best.cv_std,
        "training_time_sec": best.training_time_sec,
    }

    bus.emit_event(
        experiment_id, EventType.evaluation_completed, agent=AGENT, status="completed",
        message=f"Best model: {best.model_name} ({metric}={best.primary_score}).",
        data={
            "best_model": best_model,
            "leaderboard": [e.model_dump() for e in leaderboard],
            "explanation": report.explanation,
        },
    )
    return {
        "evaluation_results": report.model_dump(mode="json"),
        "best_model": best_model,
        "messages": [ai_message(f"🏆 {report.explanation}")],
    }
