"""EDA agent.

Python computes every statistic (via :mod:`app.tools.eda_tools`); the LLM only
writes a human-readable interpretation on top of those facts. A deterministic
interpretation is generated when no LLM is configured.
"""
from __future__ import annotations

from ..schemas.common import EventType, ProblemType
from ..schemas.eda import EdaReport
from ..services.event_bus import get_event_bus
from ..services.storage import load_dataframe
from ..tools.eda_tools import compute_eda
from .base import ai_message, graph_node
from .llm import text_invoke

AGENT = "eda"

_SYSTEM = """You are the EDA agent of an AutoML system. You are given \
already-computed statistics about a dataset. Write a concise (4-7 sentence) \
interpretation for a data scientist: highlight data quality issues, notable \
distributions/correlations, class balance, and anything that affects modelling. \
Do NOT invent numbers — only use the statistics provided. Do not use markdown \
headers."""


def _facts(report: EdaReport, target: str | None) -> str:
    corr = ", ".join(
        f"{c.feature_a}~{c.feature_b}={c.correlation}" for c in report.top_correlations[:6]
    )
    missing = ", ".join(f"{m.column}({m.missing_pct}%)" for m in report.missing_report[:6])
    return (
        f"Target: {target}\n"
        f"Summary: {report.summary}\n"
        f"Class balance ratio: {report.class_balance_ratio}\n"
        f"Target distribution: {report.target_distribution}\n"
        f"Top correlations: {corr or 'n/a'}\n"
        f"Missing (top): {missing or 'none'}\n"
        f"Warnings: {report.warnings}\n"
    )


def _heuristic_interpretation(report: EdaReport, target: str | None) -> str:
    parts = [
        f"The dataset has {report.summary.get('n_rows')} rows and "
        f"{report.summary.get('n_columns')} columns "
        f"({report.summary.get('n_numeric')} numeric, "
        f"{report.summary.get('n_categorical')} categorical)."
    ]
    if report.class_balance_ratio is not None:
        balance = "well balanced" if report.class_balance_ratio >= 0.5 else "imbalanced"
        parts.append(f"The target '{target}' is {balance} (ratio {report.class_balance_ratio}).")
    if report.top_correlations:
        top = report.top_correlations[0]
        parts.append(
            f"The strongest correlation is between {top.feature_a} and "
            f"{top.feature_b} ({top.correlation})."
        )
    if report.missing_report:
        parts.append(f"{len(report.missing_report)} column(s) contain missing values.")
    else:
        parts.append("No missing values were detected.")
    if report.warnings:
        parts.append("Warnings: " + " ".join(report.warnings))
    return " ".join(parts)


@graph_node(AGENT)
def eda_node(state: dict) -> dict:
    bus = get_event_bus()
    experiment_id = state.get("experiment_id")
    target = state.get("target_column")
    problem_type = state.get("problem_type") or ProblemType.classification.value

    bus.emit_event(experiment_id, EventType.eda_started, agent=AGENT, status="running",
                   message="Running exploratory data analysis…")

    df = load_dataframe(state["dataset_path"])
    report = compute_eda(df, target, problem_type)

    interpretation = text_invoke(_SYSTEM, _facts(report, target))
    if not interpretation:
        interpretation = _heuristic_interpretation(report, target)
    report.interpretation = interpretation.strip()

    bus.emit_event(
        experiment_id, EventType.eda_completed, agent=AGENT, status="completed",
        message="EDA completed.",
        data={
            "interpretation": report.interpretation,
            "warnings": report.warnings,
            "summary": report.summary,
            "top_correlations": [c.model_dump() for c in report.top_correlations[:8]],
            "target_distribution": report.target_distribution,
            "missing_report": [m.model_dump() for m in report.missing_report[:12]],
        },
    )
    return {"eda_report": report.model_dump(mode="json"), "messages": [ai_message(f"📊 {report.interpretation}")]}
