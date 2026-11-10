"""Model reviewer node.

Produces a recommendation, warnings and limitations. The reviewer may only use
metrics already computed by Python — it never invents numbers. Overfitting,
imbalance and small-data checks are computed deterministically and passed in;
the LLM turns them into prose (or a heuristic does, offline)."""
from __future__ import annotations

from ..ml.metrics import metric_is_higher_better
from ..schemas.common import EventType
from ..schemas.evaluation import ModelReview
from ..services.event_bus import get_event_bus
from .base import ai_message, graph_node
from .llm import structured_invoke

AGENT = "model_reviewer"

_SYSTEM = """You are the model-review agent of an AutoML system. You are given \
the best model's metrics, dataset characteristics, and training configuration, \
plus a list of automatically-detected issues. Produce a JSON review with: a \
short recommendation, an explanation, a list of warnings, and a list of \
limitations. CRITICAL: only reference the metrics and facts provided — never \
invent numbers. Be honest about weaknesses."""


def _detect_issues(state: dict) -> list[str]:
    """Deterministic checks so the reviewer never has to guess."""
    issues: list[str] = []
    best = state.get("best_model") or {}
    metric = state.get("metric", "")
    meta = state.get("training_meta") or {}
    info = state.get("dataset_info") or {}
    eda = state.get("eda_report") or {}

    # Overfitting: gap between train and test on the primary metric.
    train_score = (best.get("train_metrics") or {}).get(metric)
    test_score = (best.get("metrics") or {}).get(metric)
    if isinstance(train_score, (int, float)) and isinstance(test_score, (int, float)):
        gap = (train_score - test_score) if metric_is_higher_better(metric) else (test_score - train_score)
        if gap > 0.15:
            issues.append(
                f"Possible overfitting: train {metric}={train_score} vs test {metric}={test_score}."
            )

    n_rows = info.get("n_rows", 0)
    if n_rows and n_rows < 200:
        issues.append(f"Small dataset ({n_rows} rows) — results may not generalise.")
    if meta.get("cv_folds", 0) < 2:
        issues.append("Cross-validation was skipped (too few samples per class).")
    if meta.get("subsampled_from"):
        issues.append(f"Trained on a {meta.get('n_train')+meta.get('n_test')}-row subsample of {meta['subsampled_from']} rows.")
    balance = eda.get("class_balance_ratio")
    if isinstance(balance, (int, float)) and balance < 0.2:
        issues.append(f"Class imbalance (ratio {balance}) — prefer F1/recall over accuracy.")
    for w in eda.get("warnings", [])[:3]:
        issues.append(w)
    return issues


def _facts(state: dict, issues: list[str]) -> str:
    best = state.get("best_model") or {}
    meta = state.get("training_meta") or {}
    info = state.get("dataset_info") or {}
    return (
        f"Problem type: {state.get('problem_type')}\n"
        f"Best model: {best.get('model_name')} (key={best.get('model_key')})\n"
        f"Primary metric: {best.get('metric')} = {best.get('score')}\n"
        f"Test metrics: {best.get('metrics')}\n"
        f"Train metrics: {best.get('train_metrics')}\n"
        f"CV mean: {best.get('cv_mean')}\n"
        f"Dataset: {info.get('n_rows')} rows, {info.get('n_columns')} cols, "
        f"missing={info.get('total_missing')}\n"
        f"Training split: {meta}\n"
        f"Detected issues: {issues}\n"
    )


def _heuristic_review(state: dict, issues: list[str]) -> ModelReview:
    best = state.get("best_model") or {}
    metric = best.get("metric", "")
    score = best.get("score")
    higher = metric_is_higher_better(metric)
    if isinstance(score, (int, float)):
        if (higher and score >= 0.85) or (not higher and score <= 0.15):
            rec = "Strong performer — suitable to deploy with routine monitoring."
        elif (higher and score >= 0.7) or (not higher):
            rec = "Reasonable baseline — validate on fresh data before production use."
        else:
            rec = "Weak performance — collect more data or engineer features before relying on it."
    else:
        rec = "Review the metrics before use."
    explanation = (
        f"The best model is {best.get('model_name')} with {metric}={score}. "
        "This review is based solely on the computed metrics above."
    )
    limitations = [
        "No hyperparameter tuning was performed (models use sensible defaults).",
        "Datetime and free-text/identifier columns were dropped in this MVP.",
    ]
    return ModelReview(recommendation=rec, explanation=explanation, warnings=issues, limitations=limitations)


@graph_node(AGENT)
def model_reviewer_node(state: dict) -> dict:
    bus = get_event_bus()
    experiment_id = state.get("experiment_id")
    bus.emit_event(experiment_id, EventType.reviewer_started, agent=AGENT, status="running",
                   message="Reviewing the best model…")

    issues = _detect_issues(state)
    review = structured_invoke(ModelReview, _SYSTEM, _facts(state, issues))
    if review is None:
        review = _heuristic_review(state, issues)
    elif not review.warnings:
        # The LLM must not drop the deterministically-detected issues.
        review.warnings = issues

    bus.emit_event(
        experiment_id, EventType.reviewer_completed, agent=AGENT, status="completed",
        message=review.recommendation, data={"review": review.model_dump(mode="json")},
    )
    return {"model_review": review.model_dump(mode="json"), "messages": [ai_message(f"🧾 {review.recommendation}")]}
