"""Metric computation and metric-name normalisation.

Kept separate from training so the metric logic is unit-testable and the ranking
direction (higher-is-better vs lower-is-better) lives in exactly one place.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

from ..schemas.common import ProblemType

# Canonical metric names.
CLASSIFICATION_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
REGRESSION_METRICS = ["mae", "mse", "rmse", "r2"]

# For these, a *lower* value is better (so ranking must invert).
LOWER_IS_BETTER = {"mae", "mse", "rmse"}

_METRIC_ALIASES = {
    "f1_score": "f1",
    "f1-score": "f1",
    "fscore": "f1",
    "auc": "roc_auc",
    "rocauc": "roc_auc",
    "roc-auc": "roc_auc",
    "roc_auc_score": "roc_auc",
    "acc": "accuracy",
    "prec": "precision",
    "r^2": "r2",
    "r_squared": "r2",
    "root_mean_squared_error": "rmse",
    "mean_absolute_error": "mae",
    "mean_squared_error": "mse",
}

# CV scorer names understood by scikit-learn, keyed by canonical metric.
_CV_SCORER = {
    "accuracy": "accuracy",
    "precision": "precision_weighted",
    "recall": "recall_weighted",
    "f1": "f1_weighted",
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "mse": "neg_mean_squared_error",
    "rmse": "neg_root_mean_squared_error",
}


def metric_is_higher_better(metric: str) -> bool:
    return metric not in LOWER_IS_BETTER


def normalize_metric(name: str | None, problem_type: ProblemType | str) -> str:
    """Return a valid canonical metric for the problem type.

    Falls back to a sensible default (``f1`` / ``r2``) if the requested metric is
    missing or not applicable.
    """
    pt = problem_type.value if isinstance(problem_type, ProblemType) else problem_type
    default = "f1" if pt == "classification" else "r2"
    if not name:
        return default
    key = _METRIC_ALIASES.get(name.strip().lower(), name.strip().lower())
    valid = CLASSIFICATION_METRICS if pt == "classification" else REGRESSION_METRICS
    return key if key in valid else default


def cv_scoring_for(problem_type: ProblemType | str, n_classes: int) -> dict[str, str]:
    """Return the {canonical_metric: sklearn_scorer} map used for cross-validation."""
    pt = problem_type.value if isinstance(problem_type, ProblemType) else problem_type
    if pt == "classification":
        scoring = {m: _CV_SCORER[m] for m in ["accuracy", "precision", "recall", "f1"]}
        if n_classes == 2:
            scoring["roc_auc"] = "roc_auc"
        return scoring
    return {m: _CV_SCORER[m] for m in ["r2", "mae", "rmse"]}


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    labels: list | None = None,
) -> dict[str, float]:
    """Compute classification metrics. Uses weighted averaging so binary and
    multiclass problems are handled uniformly."""
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_proba is not None:
        try:
            n_classes = len(labels) if labels is not None else len(np.unique(y_true))
            if n_classes == 2:
                proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                metrics["roc_auc"] = float(roc_auc_score(y_true, proba_pos))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted", labels=labels)
                )
        except (ValueError, IndexError):
            # ROC-AUC is undefined for some edge cases (e.g. a single class in y_true).
            pass
    return {k: round(v, 6) for k, v in metrics.items()}


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 6),
        "mse": round(float(mean_squared_error(y_true, y_pred)), 6),
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 6),
        "r2": round(float(r2_score(y_true, y_pred)), 6),
    }
