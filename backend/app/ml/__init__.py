"""Deterministic machine-learning execution layer."""
from .metrics import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    metric_is_higher_better,
    normalize_metric,
)
from .preprocessing import build_pipeline, build_preprocessing_config
from .registry import (
    available_model_keys,
    build_estimator,
    default_candidates,
    get_model_spec,
    sanitize_candidates,
)
from .training import TrainingService, TrainingOutcome

__all__ = [
    "CLASSIFICATION_METRICS",
    "REGRESSION_METRICS",
    "metric_is_higher_better",
    "normalize_metric",
    "build_pipeline",
    "build_preprocessing_config",
    "available_model_keys",
    "build_estimator",
    "default_candidates",
    "get_model_spec",
    "sanitize_candidates",
    "TrainingService",
    "TrainingOutcome",
]
