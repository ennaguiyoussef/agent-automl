"""Model registry.

Maps stable string keys (used by the planner LLM and the API) to scikit-learn
estimator factories. The planner may only *choose* keys from here; it can never
instantiate arbitrary code. Two separate tables keep classification and
regression estimators apart even when they share a key (``random_forest`` etc.).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.svm import SVC

from ..schemas.common import ProblemType


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    factory: Callable[[int], object]
    supports_proba: bool = True


CLASSIFICATION_MODELS: dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec(
        "logistic_regression",
        "Logistic Regression",
        lambda rs: LogisticRegression(max_iter=2000, random_state=rs),
    ),
    "random_forest": ModelSpec(
        "random_forest",
        "Random Forest",
        lambda rs: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=rs),
    ),
    "gradient_boosting": ModelSpec(
        "gradient_boosting",
        "Gradient Boosting",
        lambda rs: GradientBoostingClassifier(random_state=rs),
    ),
    "hist_gradient_boosting": ModelSpec(
        "hist_gradient_boosting",
        "Hist Gradient Boosting",
        lambda rs: HistGradientBoostingClassifier(random_state=rs),
    ),
    "svc": ModelSpec(
        "svc",
        "Support Vector Classifier",
        lambda rs: SVC(probability=True, random_state=rs),
    ),
}

REGRESSION_MODELS: dict[str, ModelSpec] = {
    "linear_regression": ModelSpec(
        "linear_regression",
        "Linear Regression",
        lambda rs: LinearRegression(),
        supports_proba=False,
    ),
    "ridge": ModelSpec(
        "ridge",
        "Ridge Regression",
        lambda rs: Ridge(random_state=rs),
        supports_proba=False,
    ),
    "random_forest": ModelSpec(
        "random_forest",
        "Random Forest",
        lambda rs: RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=rs),
        supports_proba=False,
    ),
    "gradient_boosting": ModelSpec(
        "gradient_boosting",
        "Gradient Boosting",
        lambda rs: GradientBoostingRegressor(random_state=rs),
        supports_proba=False,
    ),
    "hist_gradient_boosting": ModelSpec(
        "hist_gradient_boosting",
        "Hist Gradient Boosting",
        lambda rs: HistGradientBoostingRegressor(random_state=rs),
        supports_proba=False,
    ),
}

# Default candidate sets the planner falls back to if the LLM is unavailable.
DEFAULT_CANDIDATES: dict[str, list[str]] = {
    "classification": ["logistic_regression", "random_forest", "hist_gradient_boosting"],
    "regression": ["ridge", "random_forest", "hist_gradient_boosting"],
}


def _table(problem_type: ProblemType | str) -> dict[str, ModelSpec]:
    pt = problem_type.value if isinstance(problem_type, ProblemType) else problem_type
    return CLASSIFICATION_MODELS if pt == "classification" else REGRESSION_MODELS


def available_model_keys(problem_type: ProblemType | str) -> list[str]:
    return list(_table(problem_type).keys())


def get_model_spec(problem_type: ProblemType | str, key: str) -> ModelSpec | None:
    return _table(problem_type).get(key)


def build_estimator(problem_type: ProblemType | str, key: str, random_state: int):
    spec = get_model_spec(problem_type, key)
    if spec is None:
        raise KeyError(f"Unknown model '{key}' for problem type '{problem_type}'.")
    return spec.factory(random_state)


def sanitize_candidates(problem_type: ProblemType | str, keys: list[str]) -> list[str]:
    """Keep only valid, de-duplicated keys; fall back to defaults if none remain."""
    table = _table(problem_type)
    seen: set[str] = set()
    valid: list[str] = []
    for k in keys or []:
        k = (k or "").strip().lower()
        if k in table and k not in seen:
            seen.add(k)
            valid.append(k)
    if not valid:
        pt = problem_type.value if isinstance(problem_type, ProblemType) else problem_type
        return list(DEFAULT_CANDIDATES.get(pt, list(table.keys())[:3]))
    return valid


def default_candidates(problem_type: ProblemType | str) -> list[str]:
    pt = problem_type.value if isinstance(problem_type, ProblemType) else problem_type
    return list(DEFAULT_CANDIDATES.get(pt, []))
