"""Deterministic training service.

This is the "execution" half of the system — the LLM never runs any of this. It
performs a reproducible train/test split, cross-validation, holdout evaluation
and timing for each candidate model, with per-model error handling so one broken
model never sinks the whole experiment.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, train_test_split

from ..logging_config import get_logger
from ..schemas.common import ProblemType
from ..schemas.evaluation import ModelResult
from ..schemas.plan import PreprocessingConfig
from . import metrics as M
from .preprocessing import build_pipeline
from .registry import build_estimator, get_model_spec

logger = get_logger(__name__)

# A callback used to stream progress events out of the (blocking) training loop.
ProgressCb = Callable[[str, dict], None]


@dataclass
class SplitInfo:
    n_train: int = 0
    n_test: int = 0
    cv_folds: int = 0
    n_classes: int = 0
    stratified: bool = False
    subsampled_from: int | None = None


@dataclass
class TrainingOutcome:
    results: list[ModelResult] = field(default_factory=list)
    split: SplitInfo = field(default_factory=SplitInfo)


class TrainingService:
    def __init__(self, random_state: int = 42, test_size: float = 0.2, cv_folds: int = 5,
                 max_training_rows: int = 50_000):
        self.random_state = random_state
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.max_training_rows = max_training_rows

    # ---- data preparation ------------------------------------------------
    def prepare_xy(
        self, df: pd.DataFrame, target: str, problem_type: ProblemType | str
    ) -> tuple[pd.DataFrame, pd.Series, int | None]:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset.")
        pt = problem_type.value if isinstance(problem_type, ProblemType) else problem_type

        data = df.copy()
        # Drop rows with a missing target — we cannot learn or score those.
        data = data[data[target].notna()]

        if pt == "regression":
            data[target] = pd.to_numeric(data[target], errors="coerce")
            data = data[data[target].notna()]

        if data.empty:
            raise ValueError("No usable rows remain after dropping missing target values.")

        subsampled_from = None
        if len(data) > self.max_training_rows:
            subsampled_from = len(data)
            data = data.sample(n=self.max_training_rows, random_state=self.random_state)

        y = data[target]
        X = data.drop(columns=[target])
        return X, y, subsampled_from

    def _safe_cv_folds(self, y: pd.Series, problem_type: str) -> tuple[int, int]:
        n_classes = int(y.nunique())
        if problem_type == "classification":
            min_count = int(y.value_counts().min())
            if min_count < 2:
                return 0, n_classes
            return max(2, min(self.cv_folds, min_count)), n_classes
        if len(y) < 4:
            return 0, n_classes
        return max(2, min(self.cv_folds, len(y))), n_classes

    # ---- main entry point -----------------------------------------------
    def train_candidates(
        self,
        df: pd.DataFrame,
        target: str,
        problem_type: ProblemType | str,
        metric: str,
        candidate_keys: list[str],
        config: PreprocessingConfig,
        on_model: ProgressCb | None = None,
    ) -> TrainingOutcome:
        pt = problem_type.value if isinstance(problem_type, ProblemType) else problem_type
        metric = M.normalize_metric(metric, pt)
        X, y, subsampled_from = self.prepare_xy(df, target, pt)

        cv_folds, n_classes = self._safe_cv_folds(y, pt)

        # Stratify classification splits when every class has at least 2 members.
        stratify = None
        if pt == "classification" and int(y.value_counts().min()) >= 2:
            stratify = y
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify
        )

        split = SplitInfo(
            n_train=len(X_train),
            n_test=len(X_test),
            cv_folds=cv_folds,
            n_classes=n_classes,
            stratified=stratify is not None,
            subsampled_from=subsampled_from,
        )

        results: list[ModelResult] = []
        for key in candidate_keys:
            spec = get_model_spec(pt, key)
            name = spec.name if spec else key
            if on_model:
                on_model("model_training", {"model_key": key, "model_name": name, "status": "running"})
            result = self._train_one(
                key, name, pt, metric, config, X_train, X_test, y_train, y_test, cv_folds, n_classes
            )
            results.append(result)
            if on_model:
                on_model(
                    "model_completed",
                    {
                        "model_key": key,
                        "model_name": name,
                        "status": result.status,
                        "primary_metric": result.primary_metric,
                        "primary_score": result.primary_score,
                        "training_time_sec": result.training_time_sec,
                        "error": result.error,
                    },
                )
        return TrainingOutcome(results=results, split=split)

    def _train_one(
        self, key, name, pt, metric, config, X_train, X_test, y_train, y_test, cv_folds, n_classes
    ) -> ModelResult:
        try:
            estimator = build_estimator(pt, key, self.random_state)
            pipeline = build_pipeline(config, estimator)

            start = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline.fit(X_train, y_train)
            elapsed = time.perf_counter() - start

            # Holdout (test) metrics.
            y_pred = pipeline.predict(X_test)
            y_train_pred = pipeline.predict(X_train)
            if pt == "classification":
                proba = None
                if hasattr(pipeline, "predict_proba"):
                    try:
                        proba = pipeline.predict_proba(X_test)
                    except (AttributeError, ValueError):
                        proba = None
                labels = list(getattr(pipeline, "classes_", []))
                test_metrics = M.compute_classification_metrics(
                    np.asarray(y_test), np.asarray(y_pred), proba, labels or None
                )
                train_metrics = M.compute_classification_metrics(
                    np.asarray(y_train), np.asarray(y_train_pred)
                )
            else:
                test_metrics = M.compute_regression_metrics(np.asarray(y_test), np.asarray(y_pred))
                train_metrics = M.compute_regression_metrics(np.asarray(y_train), np.asarray(y_train_pred))

            cv_mean, cv_std = self._cross_validate(pipeline, X_train, y_train, pt, cv_folds, n_classes)

            primary_score = test_metrics.get(metric)
            return ModelResult(
                model_key=key,
                model_name=name,
                status="success",
                metrics=test_metrics,
                train_metrics=train_metrics,
                cv_mean=cv_mean,
                cv_std=cv_std,
                training_time_sec=round(elapsed, 4),
                primary_metric=metric,
                primary_score=primary_score,
            )
        except Exception as exc:  # noqa: BLE001 — we intentionally isolate model failures
            logger.warning("Model '%s' failed: %s", key, exc)
            return ModelResult(
                model_key=key,
                model_name=name,
                status="failed",
                primary_metric=metric,
                error=str(exc),
            )

    def _cross_validate(self, pipeline, X_train, y_train, pt, cv_folds, n_classes):
        if cv_folds < 2:
            return {}, {}
        scoring = M.cv_scoring_for(pt, n_classes)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv = cross_validate(
                    pipeline, X_train, y_train, scoring=scoring, cv=cv_folds,
                    n_jobs=-1, error_score=np.nan,
                )
        except Exception as exc:  # noqa: BLE001
            logger.info("Cross-validation skipped: %s", exc)
            return {}, {}

        cv_mean: dict[str, float] = {}
        cv_std: dict[str, float] = {}
        for canonical in scoring:
            arr = cv.get(f"test_{canonical}")
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=float)
            if np.all(np.isnan(arr)):
                continue
            mean = float(np.nanmean(arr))
            std = float(np.nanstd(arr))
            # neg_* scorers come back negative — flip for lower-is-better metrics.
            if canonical in M.LOWER_IS_BETTER:
                mean = -mean
            cv_mean[canonical] = round(mean, 6)
            cv_std[canonical] = round(std, 6)
        return cv_mean, cv_std

    # ---- final refit for the artifact -----------------------------------
    def fit_final(self, df, target, problem_type, config, model_key):
        """Refit the chosen model on ALL available data for the saved artifact."""
        pt = problem_type.value if isinstance(problem_type, ProblemType) else problem_type
        X, y, _ = self.prepare_xy(df, target, pt)
        estimator = build_estimator(pt, model_key, self.random_state)
        pipeline = build_pipeline(config, estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X, y)
        return pipeline
