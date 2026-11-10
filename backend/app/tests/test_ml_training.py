"""Tests for the deterministic training service."""
from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from app.ml.preprocessing import build_preprocessing_config
from app.ml.training import TrainingService
from app.tools.dataset_tools import profile_dataframe


def _config(df: pd.DataFrame, target: str):
    return build_preprocessing_config(profile_dataframe(df), target=target)


def test_train_classification_candidates(clf_df):
    service = TrainingService(random_state=42, cv_folds=3)
    config = _config(clf_df, "churn")
    outcome = service.train_candidates(
        clf_df, "churn", "classification", "f1",
        ["logistic_regression", "random_forest"], config,
    )
    assert len(outcome.results) == 2
    assert all(r.status == "success" for r in outcome.results)
    best = max(outcome.results, key=lambda r: r.primary_score or 0)
    # Target is (mostly) separable -> a competent model should score well.
    assert best.primary_score is not None and best.primary_score > 0.7
    assert best.metrics  # test metrics computed
    assert best.cv_mean  # cross-validation ran (enough samples per class)
    assert outcome.split.n_train > 0 and outcome.split.n_test > 0


def test_train_regression_candidates(reg_df):
    service = TrainingService(random_state=42, cv_folds=3)
    config = _config(reg_df, "price")
    outcome = service.train_candidates(
        reg_df, "price", "regression", "r2",
        ["linear_regression", "random_forest"], config,
    )
    assert all(r.status == "success" for r in outcome.results)
    best = max(outcome.results, key=lambda r: r.primary_score or -1)
    assert best.primary_score is not None and best.primary_score > 0.5


def test_fit_final_returns_deployable_pipeline(clf_df):
    service = TrainingService(random_state=42)
    config = _config(clf_df, "churn")
    pipe = service.fit_final(clf_df, "churn", "classification", config, "logistic_regression")
    assert isinstance(pipe, Pipeline)
    assert set(pipe.named_steps) == {"preprocess", "model"}
    preds = pipe.predict(clf_df.drop(columns=["churn"]).head(5))
    assert len(preds) == 5


def test_cv_skipped_when_class_too_small():
    service = TrainingService(cv_folds=5)
    # One class has a single member -> CV must be skipped (0 folds).
    y = pd.Series([0, 0, 0, 0, 1])
    folds, n_classes = service._safe_cv_folds(y, "classification")
    assert folds == 0
    assert n_classes == 2


def test_one_model_failure_is_isolated(clf_df):
    service = TrainingService(cv_folds=3)
    config = _config(clf_df, "churn")
    outcome = service.train_candidates(
        clf_df, "churn", "classification", "f1",
        ["logistic_regression", "not_a_real_model"], config,
    )
    by_key = {r.model_key: r for r in outcome.results}
    assert by_key["logistic_regression"].status == "success"
    assert by_key["not_a_real_model"].status == "failed"
    assert by_key["not_a_real_model"].error
