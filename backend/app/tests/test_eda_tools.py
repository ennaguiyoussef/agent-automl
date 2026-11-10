"""Tests for the deterministic EDA tools."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.tools.eda_tools import compute_eda


def test_eda_classification_report(clf_df):
    report = compute_eda(clf_df, target="churn", problem_type="classification")

    assert report.summary["n_rows"] == len(clf_df)
    assert report.summary["problem_type"] == "classification"
    # numeric columns (age, income) get numeric stats
    names = {s.name for s in report.numeric_stats}
    assert {"age", "income"}.issubset(names)
    # class balance is a ratio in (0, 1]
    assert report.class_balance_ratio is not None
    assert 0.0 < report.class_balance_ratio <= 1.0
    assert report.target_distribution["kind"] == "categorical"


def test_eda_flags_class_imbalance():
    n = 200
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "x": rng.normal(0, 1, n),
            # ~5% positives -> ratio well under 0.2
            "y": (rng.random(n) < 0.05).astype(int),
        }
    )
    report = compute_eda(df, target="y", problem_type="classification")
    assert report.class_balance_ratio < 0.2
    assert any("imbalance" in w.lower() for w in report.warnings)


def test_eda_regression_report(reg_df):
    report = compute_eda(reg_df, target="price", problem_type="regression")
    assert report.summary["problem_type"] == "regression"
    # continuous target -> continuous distribution, no class balance
    assert report.target_distribution["kind"] == "continuous"
    assert report.class_balance_ratio is None
    # size and price should be strongly correlated
    assert report.top_correlations
    assert abs(report.top_correlations[0].correlation) > 0.3
