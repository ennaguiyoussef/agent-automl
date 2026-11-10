"""Tests for leakage-safe preprocessing construction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.preprocessing import (
    build_column_transformer,
    build_pipeline,
    build_preprocessing_config,
)
from app.tools.dataset_tools import profile_dataframe


def _messy_df(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "row_id": range(n),  # identifier-like integer (all unique)
            "const": ["c"] * n,  # constant
            "signup": pd.to_datetime("2020-01-01") + pd.to_timedelta(range(n), unit="D"),
            "note": [f"free text {i}" for i in range(n)],  # text / high-uniqueness
            "amount": rng.normal(100, 20, n).round(2),  # numeric feature
            "tier": rng.choice(["a", "b", "c"], n),  # categorical feature
            "target": rng.integers(0, 2, n),
        }
    )


def test_config_drops_non_features():
    df = _messy_df()
    info = profile_dataframe(df)
    config = build_preprocessing_config(info, target="target")

    assert "row_id" in config.dropped_features  # identifier
    assert "const" in config.dropped_features  # constant
    assert "signup" in config.dropped_features  # datetime
    assert "note" in config.dropped_features  # free text
    assert "amount" in config.numeric_features
    assert "tier" in config.categorical_features
    assert "target" not in (config.numeric_features + config.categorical_features)


def test_column_transformer_shape_and_unknown_handling():
    df = _messy_df()
    info = profile_dataframe(df)
    config = build_preprocessing_config(info, target="target")

    ct = build_column_transformer(config)
    X = df.drop(columns=["target"])
    transformed = ct.fit_transform(X)
    assert transformed.shape[0] == len(df)
    assert transformed.shape[1] >= 1

    # A category unseen at fit time must not crash transform (handle_unknown).
    X_new = X.iloc[:5].copy()
    X_new["tier"] = "totally_new_category"
    out = ct.transform(X_new)
    assert out.shape[0] == 5


def test_pipeline_fits_and_predicts():
    from sklearn.linear_model import LogisticRegression

    df = _messy_df()
    info = profile_dataframe(df)
    config = build_preprocessing_config(info, target="target")

    pipe = build_pipeline(config, LogisticRegression(max_iter=500))
    X, y = df.drop(columns=["target"]), df["target"]
    pipe.fit(X, y)
    preds = pipe.predict(X.iloc[:10])
    assert len(preds) == 10
    assert set(pipe.named_steps) == {"preprocess", "model"}


def test_transformer_raises_when_no_features():
    from app.schemas.plan import PreprocessingConfig

    empty = PreprocessingConfig()  # no numeric, no categorical
    with pytest.raises(ValueError):
        build_column_transformer(empty)
