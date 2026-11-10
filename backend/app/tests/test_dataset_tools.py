"""Tests for the deterministic dataset-profiling tools."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.schemas.common import InferredType
from app.tools.dataset_tools import (
    infer_column_type,
    profile_dataframe,
    target_distribution,
)


def _rich_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, n),
            "score": rng.normal(0, 1, n).round(3),
            "active": rng.choice([True, False], n),
            "plan": rng.choice(["basic", "premium", "enterprise"], n),
            "city": rng.choice([f"city_{i}" for i in range(45)], n),  # high-card categorical
            "note": [f"free text row {i}" for i in range(n)],  # all-unique -> text
            "country": ["US"] * n,  # constant
            "maybe": [np.nan if i % 2 == 0 else 1.0 for i in range(n)],  # 50% missing
        }
    )


def test_infer_column_type_variants():
    assert infer_column_type(pd.Series([True, False, True])) == InferredType.boolean
    assert infer_column_type(pd.Series([1, 2, 3, 4])) == InferredType.numeric
    assert infer_column_type(pd.Series(pd.to_datetime(["2020-01-01", "2021-06-01"]))) == InferredType.datetime
    assert infer_column_type(pd.Series(["a", "b", "a", "b"] * 10)) == InferredType.categorical
    assert infer_column_type(pd.Series([f"unique-{i}" for i in range(100)])) == InferredType.text


def test_profile_dataframe_classifies_columns():
    df = _rich_df()
    info = profile_dataframe(df)

    assert info.n_rows == len(df)
    assert info.n_columns == df.shape[1]

    assert set(["age", "score", "maybe"]).issubset(set(info.numeric_columns))
    assert "active" in info.boolean_columns
    assert "plan" in info.categorical_columns
    assert "city" in info.high_cardinality_columns
    assert "country" in info.constant_columns
    # free-text / identifier columns are neither numeric nor categorical features
    assert "note" not in info.numeric_columns
    assert "note" not in info.categorical_columns


def test_profile_dataframe_missing_and_duplicates():
    df = _rich_df(100)
    # 'maybe' is NaN for even indices -> 50 missing
    info = profile_dataframe(df)
    assert info.total_missing == 50
    maybe_col = next(c for c in info.columns if c.name == "maybe")
    assert maybe_col.missing == 50
    assert round(maybe_col.missing_pct) == 50

    # Introduce exact duplicate rows and re-profile.
    dup = pd.concat([df, df.iloc[[0, 1, 2]]], ignore_index=True)
    assert profile_dataframe(dup).duplicate_rows == 3


def test_target_distribution_categorical_and_continuous():
    df = _rich_df(120)
    cat = target_distribution(df, "plan")
    assert cat["kind"] == "categorical"
    assert cat["n_classes"] == 3
    assert abs(sum(c["pct"] for c in cat["classes"]) - 100) < 1.0

    cont = target_distribution(df, "score")
    assert cont["kind"] == "continuous"
    assert "mean" in cont and "min" in cont and "max" in cont

    assert target_distribution(df, "does_not_exist") == {}
