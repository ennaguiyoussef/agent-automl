"""Deterministic dataset-profiling tools.

These are the "Python calculates" half of the dataset-analysis step. They take a
``pandas.DataFrame`` and return a fully JSON-serialisable :class:`DatasetInfo`.
No LLM is involved here — the numbers are computed by pandas so they are always
correct and reproducible.
"""
from __future__ import annotations

import pandas as pd
from pandas.api import types as pdt

from ..schemas.dataset import ColumnInfo, DatasetInfo
from ..schemas.common import InferredType
from ..utils import to_native

# A column with more unique categorical values than this is considered
# "high cardinality" (expensive / risky to one-hot encode directly).
HIGH_CARDINALITY_THRESHOLD = 40


def infer_column_type(series: pd.Series) -> InferredType:
    """Infer a modelling-oriented type for a column."""
    if pdt.is_bool_dtype(series):
        return InferredType.boolean
    if pdt.is_datetime64_any_dtype(series):
        return InferredType.datetime
    if pdt.is_numeric_dtype(series):
        return InferredType.numeric

    # object / string / category
    n = len(series)
    nunique = int(series.nunique(dropna=True))
    if n > 20 and nunique > max(50, int(0.9 * n)):
        # Almost every value distinct -> free text or an identifier.
        return InferredType.text
    return InferredType.categorical


def _column_info(series: pd.Series, n_rows: int) -> ColumnInfo:
    inferred = infer_column_type(series)
    missing = int(series.isna().sum())
    nunique = int(series.nunique(dropna=True))
    non_null = series.dropna()
    sample = [to_native(v) for v in non_null.unique()[:5].tolist()]

    is_constant = nunique <= 1
    is_high_card = (
        inferred in (InferredType.categorical, InferredType.text)
        and nunique > HIGH_CARDINALITY_THRESHOLD
    )

    return ColumnInfo(
        name=str(series.name),
        dtype=str(series.dtype),
        inferred_type=inferred,
        missing=missing,
        missing_pct=round(100 * missing / n_rows, 3) if n_rows else 0.0,
        unique=nunique,
        unique_pct=round(100 * nunique / n_rows, 3) if n_rows else 0.0,
        is_constant=is_constant,
        is_high_cardinality=is_high_card,
        sample_values=sample,
    )


def profile_dataframe(df: pd.DataFrame, head_rows: int = 10) -> DatasetInfo:
    """Compute a complete deterministic profile of a DataFrame."""
    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
    columns = [_column_info(df[c], n_rows) for c in df.columns]

    numeric_cols = [c.name for c in columns if c.inferred_type == InferredType.numeric]
    categorical_cols = [
        c.name for c in columns if c.inferred_type == InferredType.categorical
    ]
    datetime_cols = [c.name for c in columns if c.inferred_type == InferredType.datetime]
    boolean_cols = [c.name for c in columns if c.inferred_type == InferredType.boolean]
    constant_cols = [c.name for c in columns if c.is_constant]
    high_card_cols = [c.name for c in columns if c.is_high_cardinality]

    total_missing = int(df.isna().sum().sum())
    total_cells = n_rows * n_cols
    duplicate_rows = int(df.duplicated().sum())

    head = [
        {str(k): to_native(v) for k, v in row.items()}
        for row in df.head(head_rows).to_dict(orient="records")
    ]

    return DatasetInfo(
        n_rows=n_rows,
        n_columns=n_cols,
        column_names=[str(c) for c in df.columns],
        columns=columns,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        datetime_columns=datetime_cols,
        boolean_columns=boolean_cols,
        constant_columns=constant_cols,
        high_cardinality_columns=high_card_cols,
        duplicate_rows=duplicate_rows,
        total_missing=total_missing,
        missing_pct=round(100 * total_missing / total_cells, 3) if total_cells else 0.0,
        memory_usage_mb=round(df.memory_usage(deep=True).sum() / 1024 / 1024, 3),
        dtypes={str(c): str(df[c].dtype) for c in df.columns},
        head=head,
    )


def target_distribution(df: pd.DataFrame, target: str, max_classes: int = 30) -> dict:
    """Return the distribution of the target column (JSON-serialisable)."""
    if target not in df.columns:
        return {}
    series = df[target].dropna()
    if pdt.is_numeric_dtype(series) and series.nunique() > max_classes:
        desc = series.describe()
        return {
            "kind": "continuous",
            "count": int(desc.get("count", 0)),
            "mean": to_native(desc.get("mean")),
            "std": to_native(desc.get("std")),
            "min": to_native(desc.get("min")),
            "max": to_native(desc.get("max")),
        }
    counts = series.value_counts().head(max_classes)
    total = int(counts.sum()) or 1
    return {
        "kind": "categorical",
        "classes": [
            {
                "value": to_native(idx),
                "count": int(cnt),
                "pct": round(100 * int(cnt) / total, 2),
            }
            for idx, cnt in counts.items()
        ],
        "n_classes": int(series.nunique()),
    }
