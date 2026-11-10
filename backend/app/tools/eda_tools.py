"""Deterministic EDA tools.

Again: Python computes every number here. The EDA *agent* only adds a
human-readable interpretation on top of these facts.
"""
from __future__ import annotations

import pandas as pd
from pandas.api import types as pdt

from ..schemas.common import ProblemType
from ..schemas.eda import (
    CategoricalStat,
    CorrelationPair,
    EdaReport,
    MissingEntry,
    NumericStat,
)
from ..utils import safe_float, to_native
from .dataset_tools import target_distribution


def _numeric_stats(df: pd.DataFrame, columns: list[str]) -> list[NumericStat]:
    stats: list[NumericStat] = []
    for col in columns:
        s = df[col]
        desc = s.describe()
        stats.append(
            NumericStat(
                name=col,
                count=int(desc.get("count", 0)),
                mean=safe_float(desc.get("mean")),
                std=safe_float(desc.get("std")),
                min=safe_float(desc.get("min")),
                q25=safe_float(desc.get("25%")),
                median=safe_float(desc.get("50%")),
                q75=safe_float(desc.get("75%")),
                max=safe_float(desc.get("max")),
                missing=int(s.isna().sum()),
                skew=safe_float(s.skew()) if s.notna().sum() > 2 else None,
            )
        )
    return stats


def _categorical_stats(df: pd.DataFrame, columns: list[str]) -> list[CategoricalStat]:
    stats: list[CategoricalStat] = []
    for col in columns:
        s = df[col]
        counts = s.value_counts()
        top_values = [
            {"value": to_native(idx), "count": int(cnt)}
            for idx, cnt in counts.head(10).items()
        ]
        stats.append(
            CategoricalStat(
                name=col,
                unique=int(s.nunique(dropna=True)),
                top=to_native(counts.index[0]) if len(counts) else None,
                top_freq=int(counts.iloc[0]) if len(counts) else 0,
                missing=int(s.isna().sum()),
                top_values=top_values,
            )
        )
    return stats


def _top_correlations(
    df: pd.DataFrame, numeric_cols: list[str], limit: int = 12
) -> list[CorrelationPair]:
    if len(numeric_cols) < 2:
        return []
    corr = df[numeric_cols].corr(numeric_only=True)
    pairs: list[CorrelationPair] = []
    seen: set[frozenset[str]] = set()
    for a in corr.columns:
        for b in corr.columns:
            if a == b:
                continue
            key = frozenset((a, b))
            if key in seen:
                continue
            seen.add(key)
            val = safe_float(corr.loc[a, b])
            if val is None:
                continue
            pairs.append(CorrelationPair(feature_a=a, feature_b=b, correlation=round(val, 4)))
    pairs.sort(key=lambda p: abs(p.correlation), reverse=True)
    return pairs[:limit]


def compute_eda(
    df: pd.DataFrame,
    target: str | None,
    problem_type: ProblemType | str | None,
) -> EdaReport:
    """Compute the full deterministic EDA report (interpretation left empty)."""
    n_rows, n_cols = df.shape
    numeric_cols = [c for c in df.columns if pdt.is_numeric_dtype(df[c]) and not pdt.is_bool_dtype(df[c])]
    categorical_cols = [
        c
        for c in df.columns
        if not pdt.is_numeric_dtype(df[c]) and not pdt.is_datetime64_any_dtype(df[c])
    ]

    missing_report = [
        MissingEntry(
            column=str(c),
            missing=int(df[c].isna().sum()),
            missing_pct=round(100 * float(df[c].isna().mean()), 3),
        )
        for c in df.columns
        if int(df[c].isna().sum()) > 0
    ]
    missing_report.sort(key=lambda m: m.missing, reverse=True)

    dist = target_distribution(df, target) if target else None

    # Class balance ratio (classification only): smallest / largest class share.
    class_balance = None
    problem_str = problem_type.value if isinstance(problem_type, ProblemType) else problem_type
    if dist and dist.get("kind") == "categorical" and problem_str == "classification":
        classes = dist.get("classes", [])
        if len(classes) >= 2:
            counts = [c["count"] for c in classes]
            class_balance = round(min(counts) / max(counts), 4) if max(counts) else None

    warnings: list[str] = []
    if n_rows < 50:
        warnings.append(f"Very small dataset ({n_rows} rows) — metrics may be unreliable.")
    if class_balance is not None and class_balance < 0.2:
        warnings.append(
            f"Class imbalance detected (minority/majority ratio ≈ {class_balance}). "
            "Consider precision/recall/F1 over accuracy."
        )
    high_missing = [m.column for m in missing_report if m.missing_pct > 40]
    if high_missing:
        warnings.append(
            f"Columns with >40% missing values: {', '.join(high_missing[:8])}."
        )
    dup = int(df.duplicated().sum())
    if dup > 0:
        warnings.append(f"{dup} duplicate rows found.")

    summary = {
        "n_rows": int(n_rows),
        "n_columns": int(n_cols),
        "n_numeric": len(numeric_cols),
        "n_categorical": len(categorical_cols),
        "n_missing_columns": len(missing_report),
        "duplicate_rows": dup,
        "target": target,
        "problem_type": problem_str,
    }

    return EdaReport(
        summary=summary,
        numeric_stats=_numeric_stats(df, numeric_cols),
        categorical_stats=_categorical_stats(df, [c for c in categorical_cols if c != target][:50]),
        missing_report=missing_report,
        top_correlations=_top_correlations(df, numeric_cols),
        target_distribution=dist,
        class_balance_ratio=class_balance,
        interpretation="",
        warnings=warnings,
    )
