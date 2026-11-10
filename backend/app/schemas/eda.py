"""Exploratory data analysis schemas."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class NumericStat(BaseModel):
    name: str
    count: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    q25: float | None = None
    median: float | None = None
    q75: float | None = None
    max: float | None = None
    missing: int = 0
    skew: float | None = None


class CategoricalStat(BaseModel):
    name: str
    unique: int
    top: Any | None = None
    top_freq: int = 0
    missing: int = 0
    top_values: list[dict[str, Any]] = Field(default_factory=list)


class CorrelationPair(BaseModel):
    feature_a: str
    feature_b: str
    correlation: float


class MissingEntry(BaseModel):
    column: str
    missing: int
    missing_pct: float


class EdaReport(BaseModel):
    summary: dict[str, Any] = Field(default_factory=dict)
    numeric_stats: list[NumericStat] = Field(default_factory=list)
    categorical_stats: list[CategoricalStat] = Field(default_factory=list)
    missing_report: list[MissingEntry] = Field(default_factory=list)
    top_correlations: list[CorrelationPair] = Field(default_factory=list)
    target_distribution: dict[str, Any] | None = None
    class_balance_ratio: float | None = None
    interpretation: str = ""
    warnings: list[str] = Field(default_factory=list)
