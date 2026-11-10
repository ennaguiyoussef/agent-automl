"""Dataset profiling schemas."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .common import InferredType


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    inferred_type: InferredType
    missing: int
    missing_pct: float
    unique: int
    unique_pct: float
    is_constant: bool = False
    is_high_cardinality: bool = False
    sample_values: list[Any] = Field(default_factory=list)


class DatasetInfo(BaseModel):
    """Deterministic profile of an uploaded dataset."""

    n_rows: int
    n_columns: int
    column_names: list[str]
    columns: list[ColumnInfo]
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    boolean_columns: list[str] = Field(default_factory=list)
    constant_columns: list[str] = Field(default_factory=list)
    high_cardinality_columns: list[str] = Field(default_factory=list)
    duplicate_rows: int
    total_missing: int
    missing_pct: float
    memory_usage_mb: float
    dtypes: dict[str, str]
    head: list[dict[str, Any]] = Field(default_factory=list)


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    filename: str
    info: DatasetInfo
