"""Planner and preprocessing schemas.

``MLPlan`` is the *structured output* the planner LLM must return. Keeping the
model here (rather than inside the agent) makes it reusable by the API layer and
the tests, and documents the contract the LLM is held to.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from .common import ProblemType


class MLPlan(BaseModel):
    """Structured plan produced by the planner agent."""

    problem_type: ProblemType = Field(
        description="Whether this is a classification or regression task."
    )
    target_column: str = Field(description="The column the model should predict.")
    target_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How confident the planner is about the chosen target column.",
    )
    recommended_metric: str = Field(
        description="Primary metric to optimise, e.g. f1, roc_auc, r2, rmse."
    )
    candidate_models: list[str] = Field(
        default_factory=list,
        description="Model keys to train (must exist in the model registry).",
    )
    reasoning: str = Field(description="Why this plan fits the data and the request.")


class PreprocessingConfig(BaseModel):
    """Deterministic description of the preprocessing pipeline that was built."""

    numeric_features: list[str] = Field(default_factory=list)
    categorical_features: list[str] = Field(default_factory=list)
    dropped_features: list[str] = Field(default_factory=list)
    high_cardinality_features: list[str] = Field(default_factory=list)
    numeric_imputer: str = "median"
    categorical_imputer: str = "most_frequent"
    scale_numeric: bool = True
    onehot_encode: bool = True
    max_onehot_cardinality: int = 40
    notes: list[str] = Field(default_factory=list)
