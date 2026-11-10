"""Model training, evaluation, and review schemas."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .common import ProblemType


class ModelResult(BaseModel):
    """Result of training and evaluating a single candidate model."""

    model_key: str
    model_name: str
    status: str = "success"  # success | failed
    metrics: dict[str, float] = Field(default_factory=dict)  # held-out test metrics
    train_metrics: dict[str, float] = Field(default_factory=dict)
    cv_mean: dict[str, float] = Field(default_factory=dict)
    cv_std: dict[str, float] = Field(default_factory=dict)
    training_time_sec: float = 0.0
    primary_metric: str | None = None
    primary_score: float | None = None
    error: str | None = None


class LeaderboardEntry(BaseModel):
    model_key: str
    model_name: str
    primary_score: float | None = None
    training_time_sec: float = 0.0
    rank: int


class EvaluationReport(BaseModel):
    problem_type: ProblemType
    metric: str
    best_model_key: str | None = None
    best_model_name: str | None = None
    best_score: float | None = None
    results: list[ModelResult] = Field(default_factory=list)
    leaderboard: list[LeaderboardEntry] = Field(default_factory=list)
    explanation: str = ""


class ModelReview(BaseModel):
    """LLM reviewer output. Must only reference metrics computed by Python."""

    recommendation: str
    explanation: str
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class ResultsResponse(BaseModel):
    experiment_id: str
    problem_type: ProblemType | None = None
    target_column: str | None = None
    metric: str | None = None
    evaluation: EvaluationReport | None = None
    review: ModelReview | None = None
    best_model: dict[str, Any] | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
