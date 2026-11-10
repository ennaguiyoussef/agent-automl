"""Shared enums used across schemas, state, and events."""
from __future__ import annotations

from enum import Enum


class ProblemType(str, Enum):
    classification = "classification"
    regression = "regression"


class ExperimentStatus(str, Enum):
    created = "created"
    running = "running"
    awaiting_human = "awaiting_human"
    completed = "completed"
    failed = "failed"


class InferredType(str, Enum):
    numeric = "numeric"
    categorical = "categorical"
    boolean = "boolean"
    datetime = "datetime"
    text = "text"


class EventType(str, Enum):
    """Every SSE event type streamed to the frontend."""

    experiment_started = "experiment_started"
    dataset_loaded = "dataset_loaded"
    planner_started = "planner_started"
    planner_completed = "planner_completed"
    dataset_analysis_started = "dataset_analysis_started"
    dataset_analysis_completed = "dataset_analysis_completed"
    eda_started = "eda_started"
    eda_completed = "eda_completed"
    human_input_required = "human_input_required"
    human_input_received = "human_input_received"
    preprocessing_started = "preprocessing_started"
    preprocessing_completed = "preprocessing_completed"
    model_selection_completed = "model_selection_completed"
    training_started = "training_started"
    model_training = "model_training"
    model_completed = "model_completed"
    evaluation_completed = "evaluation_completed"
    reviewer_started = "reviewer_started"
    reviewer_completed = "reviewer_completed"
    artifact_created = "artifact_created"
    experiment_completed = "experiment_completed"
    chat_message = "chat_message"
    log = "log"
    error = "error"
