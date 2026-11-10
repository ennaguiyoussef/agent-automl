"""Shared FastAPI dependencies."""
from __future__ import annotations

from ..services.experiment_manager import ExperimentManager, get_experiment_manager


def get_manager() -> ExperimentManager:
    """Provide the process-wide experiment manager."""
    return get_experiment_manager()
