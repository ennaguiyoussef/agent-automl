"""Dataset & artifact storage.

Datasets are stored on disk as ``<data_dir>/<dataset_id>.csv`` with a small
sidecar ``.meta.json``. A cached loader avoids re-parsing the same CSV on every
graph node (the parse is the expensive part; we still hand out copies so nodes
never mutate the cache).
"""
from __future__ import annotations

import json
import os
import uuid
from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..config import get_settings


def new_id() -> str:
    return uuid.uuid4().hex


def _data_dir() -> Path:
    settings = get_settings()
    settings.ensure_dirs()
    return settings.data_dir


def dataset_path(dataset_id: str) -> Path:
    return _data_dir() / f"{dataset_id}.csv"


def dataset_meta_path(dataset_id: str) -> Path:
    return _data_dir() / f"{dataset_id}.meta.json"


def save_dataset(df: pd.DataFrame, filename: str) -> tuple[str, Path]:
    dataset_id = new_id()
    path = dataset_path(dataset_id)
    df.to_csv(path, index=False)
    dataset_meta_path(dataset_id).write_text(
        json.dumps({"dataset_id": dataset_id, "filename": filename}), encoding="utf-8"
    )
    return dataset_id, path


def dataset_exists(dataset_id: str) -> bool:
    return dataset_path(dataset_id).exists()


def dataset_filename(dataset_id: str) -> str:
    meta = dataset_meta_path(dataset_id)
    if meta.exists():
        try:
            return json.loads(meta.read_text(encoding="utf-8")).get("filename", f"{dataset_id}.csv")
        except json.JSONDecodeError:
            pass
    return f"{dataset_id}.csv"


@lru_cache(maxsize=16)
def _load_cached(path: str, mtime: float, size: int) -> pd.DataFrame:
    return pd.read_csv(path)


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a CSV (cached by path+mtime+size). Returns a fresh copy each call."""
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    stat = os.stat(path)
    return _load_cached(path, stat.st_mtime, stat.st_size).copy()


# ---- artifacts ---------------------------------------------------------
def artifacts_dir(experiment_id: str) -> Path:
    settings = get_settings()
    settings.ensure_dirs()
    directory = settings.artifacts_dir / experiment_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
