"""Upload validation: enforce size / shape limits and parse defensively."""
from __future__ import annotations

import io

import pandas as pd

from ..config import get_settings


class DatasetValidationError(ValueError):
    """Raised when an uploaded file is not a usable CSV dataset."""


def validate_and_parse(raw: bytes, filename: str) -> pd.DataFrame:
    settings = get_settings()

    if not filename.lower().endswith((".csv", ".txt")):
        raise DatasetValidationError("Only .csv files are supported.")

    if len(raw) == 0:
        raise DatasetValidationError("Uploaded file is empty.")
    if len(raw) > settings.max_upload_bytes:
        raise DatasetValidationError(
            f"File too large ({len(raw) / 1e6:.1f} MB). Limit is {settings.max_upload_mb} MB."
        )

    # Parse defensively — try UTF-8 then fall back to latin-1.
    df: pd.DataFrame | None = None
    last_error: Exception | None = None
    for encoding in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=encoding)
            break
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    if df is None:
        raise DatasetValidationError(f"Could not parse CSV: {last_error}")

    # Normalise column names to strings and strip whitespace.
    df.columns = [str(c).strip() for c in df.columns]

    if df.shape[0] < 2:
        raise DatasetValidationError("Dataset must contain at least 2 rows.")
    if df.shape[1] < 2:
        raise DatasetValidationError(
            "Dataset must contain at least 2 columns (features + target)."
        )
    if df.shape[0] > settings.max_rows:
        raise DatasetValidationError(
            f"Too many rows ({df.shape[0]}). Limit is {settings.max_rows}."
        )
    if df.shape[1] > settings.max_columns:
        raise DatasetValidationError(
            f"Too many columns ({df.shape[1]}). Limit is {settings.max_columns}."
        )

    # Drop columns that are entirely empty — they carry no information.
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
    if df.shape[1] < 2:
        raise DatasetValidationError("Dataset has too few non-empty columns.")

    # Guard against duplicate column names (breaks column indexing downstream).
    if len(set(df.columns)) != len(df.columns):
        df.columns = _dedup_names(list(df.columns))

    return df


def _dedup_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        if name in seen:
            seen[name] += 1
            out.append(f"{name}.{seen[name]}")
        else:
            seen[name] = 0
            out.append(name)
    return out
