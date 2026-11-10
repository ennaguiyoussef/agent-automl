"""Small shared helpers, mainly for turning pandas/numpy values into JSON-safe
native Python types. Every tool result must be JSON-serialisable because it is
stored in the graph state and streamed over SSE.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def to_native(value: Any) -> Any:
    """Convert a pandas/numpy scalar to a JSON-serialisable native Python value.

    ``NaN`` / ``inf`` become ``None`` so the value round-trips through JSON.
    """
    if value is None:
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.str_,)):
        return str(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    try:
        # Catch pandas NA / NaT for scalar inputs only.
        if np.isscalar(value) and pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, str, bool)):
        return value
    # Fall back to a string representation for anything exotic.
    if isinstance(value, (list, tuple)):
        return [to_native(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_native(v) for k, v in value.items()}
    return value


def safe_float(value: Any) -> float | None:
    """Coerce to float, mapping NaN/inf/None to ``None``."""
    native = to_native(value)
    if native is None:
        return None
    try:
        f = float(native)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def round_float(value: Any, ndigits: int = 6) -> float | None:
    f = safe_float(value)
    return None if f is None else round(f, ndigits)
