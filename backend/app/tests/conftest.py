"""Shared test fixtures.

Critically, this module neutralises any LLM API key and redirects storage to a
temp directory **before** the application settings are first read, so the whole
test-suite runs on the deterministic heuristic path and never touches the real
``data/`` or ``artifacts/`` folders or the network.
"""
from __future__ import annotations

import os
import socket
import tempfile
import threading
import time

# --- must run before any `app.*` import (settings are lru_cached) -----------
_TMP = tempfile.mkdtemp(prefix="automl_test_")
os.environ.update(
    {
        "GROQ_API_KEY": "",
        "OPENAI_API_KEY": "",
        "LLM_API_KEY": "",
        "DATA_DIR": os.path.join(_TMP, "data"),
        "ARTIFACTS_DIR": os.path.join(_TMP, "artifacts"),
        "CV_FOLDS": "3",
        "LOG_LEVEL": "WARNING",
    }
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()
get_settings().ensure_dirs()


# ---- synthetic datasets ----------------------------------------------------
def make_classification_df(n: int = 140, seed: int = 42) -> pd.DataFrame:
    """A learnable binary-classification dataset with mixed column types."""
    rng = np.random.default_rng(seed)
    income = rng.normal(50_000, 12_000, n)
    age = rng.integers(18, 75, n)
    plan = rng.choice(["basic", "premium", "enterprise"], n)
    # A clearly-separable target with a little label noise.
    base = ((income < 48_000) | (plan == "basic")).astype(int)
    flip = rng.random(n) < 0.05
    churn = np.where(flip, 1 - base, base)
    return pd.DataFrame(
        {"age": age, "income": income.round(2), "plan": plan, "churn": churn}
    )


def make_regression_df(n: int = 140, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    size = rng.normal(120, 30, n)
    rooms = rng.integers(1, 6, n)
    location = rng.choice(["north", "south", "central"], n)
    loc_bonus = np.select(
        [location == "central", location == "north"], [50_000, 20_000], default=0
    )
    price = 2_000 * size + 15_000 * rooms + loc_bonus + rng.normal(0, 8_000, n)
    return pd.DataFrame(
        {"size": size.round(2), "rooms": rooms, "location": location, "price": price.round(2)}
    )


def make_ambiguous_df(n: int = 120, seed: int = 3) -> pd.DataFrame:
    """Generic column names so the planner cannot confidently pick a target."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)
    b = rng.normal(5, 2, n)
    c = rng.choice(["x", "y", "z"], n)
    d = (a > 0).astype(int)  # depends on 'a'; 2 classes -> classification
    return pd.DataFrame({"a": a.round(3), "b": b.round(3), "c": c, "d": d})


@pytest.fixture
def clf_df() -> pd.DataFrame:
    return make_classification_df()


@pytest.fixture
def reg_df() -> pd.DataFrame:
    return make_regression_df()


@pytest.fixture
def clf_csv_bytes() -> bytes:
    return make_classification_df().to_csv(index=False).encode("utf-8")


@pytest.fixture
def ambiguous_csv_bytes() -> bytes:
    return make_ambiguous_df().to_csv(index=False).encode("utf-8")


# ---- live server (for API integration tests) -------------------------------
def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def live_server() -> str:
    """Start the real ASGI app on a background uvicorn server; yield its base URL."""
    import uvicorn

    from app.main import app

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    for _ in range(100):  # wait up to 10s for startup
        if server.started:
            break
        time.sleep(0.1)
    if not server.started:
        raise RuntimeError("Test server failed to start.")

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def api(live_server: str):
    import httpx

    with httpx.Client(base_url=live_server, timeout=60.0) as client:
        yield client
