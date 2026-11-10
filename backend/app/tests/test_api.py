"""API integration tests against a real (background) uvicorn server."""
from __future__ import annotations

import time

import pytest

from .conftest import make_ambiguous_df, make_classification_df

TERMINAL = {"completed", "failed", "awaiting_human"}


def _poll(api, experiment_id: str, want: set[str], timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = api.get(f"/api/experiments/{experiment_id}").json()
        if snap["status"] in want:
            return snap
        time.sleep(0.4)
    raise AssertionError(f"experiment {experiment_id} never reached {want} (last={snap})")


def _upload(api, df, name="data.csv") -> str:
    resp = api.post(
        "/api/datasets/upload",
        files={"file": (name, df.to_csv(index=False).encode(), "text/csv")},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["dataset_id"]


def test_health(api):
    body = api.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["llm_available"] is False  # tests force the heuristic path


def test_upload_valid_and_invalid(api):
    dataset_id = _upload(api, make_classification_df())
    assert dataset_id

    # A single-column file is not a usable dataset -> 422.
    bad = api.post(
        "/api/datasets/upload",
        files={"file": ("bad.csv", b"only_one_column\n1\n2\n3\n", "text/csv")},
    )
    assert bad.status_code == 422


def test_full_experiment_lifecycle(api):
    dataset_id = _upload(api, make_classification_df())
    created = api.post(
        "/api/experiments",
        json={"dataset_id": dataset_id, "user_query": "predict churn", "target_column": "churn"},
    )
    assert created.status_code == 201
    eid = created.json()["id"]

    snap = _poll(api, eid, {"completed", "failed"})
    assert snap["status"] == "completed", snap.get("error")
    assert snap["problem_type"] == "classification"
    assert snap["has_results"] is True

    results = api.get(f"/api/experiments/{eid}/results").json()
    assert results["best_model"]["score"] is not None
    assert results["evaluation"]["leaderboard"]
    assert results["review"]["recommendation"]

    model = api.get(f"/api/experiments/{eid}/download-model")
    assert model.status_code == 200
    assert model.headers["content-type"] == "application/octet-stream"
    assert len(model.content) > 0

    report = api.get(f"/api/experiments/{eid}/download-report")
    assert report.status_code == 200
    assert report.json()["experiment_id"] == eid


def test_sse_stream_replays_history(api):
    """After completion, subscribing replays the full event backlog."""
    dataset_id = _upload(api, make_classification_df())
    eid = api.post(
        "/api/experiments",
        json={"dataset_id": dataset_id, "user_query": "predict churn", "target_column": "churn"},
    ).json()["id"]
    _poll(api, eid, {"completed", "failed"})

    seen: list[str] = []
    with api.stream("GET", f"/api/experiments/{eid}/events", timeout=10.0) as s:
        for line in s.iter_lines():
            if line.startswith("event:"):
                seen.append(line.split(":", 1)[1].strip())
            if "experiment_completed" in seen or len(seen) > 100:
                break
    assert "experiment_started" in seen
    assert "experiment_completed" in seen


def test_human_in_the_loop_over_http(api):
    dataset_id = _upload(api, make_ambiguous_df(), name="ambiguous.csv")
    eid = api.post(
        "/api/experiments",
        json={"dataset_id": dataset_id, "user_query": "build a model"},  # no target
    ).json()["id"]

    snap = _poll(api, eid, {"awaiting_human", "completed", "failed"})
    assert snap["status"] == "awaiting_human", snap
    assert snap["pending_human"] is not None
    assert set(snap["pending_human"]["options"]) == set(make_ambiguous_df().columns)

    resumed = api.post(f"/api/experiments/{eid}/human-response", json={"target_column": "d"})
    assert resumed.status_code == 200

    final = _poll(api, eid, {"completed", "failed"})
    assert final["status"] == "completed", final.get("error")
    assert final["target_column"] == "d"


def test_chat_after_completion_is_grounded(api):
    dataset_id = _upload(api, make_classification_df())
    eid = api.post(
        "/api/experiments",
        json={"dataset_id": dataset_id, "user_query": "predict churn", "target_column": "churn"},
    ).json()["id"]
    _poll(api, eid, {"completed", "failed"})

    reply = api.post(f"/api/experiments/{eid}/chat", json={"message": "what is the best model?"})
    assert reply.status_code == 200
    body = reply.json()
    assert body["reply"]  # heuristic fallback still returns a grounded summary


def test_unknown_experiment_returns_404(api):
    assert api.get("/api/experiments/does-not-exist").status_code == 404
