"""End-to-end smoke tests for the compiled LangGraph workflow."""
from __future__ import annotations

import json

from langgraph.types import Command

from app.config import get_settings
from app.graph import build_graph
from .conftest import make_ambiguous_df, make_classification_df


def _write_csv(df, tmp_path, name="data.csv") -> str:
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


def test_graph_completes_and_writes_artifacts(tmp_path):
    df = make_classification_df()
    dataset_path = _write_csv(df, tmp_path)
    graph = build_graph()
    config = {"configurable": {"thread_id": "smoke-clf"}}

    state = graph.invoke(
        {
            "experiment_id": "smoke-clf",
            "user_query": "predict churn",
            "dataset_path": dataset_path,
            "target_column": "churn",  # explicit -> no human interrupt
        },
        config=config,
    )

    assert state.get("error") is None
    assert state["target_column"] == "churn"
    assert state["problem_type"] == "classification"
    best = state.get("best_model")
    assert best and best.get("score") is not None

    # Artifacts were written to disk and the pickle is a working pipeline.
    artifacts_dir = get_settings().artifacts_dir / "smoke-clf"
    for fname in ("best_model.pkl", "metrics.json", "report.json", "ml_plan.json"):
        assert (artifacts_dir / fname).exists()

    metrics = json.loads((artifacts_dir / "metrics.json").read_text())
    assert metrics["best_model_key"] == best["model_key"]

    import joblib

    pipe = joblib.load(artifacts_dir / "best_model.pkl")
    preds = pipe.predict(df.drop(columns=["churn"]).head(3))
    assert len(preds) == 3


def test_graph_human_in_the_loop_interrupt_and_resume(tmp_path):
    df = make_ambiguous_df()
    dataset_path = _write_csv(df, tmp_path, "ambiguous.csv")
    graph = build_graph()
    config = {"configurable": {"thread_id": "smoke-hitl"}}

    result = graph.invoke(
        {
            "experiment_id": "smoke-hitl",
            "user_query": "build a model",
            "dataset_path": dataset_path,
        },
        config=config,
    )

    # The graph should pause awaiting target confirmation.
    snapshot = graph.get_state(config)
    assert snapshot.next == ("target_confirmation",)
    interrupts = result.get("__interrupt__")
    assert interrupts, "expected an interrupt payload"
    payload = interrupts[0].value
    assert "options" in payload and set(payload["options"]) == set(df.columns)

    # Resume with an explicit choice; the graph should run to completion.
    final = graph.invoke(Command(resume={"target_column": "d"}), config=config)
    assert final.get("error") is None
    assert final["target_column"] == "d"
    assert final.get("best_model")
    assert graph.get_state(config).next == ()
