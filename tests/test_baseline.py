"""Tests for baseline modeling and evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.disinfo_detection.evaluation import (
    build_env_record,
    build_prediction_records,
    compute_metrics,
    write_jsonl_records,
)
from src.disinfo_detection.models_baseline import TFIDFBaseline


def test_compute_metrics_returns_expected_keys() -> None:
    """Metric computation should expose top-level summary fields."""

    metrics = compute_metrics(
        y_true=[0, 1, 2, 2],
        y_pred=[0, 1, 1, 2],
        label_names=["pants-fire", "false", "barely-true"],
    )
    assert {"accuracy", "macro_f1", "per_class_f1", "classification_report"} <= set(metrics)
    assert metrics["accuracy"] == 0.75


def test_tfidf_baseline_fit_predict_and_save(tmp_path: Path) -> None:
    """A TF-IDF baseline should train, predict, and persist successfully."""

    # The production baseline config uses min_df=3, so each term in the toy
    # corpus has to appear at least 3 times or TfidfVectorizer will prune
    # everything. We duplicate a compact vocabulary accordingly.
    X_train = [
        "true policy reform success claim rumor",
        "false hoax conspiracy claim rumor fake",
        "true reform policy success verified statement",
        "false claim hoax fake misleading rumor",
        "true verified policy reform claim statement",
        "false fake hoax conspiracy misleading rumor",
    ]
    y_train = [5, 1, 4, 0, 5, 1]
    X_valid = ["true policy reform", "false claim rumor"]

    model = TFIDFBaseline("naive_bayes")
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)
    probabilities = model.predict_proba(X_valid)
    output_path = tmp_path / "baseline_nb.pkl"
    model.save(str(output_path))
    loaded = TFIDFBaseline.load(str(output_path))

    assert len(predictions) == 2
    assert probabilities.shape[0] == 2
    assert output_path.exists()
    assert loaded.predict(X_valid) == predictions


def test_build_env_record_includes_reproducibility_fields() -> None:
    """Environment fingerprints should contain the fields needed to audit a run."""

    record = build_env_record(
        seed=42,
        device="cpu",
        run_timestamp="2026-04-24T00:00:00+00:00",
    )
    expected = {
        "git_sha",
        "python_version",
        "torch_version",
        "transformers_version",
        "device",
        "seed",
        "run_timestamp",
    }
    assert expected <= set(record)
    assert record["seed"] == 42
    assert record["device"] == "cpu"


def test_prediction_jsonl_helper_writes_one_record_per_row(tmp_path: Path) -> None:
    """Prediction JSONL artifacts should preserve labels, metadata, and probabilities."""

    frame = pd.DataFrame(
        [
            {
                "id": "test-1",
                "statement": "Claim one.",
                "label_id": 0,
                "speaker": "alice",
                "party": "independent",
                "job": "analyst",
                "state": "TX",
                "subject": "economy",
                "context": "speech",
            },
            {
                "id": "test-2",
                "statement": "Claim two.",
                "label_id": 1,
                "speaker": "bob",
                "party": "democrat",
                "job": "governor",
                "state": "CA",
                "subject": "health",
                "context": "debate",
            },
        ]
    )
    records = build_prediction_records(
        frame=frame,
        predictions=[0, 0],
        probabilities=[[0.9, 0.1], [0.6, 0.4]],
        logits=None,
        label_names=["pants-fire", "false"],
        model_name="baseline_naive_bayes",
        seed=42,
        split="test",
    )
    output_path = tmp_path / "predictions.jsonl"
    write_jsonl_records(records, output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    decoded = [json.loads(line) for line in lines]
    assert len(decoded) == 2
    assert decoded[0]["model"] == "baseline_naive_bayes"
    assert decoded[0]["true_label"] == "pants-fire"
    assert decoded[0]["pred_label"] == "pants-fire"
    assert decoded[0]["correct"] is True
    assert decoded[0]["probabilities"] == [0.9, 0.1]
    assert decoded[0]["logits"] is None
