"""Tests for baseline modeling and evaluation helpers."""

from __future__ import annotations

from pathlib import Path

from src.disinfo_detection.evaluation import compute_metrics
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

    X_train = [
        "true policy success",
        "false claim rumor",
        "budget reform successful",
        "hoax conspiracy false",
        "verified statement true",
        "fake misleading rumor",
    ]
    y_train = [5, 1, 4, 0, 5, 1]
    X_valid = ["true reform", "false rumor"]

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
