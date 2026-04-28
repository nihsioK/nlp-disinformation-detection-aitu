"""Tests for bootstrap confidence interval utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.bootstrap_ci import (
    LABEL_NAMES,
    SYSTEMS,
    aggregate_seed_intervals,
    compute_metric_bundle,
    discover_prediction_files,
    paired_bootstrap_comparison,
)


def _write_prediction_file(path: Path, system: str, seed: int) -> None:
    """Write a tiny prediction JSONL file for artifact-discovery tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "model": system,
            "seed": seed,
            "id": f"{idx}.json",
            "label_id": idx % len(LABEL_NAMES),
            "pred_label_id": idx % len(LABEL_NAMES),
        }
        for idx in range(12)
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_compute_metric_bundle_reports_macro_accuracy_and_per_class_f1() -> None:
    """Metric bundles should expose point estimates for every required metric."""

    y_true = np.array([0, 1, 2, 3, 4, 5])
    y_pred = np.array([0, 1, 2, 3, 4, 0])

    metrics = compute_metric_bundle(y_true, y_pred)

    assert metrics["accuracy"] == pytest.approx(5 / 6)
    assert metrics["macro_f1"] == pytest.approx(0.7777777778)
    assert set(metrics["per_class_f1"]) == set(LABEL_NAMES)
    assert metrics["per_class_f1"]["true"] == pytest.approx(0.0)


def test_aggregate_seed_intervals_averages_points_and_ci_bounds() -> None:
    """Seed aggregation should average each percentile bound independently."""

    per_seed = {
        "seed42": {
            "macro_f1": {"point": 0.20, "ci_low": 0.10, "median": 0.21, "ci_high": 0.30},
            "accuracy": {"point": 0.40, "ci_low": 0.30, "median": 0.41, "ci_high": 0.50},
            "per_class_f1": {
                "pants-fire": {"point": 0.10, "ci_low": 0.00, "median": 0.11, "ci_high": 0.20}
            },
        },
        "seed1337": {
            "macro_f1": {"point": 0.30, "ci_low": 0.20, "median": 0.31, "ci_high": 0.40},
            "accuracy": {"point": 0.50, "ci_low": 0.40, "median": 0.51, "ci_high": 0.60},
            "per_class_f1": {
                "pants-fire": {"point": 0.30, "ci_low": 0.20, "median": 0.31, "ci_high": 0.40}
            },
        },
    }

    averaged = aggregate_seed_intervals(per_seed)

    assert averaged["macro_f1"]["point"] == pytest.approx(0.25)
    assert averaged["macro_f1"]["ci_low"] == pytest.approx(0.15)
    assert averaged["macro_f1"]["median"] == pytest.approx(0.26)
    assert averaged["macro_f1"]["ci_high"] == pytest.approx(0.35)
    assert averaged["per_class_f1"]["pants-fire"]["ci_high"] == pytest.approx(0.30)


def test_discover_prediction_files_prefers_seed_directories(tmp_path: Path) -> None:
    """The archive layout stores the 21 canonical JSONLs in seed directories."""

    for seed in (42, 1337, 2024):
        for system in SYSTEMS:
            path = tmp_path / f"seed_{seed}" / "predictions" / f"{system}_test_predictions.jsonl"
            _write_prediction_file(path, system, seed)

    discovered = discover_prediction_files(tmp_path)

    assert len(discovered) == 21
    assert discovered[("hybrid", 42)].name == "hybrid_test_predictions.jsonl"
    assert discovered[("baseline_svm", 2024)].is_file()


def test_paired_bootstrap_comparison_reports_sign_flip_p_value() -> None:
    """Paired bootstrap should estimate A-B differences and two-sided p-values."""

    np.random.seed(0)
    y_true = np.array([0, 1, 2, 3, 4, 5] * 8)
    pred_a = y_true.copy()
    pred_b = np.zeros_like(y_true)

    comparison = paired_bootstrap_comparison(
        per_seed_arrays={
            42: (y_true, pred_a, pred_b),
            1337: (y_true, pred_a, pred_b),
        },
        n_resamples=200,
    )

    assert comparison["macro_f1_diff"]["point"] > 0
    assert comparison["macro_f1_diff"]["ci_low"] > 0
    assert comparison["macro_f1_diff"]["p_value"] == pytest.approx(0.0)
    assert comparison["per_class_diff"]["true"]["point"] == pytest.approx(1.0)
