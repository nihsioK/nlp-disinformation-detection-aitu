"""Tests for grouped per-class F1 plotting helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.plot_per_class_f1 import (
    LABEL_NAMES,
    PLOT_SYSTEMS,
    aggregate_per_class_f1,
    collect_per_class_f1,
    plot_grouped_per_class_f1,
)


def _write_metrics(path: Path, offset: float) -> None:
    """Write a tiny metrics file with deterministic per-class F1 values."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "test_per_class_f1": {
            label: 0.10 + offset + idx * 0.01
            for idx, label in enumerate(LABEL_NAMES)
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_collect_per_class_f1_reads_seed_scoped_metrics(tmp_path: Path) -> None:
    """Per-class metrics should be collected from seed-specific report folders."""

    for seed, offset in [(42, 0.00), (1337, 0.10)]:
        _write_metrics(
            tmp_path / f"seed_{seed}" / "transformer_logs" / "transformer_test_metrics.json",
            offset,
        )
        _write_metrics(
            tmp_path / f"seed_{seed}" / "hybrid_logs" / "hybrid_test_metrics.json",
            offset + 0.01,
        )
        _write_metrics(
            tmp_path / f"seed_{seed}" / "hybrid_leaky_logs" / "hybrid_test_metrics.json",
            offset + 0.02,
        )

    collected = collect_per_class_f1(tmp_path, seeds=(42, 1337))

    assert set(collected) == set(PLOT_SYSTEMS)
    assert collected["transformer"][42]["pants-fire"] == pytest.approx(0.10)
    assert collected["hybrid"][1337]["true"] == pytest.approx(0.26)


def test_aggregate_per_class_f1_computes_mean_and_sample_std() -> None:
    """Aggregation should preserve label order and use across-seed sample std."""

    collected = {
        "transformer": {
            42: {label: 0.10 for label in LABEL_NAMES},
            1337: {label: 0.20 for label in LABEL_NAMES},
            2024: {label: 0.30 for label in LABEL_NAMES},
        }
    }

    aggregated = aggregate_per_class_f1(collected)

    assert list(aggregated["transformer"]) == list(LABEL_NAMES)
    assert aggregated["transformer"]["pants-fire"]["mean"] == pytest.approx(0.20)
    assert aggregated["transformer"]["pants-fire"]["std"] == pytest.approx(0.10)


def test_plot_grouped_per_class_f1_writes_pdf_and_png(tmp_path: Path) -> None:
    """The plotting function should create both requested figure formats."""

    aggregated = {
        system: {
            label: {"mean": 0.20 + idx * 0.02, "std": 0.01}
            for idx, label in enumerate(LABEL_NAMES)
        }
        for system in PLOT_SYSTEMS
    }

    outputs = plot_grouped_per_class_f1(aggregated, tmp_path)

    assert outputs["pdf"].exists()
    assert outputs["png"].exists()
    assert outputs["pdf"].stat().st_size > 0
    assert outputs["png"].stat().st_size > 0
