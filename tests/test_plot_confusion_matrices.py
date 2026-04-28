"""Tests for confusion-matrix plotting helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.plot_confusion_matrices import (
    LABEL_NAMES,
    aggregate_confusion_matrices,
    collect_confusion_matrices,
    plot_confusion_matrices,
    row_normalize,
)


def _write_metrics(path: Path, matrix: list[list[int]]) -> None:
    """Write a minimal metrics JSON file with a confusion matrix."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "test_confusion_matrix": matrix,
        "test_confusion_matrix_labels": list(LABEL_NAMES),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_row_normalize_handles_empty_rows() -> None:
    """Rows with zero support should stay zero rather than producing NaNs."""

    matrix = np.array([[2, 2], [0, 0]], dtype=float)

    normalized = row_normalize(matrix)

    assert normalized[0].tolist() == pytest.approx([0.5, 0.5])
    assert normalized[1].tolist() == pytest.approx([0.0, 0.0])


def test_collect_confusion_matrices_reads_seed_scoped_metrics(tmp_path: Path) -> None:
    """Confusion matrices should be collected from the seed-specific log folders."""

    for seed in (42, 1337):
        matrix = np.eye(len(LABEL_NAMES), dtype=int).tolist()
        _write_metrics(
            tmp_path / f"seed_{seed}" / "hybrid_logs" / "hybrid_test_metrics.json",
            matrix,
        )
        _write_metrics(
            tmp_path / f"seed_{seed}" / "hybrid_leaky_logs" / "hybrid_test_metrics.json",
            matrix,
        )

    collected = collect_confusion_matrices(tmp_path, seeds=(42, 1337))

    assert set(collected) == {"hybrid", "hybrid_leaky"}
    assert collected["hybrid"][42].shape == (6, 6)
    assert collected["hybrid_leaky"][1337][0, 0] == pytest.approx(1.0)


def test_aggregate_confusion_matrices_averages_and_normalizes() -> None:
    """Aggregation should average raw matrices before row-normalising."""

    collected = {
        "hybrid": {
            42: np.array([[2, 0], [1, 1]], dtype=float),
            1337: np.array([[0, 2], [1, 1]], dtype=float),
        }
    }

    aggregated = aggregate_confusion_matrices(collected)

    assert aggregated["hybrid"][0].tolist() == pytest.approx([0.5, 0.5])
    assert aggregated["hybrid"][1].tolist() == pytest.approx([0.5, 0.5])


def test_plot_confusion_matrices_writes_pdf_and_png(tmp_path: Path) -> None:
    """The plotting function should produce both required figure formats."""

    identity = np.eye(len(LABEL_NAMES), dtype=float)
    matrices = {"hybrid_leaky": identity, "hybrid": identity}

    outputs = plot_confusion_matrices(matrices, tmp_path)

    assert outputs["pdf"].exists()
    assert outputs["png"].exists()
    assert outputs["pdf"].stat().st_size > 0
    assert outputs["png"].stat().st_size > 0
