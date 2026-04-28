"""Tests for training-curves plotting helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.plot_training_curves import (
    PLOT_SYSTEMS,
    collect_training_logs,
    plot_training_curves,
)


def _write_log(path: Path, offset: float) -> None:
    """Write a minimal deterministic training log."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "epoch": [1, 2],
        "train_loss": [1.0 + offset, 0.9 + offset],
        "val_loss": [1.1 + offset, 1.0 + offset],
        "val_macro_f1": [0.20 + offset, 0.25 + offset],
    })
    df.to_csv(path, index=False)


def test_collect_training_logs_prefers_seed_scoped_files(tmp_path: Path) -> None:
    """Seed-scoped logs should be read before top-level fallback logs."""

    for system in PLOT_SYSTEMS:
        log_dir = "transformer_logs" if system == "transformer" else f"{system}_logs"
        _write_log(tmp_path / log_dir / "training_log.csv", 0.50)
        _write_log(tmp_path / "seed_42" / log_dir / "training_log.csv", 0.00)

    logs = collect_training_logs(tmp_path, seed=42)

    assert set(logs) == set(PLOT_SYSTEMS)
    assert logs["transformer"].loc[0, "train_loss"] == pytest.approx(1.0)


def test_collect_training_logs_falls_back_to_top_level(tmp_path: Path) -> None:
    """Older archives without seed-scoped CSVs should still be usable."""

    for system in PLOT_SYSTEMS:
        log_dir = "transformer_logs" if system == "transformer" else f"{system}_logs"
        _write_log(tmp_path / log_dir / "training_log.csv", 0.10)

    logs = collect_training_logs(tmp_path, seed=42)

    assert logs["hybrid_leaky"].loc[1, "val_macro_f1"] == pytest.approx(0.35)


def test_plot_training_curves_writes_pdf_and_png(tmp_path: Path) -> None:
    """The plotting function should produce both required figure formats."""

    logs = {
        system: pd.DataFrame({
            "epoch": [1, 2, 3],
            "train_loss": [1.0, 0.8, 0.7],
            "val_loss": [1.1, 0.9, 0.85],
            "val_macro_f1": [0.20, 0.25, 0.30],
        })
        for system in PLOT_SYSTEMS
    }

    outputs = plot_training_curves(logs, tmp_path)

    assert outputs["pdf"].exists()
    assert outputs["png"].exists()
    assert outputs["pdf"].stat().st_size > 0
    assert outputs["png"].stat().st_size > 0
