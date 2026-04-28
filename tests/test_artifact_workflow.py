"""Tests for result import/export and Overleaf packaging helpers."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts.import_results_archive import import_results_archive, validate_archive_member
from scripts.package_artifacts import (
    DEFAULT_PRIMARY_SEED,
    DEFAULT_SEEDS,
    build_results_archive_inputs,
    create_models_archive,
    create_results_archive,
)
from scripts.package_overleaf import create_overleaf_package


def _write(path: Path, content: str = "x") -> None:
    """Write a small text file, creating parents."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _populate_results_inputs(root: Path) -> None:
    """Create the canonical result files expected by package_artifacts."""

    for rel_path in build_results_archive_inputs(DEFAULT_SEEDS, DEFAULT_PRIMARY_SEED):
        path = root / rel_path
        if rel_path.endswith("predictions"):
            _write(path / "transformer_test_predictions.jsonl", "{}\n")
        else:
            _write(path, json.dumps({"path": rel_path}))


def test_build_results_archive_inputs_excludes_old_gallery_and_models() -> None:
    """The results archive should not include the old figure gallery or checkpoints."""

    paths = build_results_archive_inputs()

    assert "reports/figures_all" not in paths
    assert "models" not in paths
    assert "figures/training_curves.pdf" in paths


def test_create_results_archive_writes_manifest_and_zip(tmp_path: Path) -> None:
    """The results archive should contain final artifacts and no model checkpoints."""

    _populate_results_inputs(tmp_path)
    output_path = tmp_path / "disinformation_results.zip"

    manifest = create_results_archive(tmp_path, output_path)

    assert output_path.exists()
    assert manifest["model_checkpoints_included"] is False
    with zipfile.ZipFile(output_path) as archive:
        names = set(archive.namelist())

    assert "reports/archive_manifest.json" in names
    assert "figures/training_curves.pdf" in names
    assert not any(name.startswith("models/") for name in names)


def test_create_models_archive_is_separate(tmp_path: Path) -> None:
    """Model checkpoints should be packaged only in the explicit models archive."""

    _write(tmp_path / "models" / "hybrid" / "best_model.pt", "checkpoint")
    output_path = tmp_path / "models.zip"

    created = create_models_archive(tmp_path, output_path)

    assert created is True
    with zipfile.ZipFile(output_path) as archive:
        assert "models/hybrid/best_model.pt" in archive.namelist()


def test_create_results_archive_uses_legacy_training_log_fallback(tmp_path: Path) -> None:
    """Legacy top-level training logs should be included when seed logs are absent."""

    _populate_results_inputs(tmp_path)
    for log_dir in (
        "transformer_logs",
        "hybrid_logs",
        "hybrid_textonly_logs",
        "hybrid_leaky_logs",
    ):
        seed_log = tmp_path / "reports" / "seed_42" / log_dir / "training_log.csv"
        seed_log.unlink()
        _write(tmp_path / "reports" / log_dir / "training_log.csv", "epoch,train_loss\n")
    output_path = tmp_path / "disinformation_results.zip"

    manifest = create_results_archive(tmp_path, output_path)

    assert not manifest["missing_optional_paths"]
    with zipfile.ZipFile(output_path) as archive:
        assert "reports/transformer_logs/training_log.csv" in archive.namelist()


def test_import_results_archive_rejects_unsafe_paths() -> None:
    """Zip members must stay under the allowed reports/ and figures/ prefixes."""

    with pytest.raises(ValueError, match="unsafe"):
        validate_archive_member("../main.tex")
    with pytest.raises(ValueError, match="unexpected"):
        validate_archive_member("models/best_model.pt")


def test_import_results_archive_extracts_allowed_files(tmp_path: Path) -> None:
    """A valid results archive should extract reports and figures into the repo."""

    archive_path = tmp_path / "disinformation_results.zip"
    required = [
        "reports/results_summary.json",
        "reports/multi_seed_summary.json",
        "reports/seed_42/transformer_logs/transformer_test_metrics.json",
        "reports/seed_42/hybrid_logs/hybrid_test_metrics.json",
        "reports/seed_42/hybrid_leaky_logs/hybrid_test_metrics.json",
        "reports/seed_42/predictions/transformer_test_predictions.jsonl",
    ]
    with zipfile.ZipFile(archive_path, "w") as archive:
        for rel_path in required:
            archive.writestr(rel_path, "x")
        archive.writestr("reports/figures_all/old_diagnostic.pdf", "old")
        archive.writestr("figures/per_class_f1_grouped.pdf", "figure")

    extracted = import_results_archive(archive_path, tmp_path)

    assert len(extracted) == len(required) + 1
    assert (tmp_path / "figures" / "per_class_f1_grouped.pdf").exists()
    assert not (tmp_path / "reports" / "figures_all" / "old_diagnostic.pdf").exists()


def test_create_overleaf_package_contains_minimal_sources(tmp_path: Path) -> None:
    """The Overleaf package should contain only source, bibliography, and final figures."""

    _write(
        tmp_path / "main.tex",
        "\\documentclass{article}\n\\begin{document}x\\end{document}\n",
    )
    _write(tmp_path / "references.bib", "")
    _write(tmp_path / "figures" / "per_class_f1_grouped.pdf")
    _write(tmp_path / "figures" / "confusion_matrices.pdf")
    _write(tmp_path / "figures" / "training_curves.pdf")
    output_path = tmp_path / "overleaf_submission.zip"

    create_overleaf_package(tmp_path, output_path)

    with zipfile.ZipFile(output_path) as archive:
        names = set(archive.namelist())

    assert names == {
        "main.tex",
        "references.bib",
        "figures/per_class_f1_grouped.pdf",
        "figures/confusion_matrices.pdf",
        "figures/training_curves.pdf",
        "README_OVERLEAF.md",
    }
