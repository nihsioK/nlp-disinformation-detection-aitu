"""Create reproducible result and model archives after training."""

from __future__ import annotations

import argparse
import json
import logging
import zipfile
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (42, 1337, 2024)
DEFAULT_PRIMARY_SEED = 42
FINAL_FIGURES = (
    "figures/per_class_f1_grouped.pdf",
    "figures/per_class_f1_grouped.png",
    "figures/confusion_matrices.pdf",
    "figures/confusion_matrices.png",
    "figures/training_curves.pdf",
    "figures/training_curves.png",
)
AGGREGATE_REPORTS = (
    "reports/results_summary.json",
    "reports/multi_seed_summary.json",
    "reports/bootstrap_ci.json",
    "reports/bootstrap_ci_table.md",
    "reports/leakage_verification.json",
    "reports/leakage_verification_predictions.json",
)
MODEL_LOG_DIRS = (
    "transformer_logs",
    "hybrid_logs",
    "hybrid_textonly_logs",
    "hybrid_leaky_logs",
)


def build_results_archive_inputs(
    seeds: Sequence[int] = DEFAULT_SEEDS,
    primary_seed: int = DEFAULT_PRIMARY_SEED,
) -> list[str]:
    """Return the canonical archive path list relative to the project root."""

    paths = [*AGGREGATE_REPORTS, *FINAL_FIGURES]
    for seed in seeds:
        paths.extend([
            f"reports/seed_{seed}/baseline_detailed_metrics.json",
            f"reports/seed_{seed}/transformer_logs/transformer_test_metrics.json",
            f"reports/seed_{seed}/hybrid_logs/hybrid_test_metrics.json",
            f"reports/seed_{seed}/hybrid_textonly_logs/hybrid_test_metrics.json",
            f"reports/seed_{seed}/hybrid_leaky_logs/hybrid_test_metrics.json",
            f"reports/seed_{seed}/predictions",
        ])

    for log_dir in MODEL_LOG_DIRS:
        paths.append(f"reports/seed_{primary_seed}/{log_dir}/training_log.csv")
    return paths


def existing_and_missing_paths(
    project_root: Path,
    paths: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Split a relative path list into existing and missing entries."""

    existing = [path for path in paths if (project_root / path).exists()]
    missing = [path for path in paths if not (project_root / path).exists()]
    return existing, missing


def write_zip(project_root: Path, output_path: Path, relative_paths: Sequence[str]) -> None:
    """Write files/directories into a zip while preserving relative paths."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative_path in relative_paths:
            path = project_root / relative_path
            if path.is_dir():
                for child in sorted(item for item in path.rglob("*") if item.is_file()):
                    archive.write(child, child.relative_to(project_root).as_posix())
            elif path.is_file():
                archive.write(path, relative_path)


def create_results_archive(
    project_root: Path,
    output_path: Path,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    primary_seed: int = DEFAULT_PRIMARY_SEED,
) -> dict[str, object]:
    """Create `disinformation_results.zip` and return its manifest payload."""

    archive_inputs = [
        path
        for path in build_results_archive_inputs(seeds=seeds, primary_seed=primary_seed)
        if not path.endswith("/training_log.csv")
    ]
    archive_inputs.extend(_resolve_training_log_inputs(project_root, primary_seed))
    existing_inputs, missing_inputs = existing_and_missing_paths(project_root, archive_inputs)

    manifest = {
        "archive": str(output_path),
        "included_paths": existing_inputs,
        "missing_optional_paths": missing_inputs,
        "model_checkpoints_included": False,
        "note": (
            "This archive contains only thesis/paper artifacts: aggregate metrics, "
            "seed-scoped metrics, prediction JSONL files, bootstrap/leakage reports, "
            "final figures, and training logs for the training-curves figure. "
            "Model checkpoints, raw CSV run histories, and diagnostic figure galleries "
            "are excluded."
        ),
    }

    manifest_path = project_root / "reports" / "archive_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_zip(project_root, output_path, [*existing_inputs, "reports/archive_manifest.json"])
    return manifest


def _resolve_training_log_inputs(project_root: Path, primary_seed: int) -> list[str]:
    """Prefer seed-scoped logs, falling back to legacy top-level logs."""

    paths = []
    for log_dir in MODEL_LOG_DIRS:
        seed_scoped = f"reports/seed_{primary_seed}/{log_dir}/training_log.csv"
        legacy_top_level = f"reports/{log_dir}/training_log.csv"
        if (project_root / seed_scoped).exists():
            paths.append(seed_scoped)
        elif (project_root / legacy_top_level).exists():
            paths.append(legacy_top_level)
        else:
            paths.append(seed_scoped)
    return paths


def create_models_archive(project_root: Path, output_path: Path) -> bool:
    """Create a separate `models.zip` archive if the models directory exists."""

    models_dir = project_root / "models"
    if not models_dir.exists():
        return False
    write_zip(project_root, output_path, ["models"])
    return True


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root containing reports/, figures/, and models/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("disinformation_results.zip"),
        help="Path for the paper/reproducibility results archive.",
    )
    parser.add_argument(
        "--models-output",
        type=Path,
        default=Path("models.zip"),
        help="Path for the separate model-checkpoint archive.",
    )
    parser.add_argument(
        "--include-models",
        action="store_true",
        help="Also create a separate models.zip archive from models/ if present.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Training seeds expected in reports/seed_<N>/.",
    )
    parser.add_argument(
        "--primary-seed",
        type=int,
        default=DEFAULT_PRIMARY_SEED,
        help="Seed whose training logs are used for the paper training-curves figure.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    project_root = args.project_root.resolve()

    manifest = create_results_archive(
        project_root=project_root,
        output_path=args.output.resolve(),
        seeds=tuple(args.seeds),
        primary_seed=int(args.primary_seed),
    )
    logging.info("Wrote %s", args.output)
    if manifest["missing_optional_paths"]:
        logging.warning("Missing optional paths: %s", manifest["missing_optional_paths"])

    if args.include_models:
        wrote_models = create_models_archive(project_root, args.models_output.resolve())
        if wrote_models:
            logging.info("Wrote %s", args.models_output)
        else:
            logging.warning("Skipped models archive because models/ does not exist.")


if __name__ == "__main__":
    main()
