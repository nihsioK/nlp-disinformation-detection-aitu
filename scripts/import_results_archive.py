"""Import a Kaggle `disinformation_results.zip` archive into the repository."""

from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_PREFIXES = ("reports/", "figures/")
SKIPPED_PREFIXES = (
    "reports/figures_all/",
    "reports/paper/",
    "reports/task1_task3/",
)
REQUIRED_PATHS = (
    "reports/results_summary.json",
    "reports/multi_seed_summary.json",
    "reports/seed_42/transformer_logs/transformer_test_metrics.json",
    "reports/seed_42/hybrid_logs/hybrid_test_metrics.json",
    "reports/seed_42/hybrid_leaky_logs/hybrid_test_metrics.json",
    "reports/seed_42/predictions/transformer_test_predictions.jsonl",
)


def validate_archive_member(name: str) -> Path:
    """Validate and normalize one zip member path.

    Args:
        name: Raw zip member name.

    Returns:
        Normalized relative path.

    Raises:
        ValueError: If the member is absolute, escapes the repository, or is outside
            the allowed result-artifact prefixes.
    """

    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive path: {name}")
    normalized = Path(*path.parts)
    normalized_posix = normalized.as_posix()
    if not normalized_posix.startswith(ALLOWED_PREFIXES):
        raise ValueError(f"unexpected archive path: {name}")
    return normalized


def import_results_archive(archive_path: Path, project_root: Path = PROJECT_ROOT) -> list[Path]:
    """Extract allowed result artifacts and validate required files.

    Legacy Kaggle archives included a broad `reports/figures_all/` diagnostic
    gallery. Those files are intentionally skipped; final paper figures are
    regenerated locally from the imported metrics and predictions.
    """

    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            if info.filename.startswith(SKIPPED_PREFIXES):
                continue
            relative_path = validate_archive_member(info.filename)
            destination = project_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as target:
                target.write(source.read())
            extracted.append(destination)

    missing = [path for path in REQUIRED_PATHS if not (project_root / path).exists()]
    if missing:
        raise FileNotFoundError(
            "Imported archive is incomplete; missing: " + ", ".join(missing)
        )
    return extracted


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archive",
        type=Path,
        help="Path to disinformation_results.zip downloaded from Kaggle.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root where reports/ and figures/ will be extracted.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    extracted = import_results_archive(args.archive.resolve(), args.project_root.resolve())
    logging.info("Imported %d files from %s", len(extracted), args.archive)


if __name__ == "__main__":
    main()
