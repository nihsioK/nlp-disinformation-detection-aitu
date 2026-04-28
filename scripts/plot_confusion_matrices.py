"""Generate side-by-side row-normalised confusion matrices for the paper."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "disinfo_detection_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"
SEEDS = (42, 1337, 2024)
LABEL_NAMES = (
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true",
)
PLOT_SYSTEMS = ("hybrid_leaky", "hybrid")
SYSTEM_TITLES = {
    "hybrid_leaky": "Leaky (replicates prior work)",
    "hybrid": "Leakage-corrected",
}
METRIC_PATHS = {
    "hybrid_leaky": Path("hybrid_leaky_logs") / "hybrid_test_metrics.json",
    "hybrid": Path("hybrid_logs") / "hybrid_test_metrics.json",
}


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    """Return a row-normalised copy of a confusion matrix."""

    values = np.asarray(matrix, dtype=float)
    row_sums = values.sum(axis=1, keepdims=True)
    return np.divide(
        values,
        row_sums,
        out=np.zeros_like(values, dtype=float),
        where=row_sums != 0,
    )


def collect_confusion_matrices(
    reports_dir: Path = REPORTS_DIR,
    systems: Sequence[str] = PLOT_SYSTEMS,
    seeds: Sequence[int] = SEEDS,
) -> dict[str, dict[int, np.ndarray]]:
    """Read raw test confusion matrices from seed-scoped metrics files."""

    collected: dict[str, dict[int, np.ndarray]] = {}
    for system in systems:
        collected[system] = {}
        for seed in seeds:
            path = reports_dir / f"seed_{seed}" / METRIC_PATHS[system]
            if not path.exists():
                raise FileNotFoundError(f"missing metric file: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            labels = payload.get("test_confusion_matrix_labels")
            if labels != list(LABEL_NAMES):
                raise ValueError(f"unexpected confusion-matrix labels in {path}: {labels}")
            matrix = np.asarray(payload["test_confusion_matrix"], dtype=float)
            if matrix.shape != (len(LABEL_NAMES), len(LABEL_NAMES)):
                raise ValueError(f"unexpected confusion-matrix shape in {path}: {matrix.shape}")
            collected[system][int(seed)] = matrix
    return collected


def aggregate_confusion_matrices(
    collected: Mapping[str, Mapping[int, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Average raw matrices across seeds, then row-normalise each result."""

    aggregated: dict[str, np.ndarray] = {}
    for system, per_seed in collected.items():
        matrices = [np.asarray(matrix, dtype=float) for matrix in per_seed.values()]
        if not matrices:
            raise ValueError(f"no confusion matrices found for {system}")
        averaged_raw = np.mean(np.stack(matrices, axis=0), axis=0)
        aggregated[system] = row_normalize(averaged_raw)
    return aggregated


def plot_confusion_matrices(
    matrices: Mapping[str, np.ndarray],
    output_dir: Path = FIGURES_DIR,
) -> dict[str, Path]:
    """Plot the leaky and corrected row-normalised matrices side by side."""

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300, constrained_layout=True)
    image = None
    for ax, system in zip(axes, PLOT_SYSTEMS, strict=True):
        matrix = np.asarray(matrices[system], dtype=float)
        image = ax.imshow(matrix, vmin=0.0, vmax=0.6, cmap="Blues")
        ax.set_title(SYSTEM_TITLES[system])
        ax.set_xticks(np.arange(len(LABEL_NAMES)))
        ax.set_yticks(np.arange(len(LABEL_NAMES)))
        ax.set_xticklabels(LABEL_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(LABEL_NAMES)
        ax.set_xlabel("Predicted label")
        if ax is axes[0]:
            ax.set_ylabel("True label")
        else:
            ax.set_ylabel("")

        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                color = "white" if value >= 0.42 else "#1A1A1A"
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=color,
                )
            ax.add_patch(
                Rectangle(
                    (row_idx - 0.5, row_idx - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="#222222",
                    linewidth=1.5,
                )
            )

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", length=0)

    if image is None:
        raise ValueError("no matrices were plotted.")
    colorbar = fig.colorbar(image, ax=axes, shrink=0.9, pad=0.02)
    colorbar.set_label("Row-normalised share")

    pdf_path = output_dir / "confusion_matrices.pdf"
    png_path = output_dir / "confusion_matrices.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return {"pdf": pdf_path, "png": png_path}


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Path to the reports directory containing seed-scoped metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Directory where PDF and PNG figure files will be written.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    collected = collect_confusion_matrices(args.reports_dir)
    matrices = aggregate_confusion_matrices(collected)
    outputs = plot_confusion_matrices(matrices, args.output_dir)
    logging.info("Wrote %s", outputs["pdf"])
    logging.info("Wrote %s", outputs["png"])


if __name__ == "__main__":
    main()
