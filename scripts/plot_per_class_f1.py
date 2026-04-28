"""Generate a grouped per-class F1 bar chart for the paper."""

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
PLOT_SYSTEMS = ("transformer", "hybrid", "hybrid_leaky")
SYSTEM_LABELS = {
    "transformer": "RoBERTa text-only",
    "hybrid": "Hybrid corrected",
    "hybrid_leaky": "Hybrid leaky",
}
SYSTEM_COLORS = {
    "transformer": "#4C78A8",
    "hybrid": "#59A14F",
    "hybrid_leaky": "#E15759",
}
METRIC_PATHS = {
    "transformer": Path("transformer_logs") / "transformer_test_metrics.json",
    "hybrid": Path("hybrid_logs") / "hybrid_test_metrics.json",
    "hybrid_leaky": Path("hybrid_leaky_logs") / "hybrid_test_metrics.json",
}


def collect_per_class_f1(
    reports_dir: Path = REPORTS_DIR,
    systems: Sequence[str] = PLOT_SYSTEMS,
    seeds: Sequence[int] = SEEDS,
) -> dict[str, dict[int, dict[str, float]]]:
    """Read per-class F1 values from seed-scoped metric JSON files.

    Args:
        reports_dir: Root reports directory.
        systems: Stable system identifiers to include in the plot.
        seeds: Random seeds to aggregate.

    Returns:
        Nested mapping `system -> seed -> label -> f1`.

    Raises:
        FileNotFoundError: If any expected metric file is missing.
        KeyError: If a metric file lacks any required class value.
    """

    collected: dict[str, dict[int, dict[str, float]]] = {}
    for system in systems:
        collected[system] = {}
        for seed in seeds:
            path = reports_dir / f"seed_{seed}" / METRIC_PATHS[system]
            if not path.exists():
                raise FileNotFoundError(f"missing metric file: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            per_class = payload["test_per_class_f1"]
            collected[system][int(seed)] = {
                label: float(per_class[label])
                for label in LABEL_NAMES
            }
    return collected


def aggregate_per_class_f1(
    collected: Mapping[str, Mapping[int, Mapping[str, float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute mean and sample standard deviation per class across seeds."""

    aggregated: dict[str, dict[str, dict[str, float]]] = {}
    for system, per_seed in collected.items():
        aggregated[system] = {}
        for label in LABEL_NAMES:
            values = np.array([float(seed_values[label]) for seed_values in per_seed.values()])
            aggregated[system][label] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
    return aggregated


def plot_grouped_per_class_f1(
    aggregated: Mapping[str, Mapping[str, Mapping[str, float]]],
    output_dir: Path = FIGURES_DIR,
) -> dict[str, Path]:
    """Create the grouped bar chart and save PDF/PNG outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    x_positions = np.arange(len(LABEL_NAMES))
    bar_width = 0.24
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)

    offsets = np.linspace(
        -bar_width,
        bar_width,
        num=len(PLOT_SYSTEMS),
    )
    for offset, system in zip(offsets, PLOT_SYSTEMS, strict=True):
        means = [aggregated[system][label]["mean"] for label in LABEL_NAMES]
        stds = [aggregated[system][label]["std"] for label in LABEL_NAMES]
        ax.bar(
            x_positions + offset,
            means,
            bar_width,
            label=SYSTEM_LABELS[system],
            color=SYSTEM_COLORS[system],
            edgecolor="white",
            linewidth=0.5,
            yerr=stds,
            capsize=3,
            error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": "#333333"},
        )

    ax.set_ylabel("F1")
    ax.set_ylim(0.0, 0.7)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(LABEL_NAMES, rotation=20, ha="right")
    ax.yaxis.grid(True, color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=len(PLOT_SYSTEMS),
        frameon=False,
        handlelength=1.2,
        columnspacing=1.4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    pdf_path = output_dir / "per_class_f1_grouped.pdf"
    png_path = output_dir / "per_class_f1_grouped.png"
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
    collected = collect_per_class_f1(args.reports_dir)
    aggregated = aggregate_per_class_f1(collected)
    outputs = plot_grouped_per_class_f1(aggregated, args.output_dir)
    logging.info("Wrote %s", outputs["pdf"])
    logging.info("Wrote %s", outputs["png"])


if __name__ == "__main__":
    main()
