"""Generate the compact training-curves figure used by the paper."""

from __future__ import annotations

import argparse
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
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"
DEFAULT_SEED = 42
PLOT_SYSTEMS = ("transformer", "hybrid", "hybrid_textonly", "hybrid_leaky")
SYSTEM_LABELS = {
    "transformer": "RoBERTa text-only",
    "hybrid": "Hybrid corrected",
    "hybrid_textonly": "Hybrid text-only",
    "hybrid_leaky": "Hybrid leaky",
}
SYSTEM_COLORS = {
    "transformer": "#4C78A8",
    "hybrid": "#59A14F",
    "hybrid_textonly": "#B07AA1",
    "hybrid_leaky": "#E15759",
}
LOG_PATHS = {
    "transformer": Path("transformer_logs") / "training_log.csv",
    "hybrid": Path("hybrid_logs") / "training_log.csv",
    "hybrid_textonly": Path("hybrid_textonly_logs") / "training_log.csv",
    "hybrid_leaky": Path("hybrid_leaky_logs") / "training_log.csv",
}


def collect_training_logs(
    reports_dir: Path = REPORTS_DIR,
    systems: Sequence[str] = PLOT_SYSTEMS,
    seed: int | None = DEFAULT_SEED,
) -> dict[str, pd.DataFrame]:
    """Load per-epoch training logs for the requested systems.

    The current cleaned notebook snapshots seed-specific logs under
    `reports/seed_<N>/...`. Older archives only contain the final top-level
    `reports/*_logs/training_log.csv` files, so this helper falls back to that
    layout to keep existing result archives usable.
    """

    logs: dict[str, pd.DataFrame] = {}
    for system in systems:
        rel_path = LOG_PATHS[system]
        candidates = []
        if seed is not None:
            candidates.append(reports_dir / f"seed_{seed}" / rel_path)
        candidates.append(reports_dir / rel_path)

        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            searched = ", ".join(str(candidate) for candidate in candidates)
            raise FileNotFoundError(f"missing training log for {system}; searched {searched}")

        df = pd.read_csv(path)
        required_columns = {"epoch", "train_loss", "val_loss", "val_macro_f1"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        logs[system] = df
    return logs


def plot_training_curves(
    logs: Mapping[str, pd.DataFrame],
    output_dir: Path = FIGURES_DIR,
) -> dict[str, Path]:
    """Create the two-panel training dynamics figure and save PDF/PNG outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300, constrained_layout=True)
    loss_ax, f1_ax = axes

    for system in PLOT_SYSTEMS:
        df = logs[system]
        label = SYSTEM_LABELS[system]
        color = SYSTEM_COLORS[system]
        loss_ax.plot(
            df["epoch"],
            df["train_loss"],
            color=color,
            linewidth=1.8,
            label=f"{label} train",
        )
        loss_ax.plot(
            df["epoch"],
            df["val_loss"],
            color=color,
            linewidth=1.8,
            linestyle="--",
            label=f"{label} val",
        )
        f1_ax.plot(
            df["epoch"],
            df["val_macro_f1"],
            color=color,
            marker="o",
            linewidth=1.8,
            markersize=3.5,
            label=label,
        )

    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.set_title("Training and validation loss")
    loss_ax.grid(alpha=0.25)
    loss_ax.legend(
        ncol=2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
    )

    f1_ax.set_xlabel("Epoch")
    f1_ax.set_ylabel("Validation macro-F1")
    f1_ax.set_title("Validation macro-F1")
    f1_ax.grid(alpha=0.25)
    f1_ax.legend(frameon=False, loc="lower right")

    pdf_path = output_dir / "training_curves.pdf"
    png_path = output_dir / "training_curves.png"
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
        help="Path to the reports directory containing training logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Directory where PDF and PNG figure files will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Preferred seed-specific log directory to read before falling back "
            "to top-level logs."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        logs = collect_training_logs(args.reports_dir, seed=args.seed)
    except FileNotFoundError as exc:
        pdf_path = args.output_dir / "training_curves.pdf"
        png_path = args.output_dir / "training_curves.png"
        if pdf_path.exists() and png_path.exists():
            logging.warning("Skipping training-curve regeneration: %s", exc)
            return
        raise
    outputs = plot_training_curves(logs, args.output_dir)
    logging.info("Wrote %s", outputs["pdf"])
    logging.info("Wrote %s", outputs["png"])


if __name__ == "__main__":
    main()
