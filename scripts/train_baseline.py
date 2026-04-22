"""Train TF-IDF baseline models on processed LIAR data."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MATPLOTLIB_CACHE_DIR = REPO_ROOT / ".cache" / "matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
FONTCONFIG_CACHE_DIR = REPO_ROOT / ".cache" / "fontconfig"
FONTCONFIG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

from src.disinfo_detection.evaluation import (
    append_run_history,
    compare_models,
    compute_metrics,
    plot_confusion_matrix,
)
from src.disinfo_detection.models_baseline import TFIDFBaseline, load_dataset_config


logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

CLASSIFIER_TYPES = ["svm", "naive_bayes", "random_forest"]
ENABLE_FIGURES = os.environ.get("ENABLE_FIGURES", "0") == "1"


def load_processed_split(split_name: str, processed_dir: Path) -> pd.DataFrame:
    """Load a processed LIAR split from disk.

    Args:
        split_name: Dataset split name.
        processed_dir: Directory containing processed pickle files.

    Returns:
        Processed DataFrame for the requested split.

    Raises:
        FileNotFoundError: If the processed split does not exist.
    """

    split_path = processed_dir / f"{split_name}.pkl"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Processed split not found at {split_path}. Run scripts/preprocess.py first."
        )
    return pd.read_pickle(split_path)


def main() -> None:
    """Train and evaluate all configured TF-IDF baselines."""

    dataset_config = load_dataset_config()
    liar_cfg = dataset_config["liar"]
    processed_dir = Path(liar_cfg["processed_dir"])
    train_df = load_processed_split("train", processed_dir)
    valid_df = load_processed_split("valid", processed_dir)
    label_names = liar_cfg["label_names"]

    X_train = train_df["statement_clean"].tolist()
    y_train = train_df["label_id"].tolist()
    X_valid = valid_df["statement_clean"].tolist()
    y_valid = valid_df["label_id"].tolist()

    models_dir = Path("models")
    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    if ENABLE_FIGURES:
        figures_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now(timezone.utc).isoformat()
    summary_rows: list[dict[str, float | str]] = []
    metrics_by_model: dict[str, dict] = {}
    for classifier_type in CLASSIFIER_TYPES:
        logger.info("Training baseline model: %s", classifier_type)
        model = TFIDFBaseline(classifier_type=classifier_type)
        model.fit(X_train, y_train)
        predictions = model.predict(X_valid)
        metrics = compute_metrics(y_valid, predictions, label_names)
        metrics_by_model[classifier_type] = metrics
        model_path = models_dir / f"baseline_{classifier_type}.pkl"
        model.save(str(model_path))
        confusion_path = figures_dir / f"baseline_confusion_{classifier_type}.png"
        if ENABLE_FIGURES:
            plot_confusion_matrix(
                y_valid,
                predictions,
                label_names,
                title=f"{classifier_type.upper()} Confusion Matrix",
                save_path=str(confusion_path),
            )
        logger.info(
            "Finished %s with accuracy %.4f and macro-F1 %.4f",
            classifier_type,
            metrics["accuracy"],
            metrics["macro_f1"],
        )
        summary_rows.append(
            {
                "model": classifier_type,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "run_timestamp": run_timestamp,
                "model_path": str(model_path),
                "confusion_matrix_path": str(confusion_path) if ENABLE_FIGURES else "",
            }
        )

    summary_frame = pd.DataFrame(summary_rows)
    output_path = reports_dir / "baseline_results.csv"
    summary_frame[["model", "accuracy", "macro_f1"]].to_csv(output_path, index=False)
    if ENABLE_FIGURES:
        compare_models(metrics_by_model, save_path=str(figures_dir / "baseline_model_comparison.png"))
    append_run_history(summary_rows, str(reports_dir / "baseline_run_history.csv"))
    if not ENABLE_FIGURES:
        logger.info("Figure generation disabled. Set ENABLE_FIGURES=1 to render PNG artifacts.")
    logger.info("Saved baseline summary to %s", output_path)


if __name__ == "__main__":
    main()
