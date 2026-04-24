"""Train TF-IDF baseline models on processed LIAR data.

Changes:
- Evaluates on BOTH valid (for hyperparameter selection) and test (for reporting).
- Saves metrics as JSON (per-class F1, confusion matrix path) in addition to CSV.
- Logs the exact TF-IDF vocabulary size, which varied silently between configs.
"""

from __future__ import annotations

import json
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
    build_env_record,
    build_prediction_records,
    compare_models,
    compute_metrics,
    plot_confusion_matrix,
    write_jsonl_records,
)
from src.disinfo_detection.models_baseline import (
    TFIDFBaseline,
    load_baseline_config,
    load_dataset_config,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

CLASSIFIER_TYPES = ["svm", "naive_bayes", "random_forest"]
ENABLE_FIGURES = os.environ.get("ENABLE_FIGURES", "0") == "1"


def load_processed_split(split_name: str, processed_dir: Path) -> pd.DataFrame:
    split_path = processed_dir / f"{split_name}.pkl"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Processed split not found at {split_path}. Run scripts/preprocess.py first."
        )
    return pd.read_pickle(split_path)


def main() -> None:
    """Train and evaluate all configured TF-IDF baselines on LIAR valid + test."""

    dataset_config = load_dataset_config()
    baseline_config = load_baseline_config()
    liar_cfg = dataset_config["liar"]
    processed_dir = Path(liar_cfg["processed_dir"])
    train_df = load_processed_split("train", processed_dir)
    valid_df = load_processed_split("valid", processed_dir)
    test_df = load_processed_split("test", processed_dir)
    label_names = liar_cfg["label_names"]

    X_train = train_df["statement_clean"].tolist()
    y_train = train_df["label_id"].tolist()
    X_valid = valid_df["statement_clean"].tolist()
    y_valid = valid_df["label_id"].tolist()
    X_test = test_df["statement_clean"].tolist()
    y_test = test_df["label_id"].tolist()

    models_dir = Path("models")
    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"
    predictions_dir = reports_dir / "predictions"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    if ENABLE_FIGURES:
        figures_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now(timezone.utc).isoformat()
    run_seed = int(baseline_config.get("random_forest", {}).get("random_state", 42))
    environment = build_env_record(seed=run_seed, device="cpu", run_timestamp=run_timestamp)
    summary_rows: list[dict[str, float | str]] = []
    metrics_by_model: dict[str, dict] = {}
    detailed_results: dict[str, dict] = {}

    for classifier_type in CLASSIFIER_TYPES:
        logger.info("Training baseline model: %s", classifier_type)
        model = TFIDFBaseline(classifier_type=classifier_type)
        model.fit(X_train, y_train)

        vocab_size = len(model.pipeline.named_steps["tfidf"].vocabulary_)
        logger.info("%s — TF-IDF vocabulary size: %d", classifier_type, vocab_size)

        valid_predictions = model.predict(X_valid)
        valid_metrics = compute_metrics(y_valid, valid_predictions, label_names)
        test_predictions = model.predict(X_test)
        test_probabilities = model.predict_proba(X_test)
        test_metrics = compute_metrics(y_test, test_predictions, label_names)
        metrics_by_model[classifier_type] = test_metrics

        model_path = models_dir / f"baseline_{classifier_type}.pkl"
        model.save(str(model_path))

        confusion_path = figures_dir / f"baseline_confusion_{classifier_type}.png"
        if ENABLE_FIGURES:
            plot_confusion_matrix(
                y_test,
                test_predictions,
                label_names,
                title=f"{classifier_type.upper()} (TEST) Confusion Matrix",
                save_path=str(confusion_path),
            )
        logger.info(
            "%s — valid acc %.4f / macro-F1 %.4f ; test acc %.4f / macro-F1 %.4f",
            classifier_type,
            valid_metrics["accuracy"],
            valid_metrics["macro_f1"],
            test_metrics["accuracy"],
            test_metrics["macro_f1"],
        )
        prediction_records = build_prediction_records(
            frame=test_df,
            predictions=test_predictions,
            probabilities=test_probabilities,
            logits=None,
            label_names=label_names,
            model_name=f"baseline_{classifier_type}",
            seed=run_seed,
            split="test",
        )
        prediction_path = predictions_dir / f"baseline_{classifier_type}_test_predictions.jsonl"
        write_jsonl_records(prediction_records, prediction_path)
        logger.info("Saved baseline predictions to %s", prediction_path)

        summary_rows.append(
            {
                "model": classifier_type,
                "vocab_size": vocab_size,
                "valid_accuracy": valid_metrics["accuracy"],
                "valid_macro_f1": valid_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "run_timestamp": run_timestamp,
                "model_path": str(model_path),
                "confusion_matrix_path": str(confusion_path) if ENABLE_FIGURES else "",
            }
        )
        detailed_results[classifier_type] = {
            "valid": valid_metrics,
            "test": test_metrics,
            "vocab_size": vocab_size,
        }

    summary_frame = pd.DataFrame(summary_rows)
    output_csv = reports_dir / "baseline_results.csv"
    summary_frame[
        ["model", "vocab_size", "valid_accuracy", "valid_macro_f1", "test_accuracy", "test_macro_f1"]
    ].to_csv(output_csv, index=False)

    # Detailed JSON with per-class F1/precision/recall and confusion matrix
    # for thesis tables. classification_report is not compact enough for the
    # report, so we serialize only the fields the thesis tables consume.
    with (reports_dir / "baseline_detailed_metrics.json").open("w", encoding="utf-8") as fp:
        def _slim(split_metrics: dict) -> dict:
            return {
                "accuracy": split_metrics["accuracy"],
                "macro_f1": split_metrics["macro_f1"],
                "per_class_f1": split_metrics["per_class_f1"],
                "per_class_precision": split_metrics["per_class_precision"],
                "per_class_recall": split_metrics["per_class_recall"],
                "confusion_matrix": split_metrics["confusion_matrix"],
                "confusion_matrix_labels": split_metrics["confusion_matrix_labels"],
            }

        compact = {
            key: {
                "valid": _slim(value["valid"]),
                "test": _slim(value["test"]),
                "vocab_size": value["vocab_size"],
            }
            for key, value in detailed_results.items()
        }
        compact["_environment"] = environment
        json.dump(compact, fp, indent=2)

    if ENABLE_FIGURES:
        compare_models(metrics_by_model, save_path=str(figures_dir / "baseline_model_comparison.png"))
    append_run_history(summary_rows, str(reports_dir / "baseline_run_history.csv"))
    if not ENABLE_FIGURES:
        logger.info("Figure generation disabled. Set ENABLE_FIGURES=1 to render PNG artifacts.")
    logger.info("Saved baseline summary to %s", output_csv)


if __name__ == "__main__":
    main()
