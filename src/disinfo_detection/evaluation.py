"""Evaluation helpers for LIAR classification experiments."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def _optional_version(module_name: str) -> str:
    """Return an installed module version, or `unavailable` if it cannot load."""

    try:
        module = __import__(module_name)
    except Exception:
        return "unavailable"
    return str(getattr(module, "__version__", "unknown"))


def _git_sha() -> str:
    """Return the current git SHA, or `unknown` outside a git checkout."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def build_env_record(seed: int, device: str | Any, run_timestamp: str) -> dict[str, Any]:
    """Build a reproducibility fingerprint for an experiment run.

    Args:
        seed: Random seed used by the run.
        device: Device string or `torch.device` used for inference/training.
        run_timestamp: UTC timestamp captured by the training script.

    Returns:
        Dictionary suitable for embedding in metrics JSON files.
    """

    return {
        "git_sha": _git_sha(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": _optional_version("torch"),
        "transformers_version": _optional_version("transformers"),
        "device": str(device),
        "seed": int(seed),
        "run_timestamp": run_timestamp,
    }


def _native_scalar(value: Any) -> Any:
    """Convert pandas/numpy scalar values to JSON-friendly Python values."""

    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _float_list(values: Sequence[float] | np.ndarray | None) -> list[float] | None:
    """Convert a vector-like object to a list of floats."""

    if values is None:
        return None
    return [float(value) for value in list(values)]


def _label_name(label_id: int, label_names: Sequence[str]) -> str:
    """Map an integer label id to a human-readable label."""

    if 0 <= label_id < len(label_names):
        return str(label_names[label_id])
    return str(label_id)


def build_prediction_records(
    frame: pd.DataFrame,
    predictions: Sequence[int],
    probabilities: Sequence[Sequence[float]] | np.ndarray | None,
    logits: Sequence[Sequence[float]] | np.ndarray | None,
    label_names: Sequence[str],
    model_name: str,
    seed: int,
    split: str = "test",
) -> list[dict[str, Any]]:
    """Build per-example prediction records for downstream analysis.

    Args:
        frame: DataFrame aligned to the predictions.
        predictions: Predicted integer label ids.
        probabilities: Per-class probabilities aligned to `frame`.
        logits: Per-class logits/scores aligned to `frame`, or `None` for models
            that do not expose logits.
        label_names: Ordered label names.
        model_name: Stable model identifier for the artifact.
        seed: Random seed used by the run.
        split: Dataset split name.

    Returns:
        List of JSON-serializable prediction records.
    """

    if len(frame) != len(predictions):
        raise ValueError("frame and predictions must have the same length.")
    if probabilities is not None and len(probabilities) != len(frame):
        raise ValueError("probabilities and frame must have the same length.")
    if logits is not None and len(logits) != len(frame):
        raise ValueError("logits and frame must have the same length.")

    records: list[dict[str, Any]] = []
    probability_rows = probabilities if probabilities is not None else [None] * len(frame)
    logit_rows = logits if logits is not None else [None] * len(frame)

    for row_index, (_, row) in enumerate(frame.iterrows()):
        true_id = int(row["label_id"])
        pred_id = int(predictions[row_index])
        statement = row.get(
            "statement",
            row.get("statement_raw", row.get("statement_transformer", "")),
        )
        records.append(
            {
                "model": model_name,
                "seed": int(seed),
                "split": split,
                "row_index": int(row_index),
                "id": _native_scalar(row.get("id")),
                "statement": _native_scalar(statement),
                "label_id": true_id,
                "true_label": _label_name(true_id, label_names),
                "pred_label_id": pred_id,
                "pred_label": _label_name(pred_id, label_names),
                "correct": bool(pred_id == true_id),
                "speaker": _native_scalar(row.get("speaker")),
                "party": _native_scalar(row.get("party")),
                "job": _native_scalar(row.get("job")),
                "state": _native_scalar(row.get("state")),
                "subject": _native_scalar(row.get("subject")),
                "context": _native_scalar(row.get("context")),
                "probabilities": _float_list(probability_rows[row_index]),
                "logits": _float_list(logit_rows[row_index]),
            }
        )
    return records


def write_jsonl_records(records: list[dict[str, Any]], output_path: str | Path) -> None:
    """Write JSON-serializable records to a JSON Lines file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def compute_metrics(y_true: list[int], y_pred: list[int], label_names: list[str]) -> dict:
    """Compute core multi-class evaluation metrics.

    Args:
        y_true: Ground-truth label ids.
        y_pred: Predicted label ids.
        label_names: Ordered human-readable label names.

    Returns:
        Dictionary containing accuracy, macro-F1, per-class F1/precision/recall,
        the raw confusion matrix (list of lists, rows = true, cols = pred), and
        the sklearn classification report.
    """

    labels = list(range(len(label_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    per_class_f1 = {
        label_name: float(report[label_name]["f1-score"])
        for label_name in label_names
    }
    per_class_precision = {
        label_name: float(report[label_name]["precision"])
        for label_name in label_names
    }
    per_class_recall = {
        label_name: float(report[label_name]["recall"])
        for label_name in label_names
    }
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "confusion_matrix": matrix,
        "confusion_matrix_labels": list(label_names),
        "classification_report": report,
    }


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str],
    title: str,
    save_path: str | None = None,
) -> None:
    """Render a normalized confusion matrix heatmap.

    Args:
        y_true: Ground-truth label ids.
        y_pred: Predicted label ids.
        label_names: Ordered human-readable label names.
        title: Figure title.
        save_path: Optional output image path.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    matrix = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(8, 6), dpi=150)
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
    plt.close()


def compare_models(results: dict[str, dict], save_path: str | None = None) -> pd.DataFrame:
    """Build a model-comparison table and optional bar chart.

    Args:
        results: Mapping from model name to metric dictionary.
        save_path: Optional output path for a comparison figure.

    Returns:
        DataFrame sorted by macro-F1 descending.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    records = [
        {
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        }
        for model_name, metrics in results.items()
    ]
    frame = pd.DataFrame(records).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    if save_path is not None and not frame.empty:
        plt.figure(figsize=(8, 5), dpi=150)
        sns.barplot(data=frame, x="macro_f1", y="model", palette="colorblind")
        plt.title("Model Comparison by Macro-F1")
        plt.xlabel("Macro-F1")
        plt.ylabel("Model")
        plt.tight_layout()
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    return frame


def aggregate_seed_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed test-summary dicts into mean / std reports.

    Each input summary is expected to expose `test_macro_f1`, `test_accuracy`,
    `test_per_class_f1`, `test_per_class_precision`, `test_per_class_recall`,
    and `seed`. The aggregate keeps every per-seed value alongside the mean
    and population standard deviation so downstream code can build error bars
    or run paired statistical tests.
    """

    if not summaries:
        raise ValueError("aggregate_seed_summaries requires at least one summary")

    seeds = [int(summary.get("seed", index)) for index, summary in enumerate(summaries)]
    label_names = list(summaries[0].get("test_confusion_matrix_labels", []))

    def _mean_std(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "values": [float(value) for value in values],
            "mean": float(array.mean()),
            "std": float(array.std(ddof=0)),
        }

    def _aggregate_per_class(records: list[dict[str, float]]) -> dict[str, Any]:
        names = label_names or list(records[0].keys())
        per_seed = {name: [float(record.get(name, 0.0)) for record in records] for name in names}
        means = {name: float(np.mean(values)) for name, values in per_seed.items()}
        stds = {name: float(np.std(values, ddof=0)) for name, values in per_seed.items()}
        return {"per_seed": per_seed, "mean": means, "std": stds}

    macro_f1 = [float(summary["test_macro_f1"]) for summary in summaries]
    accuracy = [float(summary["test_accuracy"]) for summary in summaries]
    per_class_f1 = [summary["test_per_class_f1"] for summary in summaries]
    per_class_precision = [summary["test_per_class_precision"] for summary in summaries]
    per_class_recall = [summary["test_per_class_recall"] for summary in summaries]

    return {
        "num_seeds": len(summaries),
        "seeds": seeds,
        "test_macro_f1": _mean_std(macro_f1),
        "test_accuracy": _mean_std(accuracy),
        "test_per_class_f1": _aggregate_per_class(per_class_f1),
        "test_per_class_precision": _aggregate_per_class(per_class_precision),
        "test_per_class_recall": _aggregate_per_class(per_class_recall),
    }


def append_run_history(records: list[dict], output_path: str) -> pd.DataFrame:
    """Append experiment records to a CSV run-history log.

    Args:
        records: Run records to append.
        output_path: CSV path for the run-history file.

    Returns:
        Full run-history DataFrame after append.
    """

    history_path = Path(output_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    new_rows = pd.DataFrame(records)
    if history_path.exists():
        history = pd.read_csv(history_path)
        history = pd.concat([history, new_rows], ignore_index=True)
    else:
        history = new_rows
    history.to_csv(history_path, index=False)
    return history


def plot_training_history(history_frame: pd.DataFrame, save_path: str) -> None:
    """Plot transformer training curves from a history DataFrame.

    Args:
        history_frame: DataFrame containing epoch-wise losses and macro-F1.
        save_path: Output figure path.
    """

    import matplotlib.pyplot as plt

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4), dpi=150)
    plt.subplot(1, 2, 1)
    plt.plot(history_frame["epoch"], history_frame["train_loss"], marker="o", label="Train Loss")
    plt.plot(history_frame["epoch"], history_frame["val_loss"], marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Transformer Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_frame["epoch"], history_frame["val_macro_f1"], marker="o", color="tab:green")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Validation Macro-F1")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
