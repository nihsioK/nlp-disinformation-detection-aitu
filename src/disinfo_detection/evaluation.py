"""Evaluation helpers for LIAR classification experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def compute_metrics(y_true: list[int], y_pred: list[int], label_names: list[str]) -> dict:
    """Compute core multi-class evaluation metrics.

    Args:
        y_true: Ground-truth label ids.
        y_pred: Predicted label ids.
        label_names: Ordered human-readable label names.

    Returns:
        Dictionary containing accuracy, macro-F1, per-class F1, and the sklearn report.
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
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": per_class_f1,
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
