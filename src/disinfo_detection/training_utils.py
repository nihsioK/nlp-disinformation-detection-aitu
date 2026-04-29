"""Shared utilities for the LIAR training scripts.

Centralises plumbing that was previously duplicated across
`scripts/train_baseline.py`, `scripts/train_transformer.py`, and
`scripts/train_hybrid.py`:

- random seeding and device resolution,
- inverse-sqrt-frequency class weights,
- processed-split loading,
- multi-seed plumbing (seed list, suffixed paths, aggregate JSON),
- runtime cache configuration so matplotlib/fontconfig stay inside the repo,
- the per-epoch training loop with checkpointing and early stopping.

Anything that is genuinely script-specific (model construction, optimizer
parameter groups, prediction-record naming) stays in the scripts.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import torch

from src.disinfo_detection.evaluation import (
    aggregate_seed_summaries,
    append_run_history,
    build_prediction_records,
    plot_training_history,
    write_jsonl_records,
)


def setup_runtime_caches(repo_root: Path) -> None:
    """Point matplotlib and fontconfig at an in-repo cache directory."""

    cache_dir = repo_root / ".cache"
    matplotlib_cache = cache_dir / "matplotlib"
    fontconfig_cache = cache_dir / "fontconfig"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    fontconfig_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


def resolve_device() -> torch.device:
    """Pick the best available device (cuda → mps → cpu)."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_processed_split(split_name: str, processed_dir: Path) -> pd.DataFrame:
    """Load a processed LIAR split from `<processed_dir>/<split>.pkl`."""

    split_path = processed_dir / f"{split_name}.pkl"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Processed split not found at {split_path}. Run scripts/preprocess.py first."
        )
    return pd.read_pickle(split_path)


def compute_class_weights(labels: list[int], num_labels: int) -> torch.Tensor:
    """Inverse-sqrt-frequency class weights.

    Full inverse-frequency overshoots toward the rarest class on LIAR
    (`pants-fire`); inverse-sqrt is the common compromise used in text
    classification.
    """

    counts = Counter(labels)
    total = sum(counts.values())
    weights: list[float] = []
    for class_id in range(num_labels):
        count = max(counts.get(class_id, 0), 1)
        weights.append(math.sqrt(total / (num_labels * count)))
    return torch.tensor(weights, dtype=torch.float32)


def seeded_path(path: str | Path, seed: int) -> Path:
    """Return `path` with `_seed{N}` inserted before the extension."""

    p = Path(path)
    return p.with_name(f"{p.stem}_seed{seed}{p.suffix}")


def maybe_seeded_path(path: str | Path, seed: int, multi_seed: bool) -> Path:
    """Return a seeded path when `multi_seed` is true, else the bare path."""

    return seeded_path(path, seed) if multi_seed else Path(path)


def resolve_seeds(training_cfg: dict) -> list[int]:
    """Resolve `training.seeds` (list) or fall back to `training.seed`."""

    seeds_value = training_cfg.get("seeds")
    if seeds_value:
        return [int(seed) for seed in seeds_value]
    return [int(training_cfg["seed"])]


def resolve_sample_limit(config_value, env_name: str) -> int | None:
    """Resolve a sample-size limit, allowing an env-var override."""

    env_value = os.environ.get(env_name)
    if env_value is not None:
        return int(env_value)
    if config_value in (None, "", 0):
        return None
    return int(config_value)


def write_seed_aggregate(
    summaries: list[dict[str, Any]],
    output_path: Path,
    *,
    extra_fields: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
    log_label: str = "model",
) -> dict[str, Any]:
    """Aggregate per-seed summaries to mean ± std and persist to JSON."""

    aggregate = aggregate_seed_summaries(summaries)
    payload: dict[str, Any] = {"aggregate": aggregate, "per_seed": summaries}
    if extra_fields:
        payload.update(extra_fields)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    if logger is not None:
        logger.info(
            "Aggregated %d seeds — TEST macro-F1 %.4f ± %.4f, accuracy %.4f ± %.4f",
            aggregate["num_seeds"],
            aggregate["test_macro_f1"]["mean"],
            aggregate["test_macro_f1"]["std"],
            aggregate["test_accuracy"]["mean"],
            aggregate["test_accuracy"]["std"],
        )
        logger.info("Saved %s multi-seed metrics to %s", log_label, output_path)
    return aggregate


def run_epoch_loop(
    *,
    trainer: Any,
    train_loader: Iterable,
    valid_loader: Iterable,
    optimizer,
    scheduler,
    device: torch.device,
    training_cfg: dict,
    checkpoint_path: Path,
    seed: int,
    run_timestamp: str,
    logger: logging.Logger,
    extra_train_kwargs: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Run the multi-epoch training loop with checkpointing and early stopping.

    `trainer` must expose `train_epoch(...)`, `evaluate(...)`, and `save(path)`
    in the same shape as `RoBERTaClassifier` and `HybridTrainer`.
    """

    history_rows: list[dict[str, Any]] = []
    best_macro_f1 = float("-inf")
    epochs_without_improvement = 0
    patience = int(training_cfg.get("early_stopping_patience", 0))
    train_kwargs = dict(extra_train_kwargs or {})

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, training_cfg["epochs"] + 1):
        logger.info("Seed %d — Starting epoch %d/%d", seed, epoch, training_cfg["epochs"])
        train_loss = trainer.train_epoch(
            train_loader,
            optimizer,
            scheduler,
            device,
            gradient_clip=float(training_cfg["gradient_clip"]),
            gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 1)),
            log_every_steps=int(training_cfg.get("log_every_steps", 0)),
            logger=logger,
            **train_kwargs,
        )
        validation = trainer.evaluate(valid_loader, device)
        history_rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": validation["val_loss"],
                "val_macro_f1": validation["macro_f1"],
                "val_accuracy": validation["accuracy"],
                "device": str(device),
                "run_timestamp": run_timestamp,
            }
        )
        logger.info(
            "Seed %d — Epoch %d/%d — train %.4f — val loss %.4f — val macro-F1 %.4f — val acc %.4f",
            seed,
            epoch,
            training_cfg["epochs"],
            train_loss,
            validation["val_loss"],
            validation["macro_f1"],
            validation["accuracy"],
        )
        if validation["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation["macro_f1"]
            epochs_without_improvement = 0
            trainer.save(str(checkpoint_path))
            logger.info("Saved new best checkpoint to %s", checkpoint_path)
        else:
            epochs_without_improvement += 1
            logger.info(
                "Seed %d — No improvement in val macro-F1 (%d/%d)",
                seed,
                epochs_without_improvement,
                patience,
            )
            if patience and epochs_without_improvement >= patience:
                logger.info("Seed %d — Early stopping triggered after epoch %d", seed, epoch)
                break

    return history_rows, best_macro_f1


def evaluate_and_persist_test(
    trainer: Any,
    *,
    test_loader: Iterable,
    test_df: pd.DataFrame,
    device: torch.device,
    seed: int,
    is_multi_seed: bool,
    checkpoint_path: Path,
    label_names: list[str],
    logs_dir: Path,
    predictions_dir: Path,
    log_label: str,
    prediction_model_name: str,
    extra_summary_fields: dict[str, Any] | None = None,
    best_val_macro_f1: float,
    run_timestamp: str,
    environment: dict[str, Any],
    logger: logging.Logger,
    on_after_load: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    """Reload best checkpoint, evaluate on TEST, persist predictions and metrics.

    `on_after_load` lets scripts re-pin a model to a device after a CPU-side
    `load_state_dict` (transformer/hybrid both need this).
    """

    if checkpoint_path.exists():
        trainer.load(str(checkpoint_path))
        if on_after_load is not None:
            on_after_load(trainer)
    test_metrics = trainer.evaluate(test_loader, device, return_outputs=True)
    logger.info(
        "Seed %d — TEST — macro-F1 %.4f — accuracy %.4f",
        seed,
        test_metrics["macro_f1"],
        test_metrics["accuracy"],
    )

    prediction_records = build_prediction_records(
        frame=test_df,
        predictions=test_metrics["predictions"],
        probabilities=test_metrics["probabilities"],
        logits=test_metrics["logits"],
        label_names=label_names,
        model_name=prediction_model_name,
        seed=seed,
        split="test",
    )
    prediction_path = maybe_seeded_path(
        predictions_dir / f"{prediction_model_name}_test_predictions.jsonl",
        seed,
        is_multi_seed,
    )
    write_jsonl_records(prediction_records, prediction_path)
    logger.info("Saved %s predictions to %s", log_label, prediction_path)

    test_summary: dict[str, Any] = {
        "best_val_macro_f1": best_val_macro_f1,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_per_class_f1": test_metrics["per_class_f1"],
        "test_per_class_precision": test_metrics["per_class_precision"],
        "test_per_class_recall": test_metrics["per_class_recall"],
        "test_confusion_matrix": test_metrics["confusion_matrix"],
        "test_confusion_matrix_labels": test_metrics["confusion_matrix_labels"],
        "seed": seed,
        "run_timestamp": run_timestamp,
        "device": str(device),
        "environment": environment,
    }
    if extra_summary_fields:
        test_summary.update(extra_summary_fields)

    test_json_path = maybe_seeded_path(
        logs_dir / f"{log_label}_test_metrics.json", seed, is_multi_seed
    )
    test_json_path.parent.mkdir(parents=True, exist_ok=True)
    with test_json_path.open("w", encoding="utf-8") as fp:
        json.dump(test_summary, fp, indent=2)
    logger.info("Saved %s test metrics to %s", log_label, test_json_path)
    return test_summary


def persist_training_history(
    history_rows: list[dict[str, Any]],
    *,
    training_log_path: Path,
    run_history_path: Path,
    figure_path: Path | None,
    enable_figures: bool,
    logger: logging.Logger | None,
    log_label: str,
) -> pd.DataFrame:
    """Write per-epoch history to CSV, append to run history, plot if enabled."""

    history_frame = pd.DataFrame(history_rows)
    training_log_path.parent.mkdir(parents=True, exist_ok=True)
    history_frame.to_csv(training_log_path, index=False)
    if enable_figures and figure_path is not None:
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        plot_training_history(history_frame, str(figure_path))
    append_run_history(history_rows, str(run_history_path))
    if logger is not None:
        logger.info("Saved %s training log to %s", log_label, training_log_path)
    return history_frame


__all__ = [
    "compute_class_weights",
    "evaluate_and_persist_test",
    "load_processed_split",
    "maybe_seeded_path",
    "persist_training_history",
    "resolve_device",
    "resolve_sample_limit",
    "resolve_seeds",
    "run_epoch_loop",
    "seeded_path",
    "set_seed",
    "setup_runtime_caches",
    "write_seed_aggregate",
]
