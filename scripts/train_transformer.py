"""Train the RoBERTa transformer baseline on processed LIAR data.

Behaviour:
- Class-weighted cross entropy (inverse-sqrt frequency) to mitigate LIAR
  class imbalance ('half-true' is ~3x 'pants-fire').
- Optional ordinal-aware loss (CE + squared EMD over softmax CDFs) for the
  ordered truthfulness scale; toggled in the YAML config.
- Label smoothing, gradient accumulation, warmup, and early stopping driven
  from `config/transformer.yaml`.
- TEST-split evaluation after training, with per-class F1 written to JSON.
- Multi-seed reporting: when `training.seeds` is a list, the full loop runs
  for each seed and an aggregate `*_test_metrics_multiseed.json` reports
  mean ± std over seeds.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.disinfo_detection.evaluation import build_env_record
from src.disinfo_detection.losses import build_loss_module
from src.disinfo_detection.models_baseline import load_dataset_config
from src.disinfo_detection.models_transformers import LIARDataset, RoBERTaClassifier, load_transformer_config
from src.disinfo_detection.training_utils import (
    compute_class_weights,
    evaluate_and_persist_test,
    load_processed_split,
    maybe_seeded_path,
    persist_training_history,
    resolve_device,
    resolve_sample_limit,
    resolve_seeds,
    run_epoch_loop,
    set_seed,
    setup_runtime_caches,
    write_seed_aggregate,
)


setup_runtime_caches(REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)
ENABLE_FIGURES = os.environ.get("ENABLE_FIGURES", "0") == "1"


def train_one_seed(
    seed: int,
    *,
    transformer_config: dict,
    dataset_config: dict,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: torch.device,
    logs_dir: Path,
    figures_dir: Path,
    predictions_dir: Path,
    output_dir: Path,
    is_multi_seed: bool,
) -> dict[str, Any]:
    """Run a single training-evaluation cycle for the supplied seed."""

    training_cfg = transformer_config["training"]
    model_cfg = transformer_config["model"]
    runtime_cfg = transformer_config["runtime"]
    paths_cfg = transformer_config["paths"]
    liar_cfg = dataset_config["liar"]

    set_seed(seed)
    logger.info("=== Seed %d ===", seed)

    num_labels = model_cfg["num_labels"]
    train_labels = train_df["label_id"].tolist()
    class_weights = compute_class_weights(train_labels, num_labels=num_labels)
    logger.info("Class weights (inverse-sqrt-frequency): %s", class_weights.tolist())

    label_smoothing = float(training_cfg.get("label_smoothing", 0.0))
    loss_module = build_loss_module(
        loss_cfg=training_cfg.get("loss"),
        num_classes=num_labels,
        class_weights=class_weights,
        label_smoothing=label_smoothing,
    )
    logger.info("Loss configuration: %s", loss_module.extra_repr())

    classifier = RoBERTaClassifier(
        model_name=model_cfg["name"],
        num_labels=num_labels,
        hidden_dropout_prob=model_cfg["hidden_dropout_prob"],
        attention_probs_dropout_prob=model_cfg.get(
            "attention_probs_dropout_prob", model_cfg["hidden_dropout_prob"]
        ),
        loss_module=loss_module,
    )
    classifier.model = classifier.model.to(device)

    train_dataset = LIARDataset(
        texts=train_df["statement_transformer"].tolist(),
        labels=train_labels,
        tokenizer=classifier.tokenizer,
        max_length=model_cfg["max_length"],
    )
    valid_dataset = LIARDataset(
        texts=valid_df["statement_transformer"].tolist(),
        labels=valid_df["label_id"].tolist(),
        tokenizer=classifier.tokenizer,
        max_length=model_cfg["max_length"],
    )
    test_dataset = LIARDataset(
        texts=test_df["statement_transformer"].tolist(),
        labels=test_df["label_id"].tolist(),
        tokenizer=classifier.tokenizer,
        max_length=model_cfg["max_length"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=runtime_cfg["num_workers"],
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=runtime_cfg["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=runtime_cfg["num_workers"],
    )

    gradient_accumulation_steps = int(training_cfg.get("gradient_accumulation_steps", 1))
    optimizer = AdamW(
        classifier.model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / max(gradient_accumulation_steps, 1))
    total_steps = optimizer_steps_per_epoch * training_cfg["epochs"]
    warmup_steps = int(total_steps * float(training_cfg["warmup_ratio"]))
    logger.info(
        "Schedule — epochs: %d, batches/epoch: %d, opt steps: %d, warmup: %d",
        training_cfg["epochs"],
        len(train_loader),
        total_steps,
        warmup_steps,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    run_timestamp = datetime.now(timezone.utc).isoformat()
    environment = build_env_record(seed=seed, device=device, run_timestamp=run_timestamp)
    checkpoint_path = maybe_seeded_path(paths_cfg["best_checkpoint"], seed, is_multi_seed)

    history_rows, best_macro_f1 = run_epoch_loop(
        trainer=classifier,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        training_cfg=training_cfg,
        checkpoint_path=checkpoint_path,
        seed=seed,
        run_timestamp=run_timestamp,
        logger=logger,
    )

    persist_training_history(
        history_rows,
        training_log_path=maybe_seeded_path(logs_dir / "training_log.csv", seed, is_multi_seed),
        run_history_path=logs_dir / "transformer_run_history.csv",
        figure_path=maybe_seeded_path(
            figures_dir / "transformer_training_curves.png", seed, is_multi_seed
        ),
        enable_figures=ENABLE_FIGURES,
        logger=logger,
        log_label="transformer",
    )

    return evaluate_and_persist_test(
        classifier,
        test_loader=test_loader,
        test_df=test_df,
        device=device,
        seed=seed,
        is_multi_seed=is_multi_seed,
        checkpoint_path=checkpoint_path,
        label_names=liar_cfg["label_names"],
        logs_dir=logs_dir,
        predictions_dir=predictions_dir,
        log_label="transformer",
        prediction_model_name="transformer",
        extra_summary_fields={"loss_config": training_cfg.get("loss")},
        best_val_macro_f1=best_macro_f1,
        run_timestamp=run_timestamp,
        environment=environment,
        logger=logger,
        on_after_load=lambda trainer: setattr(trainer, "model", trainer.model.to(device)),
    )


def main() -> None:
    """Train and evaluate the RoBERTa transformer baseline."""

    dataset_config = load_dataset_config()
    transformer_config = load_transformer_config()
    liar_cfg = dataset_config["liar"]
    processed_dir = Path(liar_cfg["processed_dir"])

    train_df = load_processed_split("train", processed_dir)
    valid_df = load_processed_split("valid", processed_dir)
    test_df = load_processed_split("test", processed_dir)

    runtime_cfg = transformer_config["runtime"]
    paths_cfg = transformer_config["paths"]
    training_cfg = transformer_config["training"]

    device = resolve_device()
    logger.info("Using device: %s", device)

    max_train_samples = resolve_sample_limit(runtime_cfg.get("max_train_samples"), "MAX_TRAIN_SAMPLES")
    max_valid_samples = resolve_sample_limit(runtime_cfg.get("max_valid_samples"), "MAX_VALID_SAMPLES")
    if max_train_samples is not None:
        train_df = train_df.head(max_train_samples).copy()
    if max_valid_samples is not None:
        valid_df = valid_df.head(max_valid_samples).copy()
    logger.info(
        "Transformer data sizes — train: %d, valid: %d, test: %d",
        len(train_df),
        len(valid_df),
        len(test_df),
    )

    reports_cfg = dataset_config.get("reports", {})
    output_dir = Path(paths_cfg["output_dir"])
    logs_dir = Path(paths_cfg["logs_dir"])
    figures_dir = Path(reports_cfg.get("figures_dir", "reports/figures"))
    predictions_dir = Path(reports_cfg.get("predictions_dir", "reports/predictions"))
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    if ENABLE_FIGURES:
        figures_dir.mkdir(parents=True, exist_ok=True)

    seeds = resolve_seeds(training_cfg)
    is_multi_seed = len(seeds) > 1
    logger.info("Training over %d seed(s): %s", len(seeds), seeds)

    summaries: list[dict[str, Any]] = []
    for seed in seeds:
        summary = train_one_seed(
            seed=seed,
            transformer_config=transformer_config,
            dataset_config=dataset_config,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            device=device,
            logs_dir=logs_dir,
            figures_dir=figures_dir,
            predictions_dir=predictions_dir,
            output_dir=output_dir,
            is_multi_seed=is_multi_seed,
        )
        summaries.append(summary)

    if is_multi_seed:
        write_seed_aggregate(
            summaries,
            logs_dir / "transformer_test_metrics_multiseed.json",
            extra_fields={"loss_config": training_cfg.get("loss")},
            logger=logger,
            log_label="transformer",
        )


if __name__ == "__main__":
    main()
