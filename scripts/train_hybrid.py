"""Train the hybrid RoBERTa + metadata classifier on processed LIAR data.

Mirrors `scripts/train_transformer.py` for an apples-to-apples comparison:
- Same class-weighted CE (inverse-sqrt frequency).
- Same label smoothing, warmup, early stopping, gradient clipping.
- Same TEST-split reporting for the thesis.
- Same per-class F1 logging.

Hybrid-specific behaviour:
- Two optimizer param groups (encoder vs. head) with separate LRs.
- Aligned metadata tensors packed into each batch via `HybridLIARDataset`.
- `model.use_metadata: false` degenerates into a text-only RoBERTa run for the
  RQ2 ablation without touching code.
- `metadata.leakage_corrected: true` builds the dense feature matrix from
  credibility columns that exclude the row's own verdict (defensible thesis
  default; flip to `false` for the prior-art leaky comparison).
- `training.seeds: [...]` repeats the full training loop for every listed
  seed and writes both per-seed and mean ± std aggregate metrics.
- `training.loss` selects between plain weighted CE and the ordinal-aware
  CE + squared-EMD blend.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MATPLOTLIB_CACHE_DIR = REPO_ROOT / ".cache" / "matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
FONTCONFIG_CACHE_DIR = REPO_ROOT / ".cache" / "fontconfig"
FONTCONFIG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

from src.disinfo_detection.datasets_hybrid import HybridLIARDataset
from src.disinfo_detection.evaluation import (
    aggregate_seed_summaries,
    append_run_history,
    build_env_record,
    build_prediction_records,
    plot_training_history,
    write_jsonl_records,
)
from src.disinfo_detection.losses import build_loss_module
from src.disinfo_detection.metadata_features import MetadataSpec, describe_feature_layout
from src.disinfo_detection.models_baseline import load_dataset_config
from src.disinfo_detection.models_hybrid import HybridClassifier, HybridTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)
ENABLE_FIGURES = os.environ.get("ENABLE_FIGURES", "0") == "1"


def load_hybrid_config(config_path: str | None = None) -> dict:
    # Allow overriding the config path from the environment so the same
    # script can run the hybrid or text-only ablation without code changes.
    resolved_path = config_path or os.environ.get("HYBRID_CONFIG") or "config/hybrid.yaml"
    with Path(resolved_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _resolve_sample_limit(config_value, env_name: str) -> int | None:
    env_value = os.environ.get(env_name)
    if env_value is not None:
        return int(env_value)
    if config_value in (None, "", 0):
        return None
    return int(config_value)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_processed_split(split_name: str, processed_dir: Path) -> pd.DataFrame:
    split_path = processed_dir / f"{split_name}.pkl"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Processed split not found at {split_path}. Run scripts/preprocess.py first."
        )
    return pd.read_pickle(split_path)


def compute_class_weights(labels: list[int], num_labels: int) -> torch.Tensor:
    counts = Counter(labels)
    total = sum(counts.values())
    weights = []
    for class_id in range(num_labels):
        count = max(counts.get(class_id, 0), 1)
        weights.append(math.sqrt(total / (num_labels * count)))
    return torch.tensor(weights, dtype=torch.float32)


def build_metadata_spec(config: dict) -> MetadataSpec:
    metadata_cfg = config.get("metadata", {}) or {}
    categorical_fields = tuple(
        metadata_cfg.get("categorical_fields") or MetadataSpec().categorical_fields
    )
    raw_buckets = metadata_cfg.get("num_buckets", MetadataSpec().num_buckets)
    if isinstance(raw_buckets, dict):
        num_buckets: int | dict[str, int] = {key: int(value) for key, value in raw_buckets.items()}
    else:
        num_buckets = int(raw_buckets)
    leakage_corrected = bool(metadata_cfg.get("leakage_corrected", False))
    return MetadataSpec(
        categorical_fields=categorical_fields,
        num_buckets=num_buckets,
        leakage_corrected=leakage_corrected,
    )


def build_param_groups(
    model: HybridClassifier,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
) -> list[dict]:
    """Split parameters into encoder vs. head groups with different LRs."""

    encoder_params = list(model.encoder.parameters())
    head_params = [param for name, param in model.named_parameters() if not name.startswith("encoder.")]
    return [
        {"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
    ]


def _seeded_path(path: str | Path, seed: int) -> Path:
    """Return `path` with `_seed{N}` inserted before the extension."""

    p = Path(path)
    return p.with_name(f"{p.stem}_seed{seed}{p.suffix}")


def _resolve_seeds(training_cfg: dict) -> list[int]:
    seeds_value = training_cfg.get("seeds")
    if seeds_value:
        return [int(seed) for seed in seeds_value]
    return [int(training_cfg["seed"])]


def train_one_seed(
    seed: int,
    *,
    hybrid_config: dict,
    dataset_config: dict,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metadata_spec: MetadataSpec,
    device: torch.device,
    logs_dir: Path,
    figures_dir: Path,
    predictions_dir: Path,
    output_dir: Path,
    is_multi_seed: bool,
) -> dict[str, Any]:
    """Run a single training-evaluation cycle for the supplied seed."""

    training_cfg = hybrid_config["training"]
    model_cfg = hybrid_config["model"]
    runtime_cfg = hybrid_config["runtime"]
    paths_cfg = hybrid_config["paths"]
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

    model = HybridClassifier(
        model_name=model_cfg["name"],
        num_labels=num_labels,
        hidden_dropout_prob=model_cfg["hidden_dropout_prob"],
        attention_probs_dropout_prob=model_cfg.get(
            "attention_probs_dropout_prob", model_cfg["hidden_dropout_prob"]
        ),
        use_metadata=bool(model_cfg.get("use_metadata", True)),
        metadata_spec=metadata_spec,
        metadata_output_dim=int(model_cfg.get("metadata_output_dim", 64)),
        metadata_embedding_dim=int(model_cfg.get("metadata_embedding_dim", 16)),
        fusion_hidden_dim=int(model_cfg.get("fusion_hidden_dim", 128)),
        fusion_dropout=float(model_cfg.get("fusion_dropout", 0.2)),
    )
    model = model.to(device)

    trainer = HybridTrainer(
        model=model,
        label_smoothing=label_smoothing,
        class_weights=class_weights,
        loss_module=loss_module,
    )

    include_metadata = bool(model_cfg.get("use_metadata", True))
    train_dataset = HybridLIARDataset(
        df=train_df,
        tokenizer=model.tokenizer,
        max_length=model_cfg["max_length"],
        metadata_spec=metadata_spec,
        include_metadata=include_metadata,
    )
    valid_dataset = HybridLIARDataset(
        df=valid_df,
        tokenizer=model.tokenizer,
        max_length=model_cfg["max_length"],
        metadata_spec=metadata_spec,
        include_metadata=include_metadata,
    )
    test_dataset = HybridLIARDataset(
        df=test_df,
        tokenizer=model.tokenizer,
        max_length=model_cfg["max_length"],
        metadata_spec=metadata_spec,
        include_metadata=include_metadata,
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
    encoder_lr = float(training_cfg.get("encoder_learning_rate", training_cfg.get("learning_rate", 1e-5)))
    head_lr = float(training_cfg.get("head_learning_rate", encoder_lr))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))
    optimizer = AdamW(
        build_param_groups(model, encoder_lr=encoder_lr, head_lr=head_lr, weight_decay=weight_decay)
    )

    optimizer_steps_per_epoch = math.ceil(len(train_loader) / max(gradient_accumulation_steps, 1))
    total_steps = optimizer_steps_per_epoch * training_cfg["epochs"]
    warmup_steps = int(total_steps * float(training_cfg["warmup_ratio"]))
    logger.info(
        "Schedule — epochs: %d, batches/epoch: %d, opt steps: %d, warmup: %d, encoder_lr=%.2e, head_lr=%.2e",
        training_cfg["epochs"],
        len(train_loader),
        total_steps,
        warmup_steps,
        encoder_lr,
        head_lr,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_macro_f1 = float("-inf")
    epochs_without_improvement = 0
    patience = int(training_cfg.get("early_stopping_patience", 0))
    run_timestamp = datetime.now(timezone.utc).isoformat()
    environment = build_env_record(
        seed=seed,
        device=device,
        run_timestamp=run_timestamp,
    )
    history_rows: list[dict[str, float | int | str]] = []

    checkpoint_path = (
        _seeded_path(paths_cfg["best_checkpoint"], seed)
        if is_multi_seed
        else Path(paths_cfg["best_checkpoint"])
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, training_cfg["epochs"] + 1):
        logger.info("Seed %d — Starting epoch %d/%d", seed, epoch, training_cfg["epochs"])
        train_loss = trainer.train_epoch(
            train_loader,
            optimizer,
            scheduler,
            device,
            gradient_clip=float(training_cfg["gradient_clip"]),
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_every_steps=int(training_cfg.get("log_every_steps", 0)),
            logger=logger,
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

    history_frame = pd.DataFrame(history_rows)
    training_log_path = (
        _seeded_path(logs_dir / "training_log.csv", seed)
        if is_multi_seed
        else logs_dir / "training_log.csv"
    )
    history_frame.to_csv(training_log_path, index=False)
    if ENABLE_FIGURES:
        figure_path = (
            _seeded_path(figures_dir / "hybrid_training_curves.png", seed)
            if is_multi_seed
            else figures_dir / "hybrid_training_curves.png"
        )
        plot_training_history(history_frame, str(figure_path))
    append_run_history(history_rows, str(logs_dir / "hybrid_run_history.csv"))
    logger.info("Saved hybrid training log to %s", training_log_path)

    if checkpoint_path.exists():
        trainer.load(str(checkpoint_path))
        model.to(device)
    test_metrics = trainer.evaluate(test_loader, device, return_outputs=True)
    logger.info(
        "Seed %d — TEST — macro-F1 %.4f — accuracy %.4f",
        seed,
        test_metrics["macro_f1"],
        test_metrics["accuracy"],
    )

    if not include_metadata:
        prediction_model_name = "hybrid_textonly"
    elif metadata_spec.leakage_corrected:
        prediction_model_name = "hybrid"
    else:
        prediction_model_name = "hybrid_leaky"
    prediction_records = build_prediction_records(
        frame=test_df,
        predictions=test_metrics["predictions"],
        probabilities=test_metrics["probabilities"],
        logits=test_metrics["logits"],
        label_names=liar_cfg["label_names"],
        model_name=prediction_model_name,
        seed=seed,
        split="test",
    )
    prediction_path = predictions_dir / f"{prediction_model_name}_test_predictions"
    prediction_path = (
        prediction_path.with_name(f"{prediction_path.name}_seed{seed}.jsonl")
        if is_multi_seed
        else prediction_path.with_suffix(".jsonl")
    )
    write_jsonl_records(prediction_records, prediction_path)
    logger.info("Saved hybrid predictions to %s", prediction_path)

    test_summary: dict[str, Any] = {
        "best_val_macro_f1": best_macro_f1,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_per_class_f1": test_metrics["per_class_f1"],
        "test_per_class_precision": test_metrics["per_class_precision"],
        "test_per_class_recall": test_metrics["per_class_recall"],
        "test_confusion_matrix": test_metrics["confusion_matrix"],
        "test_confusion_matrix_labels": test_metrics["confusion_matrix_labels"],
        "use_metadata": include_metadata,
        "leakage_corrected": metadata_spec.leakage_corrected,
        "loss_config": training_cfg.get("loss"),
        "field_bucket_sizes": dict(
            zip(metadata_spec.categorical_fields, metadata_spec.field_bucket_sizes)
        ),
        "metadata_output_dim": int(model_cfg.get("metadata_output_dim", 64)),
        "seed": seed,
        "run_timestamp": run_timestamp,
        "device": str(device),
        "environment": environment,
    }
    test_json_path = (
        _seeded_path(logs_dir / "hybrid_test_metrics.json", seed)
        if is_multi_seed
        else logs_dir / "hybrid_test_metrics.json"
    )
    with test_json_path.open("w", encoding="utf-8") as fp:
        json.dump(test_summary, fp, indent=2)
    logger.info("Saved hybrid test metrics to %s", test_json_path)

    return test_summary


def main() -> None:
    dataset_config = load_dataset_config()
    hybrid_config = load_hybrid_config()
    liar_cfg = dataset_config["liar"]
    processed_dir = Path(liar_cfg["processed_dir"])

    train_df = load_processed_split("train", processed_dir)
    valid_df = load_processed_split("valid", processed_dir)
    test_df = load_processed_split("test", processed_dir)

    runtime_cfg = hybrid_config["runtime"]
    paths_cfg = hybrid_config["paths"]
    training_cfg = hybrid_config["training"]
    model_cfg = hybrid_config["model"]

    device = resolve_device()
    metadata_spec = build_metadata_spec(hybrid_config)
    logger.info("Using device: %s", device)
    logger.info("Metadata layout: %s", describe_feature_layout(metadata_spec))
    logger.info("use_metadata=%s", model_cfg.get("use_metadata", True))

    max_train_samples = _resolve_sample_limit(runtime_cfg.get("max_train_samples"), "MAX_TRAIN_SAMPLES")
    max_valid_samples = _resolve_sample_limit(runtime_cfg.get("max_valid_samples"), "MAX_VALID_SAMPLES")
    if max_train_samples is not None:
        train_df = train_df.head(max_train_samples).copy()
    if max_valid_samples is not None:
        valid_df = valid_df.head(max_valid_samples).copy()
    logger.info(
        "Hybrid data sizes — train: %d, valid: %d, test: %d",
        len(train_df),
        len(valid_df),
        len(test_df),
    )

    output_dir = Path(paths_cfg["output_dir"])
    logs_dir = Path(paths_cfg["logs_dir"])
    figures_dir = Path("reports/figures")
    predictions_dir = Path("reports/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    if ENABLE_FIGURES:
        figures_dir.mkdir(parents=True, exist_ok=True)

    seeds = _resolve_seeds(training_cfg)
    is_multi_seed = len(seeds) > 1
    logger.info("Training over %d seed(s): %s", len(seeds), seeds)

    summaries: list[dict[str, Any]] = []
    for seed in seeds:
        summary = train_one_seed(
            seed=seed,
            hybrid_config=hybrid_config,
            dataset_config=dataset_config,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            metadata_spec=metadata_spec,
            device=device,
            logs_dir=logs_dir,
            figures_dir=figures_dir,
            predictions_dir=predictions_dir,
            output_dir=output_dir,
            is_multi_seed=is_multi_seed,
        )
        summaries.append(summary)

    if is_multi_seed:
        aggregate = aggregate_seed_summaries(summaries)
        aggregate_path = logs_dir / "hybrid_test_metrics_multiseed.json"
        with aggregate_path.open("w", encoding="utf-8") as fp:
            json.dump(
                {
                    "aggregate": aggregate,
                    "per_seed": summaries,
                    "leakage_corrected": metadata_spec.leakage_corrected,
                    "loss_config": training_cfg.get("loss"),
                },
                fp,
                indent=2,
            )
        logger.info(
            "Aggregated %d seeds — TEST macro-F1 %.4f ± %.4f, accuracy %.4f ± %.4f",
            aggregate["num_seeds"],
            aggregate["test_macro_f1"]["mean"],
            aggregate["test_macro_f1"]["std"],
            aggregate["test_accuracy"]["mean"],
            aggregate["test_accuracy"]["std"],
        )
        logger.info("Saved hybrid multi-seed metrics to %s", aggregate_path)


if __name__ == "__main__":
    main()
