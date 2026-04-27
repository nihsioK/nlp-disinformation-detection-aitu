"""Transformer dataset and model wrappers for LIAR classification.

Key improvements over the previous version:
- `LIARDataset` pre-tokenizes once at construction time, eliminating the
  per-step tokenizer call that dominated CPU time on Apple Silicon.
- `RoBERTaClassifier` now supports:
    * class-weighted cross entropy (optional, for imbalanced LIAR)
    * label smoothing (helps on ordinal labels like truthfulness scale)
    * gradient accumulation for effective larger batch on MPS
- `evaluate` returns per-class F1 and confusion matrix via the shared utility.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from src.disinfo_detection.evaluation import compute_metrics
from src.disinfo_detection.losses import OrdinalAwareLoss


def load_transformer_config(config_path: str = "config/transformer.yaml") -> dict:
    """Load transformer training configuration."""

    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_dataset_config(config_path: str = "config/dataset.yaml") -> dict:
    """Load dataset configuration for label names."""

    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


class LIARDataset(Dataset):
    """PyTorch dataset wrapping pre-tokenized LIAR statements.

    Pre-tokenization is ~5x faster than re-tokenizing every __getitem__,
    especially on MPS where the tokenizer is CPU-bound.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_length: int,
    ) -> None:
        encodings = tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "label": self.labels[index],
        }


class RoBERTaClassifier:
    """Fine-tunable RoBERTa-based classifier for six-class LIAR prediction."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_labels: int = 6,
        hidden_dropout_prob: float = 0.2,
        attention_probs_dropout_prob: float = 0.2,
        dataset_config_path: str = "config/dataset.yaml",
        label_smoothing: float = 0.0,
        class_weights: torch.Tensor | None = None,
        model=None,
        tokenizer=None,
        loss_module: OrdinalAwareLoss | None = None,
    ) -> None:
        self.model_name = model_name
        self.num_labels = num_labels
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
        dataset_config = load_dataset_config(dataset_config_path)
        self.label_names = dataset_config["liar"]["label_names"]
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)

        if model is None:
            config = AutoConfig.from_pretrained(
                model_name,
                num_labels=num_labels,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
        else:
            self.model = model

        self.loss_module = loss_module or OrdinalAwareLoss(
            num_classes=num_labels,
            ce_weight=1.0,
            emd_weight=0.0,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )

    def _loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Delegate to the configured loss module (CE by default, EMD-aware when set)."""

        return self.loss_module(logits, labels)

    def train_epoch(
        self,
        dataloader,
        optimizer,
        scheduler,
        device,
        gradient_clip: float | None = None,
        gradient_accumulation_steps: int = 1,
        log_every_steps: int = 0,
        logger=None,
    ) -> float:
        """Run one training epoch and return average loss."""

        self.model.train()
        total_loss = 0.0
        total_batches = 0
        optimizer.zero_grad()

        for step_index, batch in enumerate(dataloader, start=1):
            inputs = {
                "input_ids": batch["input_ids"].to(device, non_blocking=True),
                "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
            }
            labels = batch["label"].to(device, non_blocking=True)
            outputs = self.model(**inputs)
            loss = self._loss_fn(outputs.logits, labels) / max(gradient_accumulation_steps, 1)
            loss.backward()

            if step_index % max(gradient_accumulation_steps, 1) == 0:
                if gradient_clip is not None and gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            total_loss += float(loss.item()) * max(gradient_accumulation_steps, 1)
            total_batches += 1
            if logger is not None and log_every_steps > 0 and step_index % log_every_steps == 0:
                logger.info(
                    "Training step %d/%d — running mean loss %.4f",
                    step_index,
                    len(dataloader),
                    total_loss / total_batches,
                )

        return total_loss / max(total_batches, 1)

    def evaluate(self, dataloader, device, return_outputs: bool = False) -> dict:
        """Evaluate the model on a dataloader."""

        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        predictions: list[int] = []
        labels: list[int] = []
        logits_rows: list[list[float]] = []
        probability_rows: list[list[float]] = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = {
                    "input_ids": batch["input_ids"].to(device, non_blocking=True),
                    "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
                }
                batch_labels = batch["label"].to(device, non_blocking=True)
                outputs = self.model(**inputs)
                logits = outputs.logits
                loss = self._loss_fn(logits, batch_labels)
                batch_predictions = torch.argmax(logits, dim=1)

                total_loss += float(loss.item())
                total_batches += 1
                predictions.extend(batch_predictions.cpu().tolist())
                labels.extend(batch_labels.cpu().tolist())
                if return_outputs:
                    logits_rows.extend(logits.detach().cpu().tolist())
                    probability_rows.extend(torch.softmax(logits, dim=1).detach().cpu().tolist())

        metrics = compute_metrics(labels, predictions, self.label_names)
        metrics["val_loss"] = total_loss / max(total_batches, 1)
        if return_outputs:
            metrics["labels"] = labels
            metrics["predictions"] = predictions
            metrics["probabilities"] = probability_rows
            metrics["logits"] = logits_rows
        return metrics

    def save(self, path: str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), output_path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state_dict)
