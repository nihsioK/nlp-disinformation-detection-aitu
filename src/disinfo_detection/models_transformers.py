"""Transformer dataset and model wrappers for LIAR classification.

`LIARDataset` pre-tokenizes the entire split at construction time, which is
~5× faster than re-tokenizing per `__getitem__` on Apple Silicon (the
tokenizer is CPU-bound). `RoBERTaClassifier` is a thin wrapper that owns
the HF model and tokenizer and delegates training to `BaseTrainer`.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from src.disinfo_detection.losses import OrdinalAwareLoss
from src.disinfo_detection.trainer import BaseTrainer


def load_transformer_config(config_path: str = "config/transformer.yaml") -> dict:
    """Load transformer training configuration."""

    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_dataset_config(config_path: str = "config/dataset.yaml") -> dict:
    """Load dataset configuration for label names."""

    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


class LIARDataset(Dataset):
    """PyTorch dataset wrapping pre-tokenized LIAR statements."""

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


class RoBERTaClassifier(BaseTrainer):
    """Fine-tunable RoBERTa-based classifier for six-class LIAR prediction."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_labels: int = 6,
        hidden_dropout_prob: float = 0.2,
        attention_probs_dropout_prob: float = 0.2,
        dataset_config_path: str = "config/dataset.yaml",
        model=None,
        tokenizer=None,
        loss_module: OrdinalAwareLoss | None = None,
    ) -> None:
        self.model_name = model_name
        self.num_labels = num_labels
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
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
        )

    def _split_batch(
        self, batch: dict[str, torch.Tensor], device: torch.device
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        model_kwargs = {
            "input_ids": batch["input_ids"].to(device, non_blocking=True),
            "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        }
        labels = batch["label"].to(device, non_blocking=True)
        return model_kwargs, labels

    def _compute_logits(self, model_kwargs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(**model_kwargs).logits


__all__ = ["LIARDataset", "RoBERTaClassifier", "load_dataset_config", "load_transformer_config"]
