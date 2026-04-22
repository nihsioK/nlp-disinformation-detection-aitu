"""Transformer dataset and model wrappers for LIAR classification."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from src.disinfo_detection.evaluation import compute_metrics


def load_transformer_config(config_path: str = "config/transformer.yaml") -> dict:
    """Load transformer training configuration.

    Args:
        config_path: Path to the transformer YAML config.

    Returns:
        Parsed transformer configuration dictionary.
    """

    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_dataset_config(config_path: str = "config/dataset.yaml") -> dict:
    """Load dataset configuration for label names.

    Args:
        config_path: Path to the dataset YAML config.

    Returns:
        Parsed dataset configuration dictionary.
    """

    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


class LIARDataset(Dataset):
    """PyTorch dataset wrapping tokenized LIAR statements.

    Args:
        texts: Input statement strings.
        labels: Integer label ids.
        tokenizer: Hugging Face tokenizer or compatible callable.
        max_length: Maximum token length.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_length: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""

        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one tokenized sample.

        Args:
            index: Sample index.

        Returns:
            Dictionary with `input_ids`, `attention_mask`, and `label`.
        """

        encoded = self.tokenizer(
            self.texts[index],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
        }


class RoBERTaClassifier:
    """Fine-tunable RoBERTa-based classifier for six-class LIAR prediction."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_labels: int = 6,
        hidden_dropout_prob: float = 0.1,
        dataset_config_path: str = "config/dataset.yaml",
        model=None,
        tokenizer=None,
    ) -> None:
        self.model_name = model_name
        self.num_labels = num_labels
        self.hidden_dropout_prob = hidden_dropout_prob
        dataset_config = load_dataset_config(dataset_config_path)
        self.label_names = dataset_config["liar"]["label_names"]
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)

        if model is None:
            config = AutoConfig.from_pretrained(
                model_name,
                num_labels=num_labels,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=hidden_dropout_prob,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
        else:
            self.model = model

    def train_epoch(
        self,
        dataloader,
        optimizer,
        scheduler,
        device,
        gradient_clip: float | None = None,
        log_every_steps: int = 0,
        logger=None,
    ) -> float:
        """Run one training epoch and return average loss.

        Args:
            dataloader: Training dataloader.
            optimizer: Torch optimizer.
            scheduler: Learning-rate scheduler or `None`.
            device: Torch device.
            gradient_clip: Optional max gradient norm.
            log_every_steps: Step interval for progress logging.
            logger: Optional logger for progress updates.

        Returns:
            Mean training loss for the epoch.
        """

        self.model.train()
        total_loss = 0.0
        total_batches = 0
        for step_index, batch in enumerate(dataloader, start=1):
            optimizer.zero_grad()
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["label"].to(device),
            }
            outputs = self.model(**inputs)
            loss = outputs.loss
            loss.backward()
            if gradient_clip is not None and gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += float(loss.item())
            total_batches += 1
            if logger is not None and log_every_steps > 0 and step_index % log_every_steps == 0:
                logger.info(
                    "Training step %d/%d — running mean loss %.4f",
                    step_index,
                    len(dataloader),
                    total_loss / total_batches,
                )

        return total_loss / max(total_batches, 1)

    def evaluate(self, dataloader, device) -> dict:
        """Evaluate the model on a dataloader.

        Args:
            dataloader: Evaluation dataloader.
            device: Torch device.

        Returns:
            Metric dictionary including validation loss.
        """

        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        predictions: list[int] = []
        labels: list[int] = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                    "labels": batch["label"].to(device),
                }
                outputs = self.model(**inputs)
                logits = outputs.logits
                batch_predictions = torch.argmax(logits, dim=1)

                total_loss += float(outputs.loss.item())
                total_batches += 1
                predictions.extend(batch_predictions.cpu().tolist())
                labels.extend(batch["label"].cpu().tolist())

        metrics = compute_metrics(labels, predictions, self.label_names)
        metrics["val_loss"] = total_loss / max(total_batches, 1)
        return metrics

    def save(self, path: str) -> None:
        """Save the underlying model state.

        Args:
            path: Destination path.
        """

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), output_path)

    def load(self, path: str) -> None:
        """Load model state from disk.

        Args:
            path: State dict path.
        """

        state_dict = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state_dict)
