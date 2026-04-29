"""Shared training loop for the LIAR classifiers.

`RoBERTaClassifier` and `HybridTrainer` both follow the same per-epoch /
evaluation / checkpoint pattern. The only differences are how a batch is
split into model kwargs vs. labels and how the logits come out of the
underlying module (HF returns `outputs.logits`; the hybrid model returns
the logits tensor directly). `BaseTrainer` captures the common shape and
exposes `_split_batch` / `_compute_logits` as the per-subclass hooks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

from src.disinfo_detection.evaluation import compute_metrics


class BaseTrainer:
    """Common train / evaluate / save / load implementation.

    Subclasses must:
    - assign `self.model` (a `nn.Module`) before invoking any method,
    - implement `_split_batch(batch, device)` returning `(model_kwargs, labels)`,
    - implement `_compute_logits(model_kwargs)` returning a logits tensor.
    """

    model: nn.Module
    label_names: list[str]
    loss_module: nn.Module

    def _loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.loss_module(logits, labels)

    def _split_batch(
        self, batch: dict[str, torch.Tensor], device: torch.device
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        raise NotImplementedError

    def _compute_logits(self, model_kwargs: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def train_epoch(
        self,
        dataloader: Iterable[dict[str, torch.Tensor]],
        optimizer,
        scheduler,
        device: torch.device,
        gradient_clip: float | None = None,
        gradient_accumulation_steps: int = 1,
        log_every_steps: int = 0,
        logger: logging.Logger | None = None,
    ) -> float:
        """Run one training epoch and return the mean loss."""

        self.model.train()
        total_loss = 0.0
        total_batches = 0
        accumulation = max(gradient_accumulation_steps, 1)
        optimizer.zero_grad()

        for step_index, batch in enumerate(dataloader, start=1):
            model_kwargs, labels = self._split_batch(batch, device)
            logits = self._compute_logits(model_kwargs)
            loss = self._loss_fn(logits, labels) / accumulation
            loss.backward()

            if step_index % accumulation == 0:
                if gradient_clip is not None and gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            total_loss += float(loss.item()) * accumulation
            total_batches += 1
            if logger is not None and log_every_steps > 0 and step_index % log_every_steps == 0:
                logger.info(
                    "Training step %d/%d — running mean loss %.4f",
                    step_index,
                    len(dataloader),
                    total_loss / total_batches,
                )

        return total_loss / max(total_batches, 1)

    def evaluate(
        self,
        dataloader: Iterable[dict[str, torch.Tensor]],
        device: torch.device,
        return_outputs: bool = False,
    ) -> dict[str, Any]:
        """Evaluate the model and return metrics (and optionally raw outputs)."""

        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        predictions: list[int] = []
        labels: list[int] = []
        logits_rows: list[list[float]] = []
        probability_rows: list[list[float]] = []

        with torch.no_grad():
            for batch in dataloader:
                model_kwargs, batch_labels = self._split_batch(batch, device)
                logits = self._compute_logits(model_kwargs)
                loss = self._loss_fn(logits, batch_labels)
                total_loss += float(loss.item())
                total_batches += 1
                batch_predictions = torch.argmax(logits, dim=1)
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


__all__ = ["BaseTrainer"]
