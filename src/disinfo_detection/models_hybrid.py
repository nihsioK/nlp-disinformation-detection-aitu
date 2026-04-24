"""Hybrid text + metadata classifier — the thesis' novel contribution.

Architecture (Section 4 of MSRW 2 / ИПМ):

                 ┌────────────────────────┐
    statement -->│ RoBERTa encoder        │──> pooled [CLS] (768)
                 └────────────────────────┘                    │
                                                               ▼
                                                       ┌───────────────┐
    credibility vector (5) + scalars (4) --┐           │               │
                                           ├─dense────>│               │
    hashed categorical fields (6 IDs) -----┘           │   fusion MLP  │──> 6 logits
                                                       │               │
                                                       └───────────────┘

Design notes:
- Metadata branch is **ablatable at construction time** (`use_metadata=False`)
  so the exact same module can be used as the text-only RoBERTa baseline for
  RQ2. This is important for an honest apples-to-apples comparison.
- Categorical fields share a single embedding table indexed into via offsets
  per field — that's the standard "shared hashing trick" layout and it keeps
  the parameter count tiny.
- Fusion MLP is intentionally small (one hidden layer, GELU, dropout). On
  LIAR the text encoder carries most of the capacity; a wide fusion head just
  overfits.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.disinfo_detection.evaluation import compute_metrics
from src.disinfo_detection.metadata_features import MetadataSpec
from src.disinfo_detection.models_transformers import load_dataset_config


class MetadataBranch(nn.Module):
    """Encodes dense credibility features + hashed categorical IDs.

    Output dimensionality: `output_dim` (default 64), ready to concatenate
    with the [CLS] embedding before the fusion MLP.
    """

    def __init__(
        self,
        spec: MetadataSpec,
        categorical_embedding_dim: int = 16,
        output_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.categorical_embedding_dim = categorical_embedding_dim
        self.output_dim = output_dim

        # One shared embedding table, with per-field offsets to avoid
        # cross-field bucket collisions (each field gets its own bucket range).
        self.num_total_buckets = spec.num_buckets * spec.num_categorical
        self.categorical_embedding = nn.Embedding(
            num_embeddings=self.num_total_buckets,
            embedding_dim=categorical_embedding_dim,
        )
        offsets = torch.arange(spec.num_categorical, dtype=torch.long) * spec.num_buckets
        self.register_buffer("field_offsets", offsets, persistent=False)

        categorical_flat_dim = categorical_embedding_dim * spec.num_categorical
        input_dim = spec.num_dense + categorical_flat_dim

        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        dense_features: torch.Tensor,
        categorical_ids: torch.Tensor,
    ) -> torch.Tensor:
        # `categorical_ids` has shape (batch, num_categorical) with values in
        # [0, num_buckets). We shift each column into its reserved range and
        # look up the shared embedding table.
        offset_ids = categorical_ids + self.field_offsets.unsqueeze(0)
        categorical_embedding = self.categorical_embedding(offset_ids)
        categorical_flat = categorical_embedding.flatten(start_dim=1)
        combined = torch.cat([dense_features, categorical_flat], dim=1)
        return self.projection(combined)


class HybridClassifier(nn.Module):
    """RoBERTa + metadata branch + fusion MLP → 6-class logits."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_labels: int = 6,
        hidden_dropout_prob: float = 0.2,
        attention_probs_dropout_prob: float = 0.2,
        use_metadata: bool = True,
        metadata_spec: MetadataSpec | None = None,
        metadata_output_dim: int = 64,
        metadata_embedding_dim: int = 16,
        fusion_hidden_dim: int = 128,
        fusion_dropout: float = 0.2,
        pretrained_encoder: nn.Module | None = None,
        tokenizer=None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.use_metadata = use_metadata
        self.metadata_spec = metadata_spec or MetadataSpec()

        if pretrained_encoder is None:
            config = AutoConfig.from_pretrained(
                model_name,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
            )
            self.encoder = AutoModel.from_pretrained(model_name, config=config)
        else:
            self.encoder = pretrained_encoder

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)

        encoder_hidden = self.encoder.config.hidden_size
        self.pooler_dropout = nn.Dropout(hidden_dropout_prob)

        if self.use_metadata:
            self.metadata_branch: nn.Module | None = MetadataBranch(
                spec=self.metadata_spec,
                categorical_embedding_dim=metadata_embedding_dim,
                output_dim=metadata_output_dim,
                dropout=fusion_dropout,
            )
            fusion_input_dim = encoder_hidden + metadata_output_dim
        else:
            self.metadata_branch = None
            fusion_input_dim = encoder_hidden

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, num_labels),
        )

    @staticmethod
    def _pool_cls(encoder_outputs, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return the first-token ([CLS]) representation from the encoder.

        RoBERTa's native `pooler_output` applies tanh on top of [CLS], which
        empirically under-performs the raw [CLS] state on small classification
        datasets. We therefore take `last_hidden_state[:, 0]` directly.
        """

        return encoder_outputs.last_hidden_state[:, 0, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        dense_features: torch.Tensor | None = None,
        categorical_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool_cls(encoder_outputs, attention_mask)
        pooled = self.pooler_dropout(pooled)

        if self.use_metadata:
            if dense_features is None or categorical_ids is None:
                raise ValueError(
                    "HybridClassifier constructed with use_metadata=True but received "
                    "no metadata tensors — did the DataLoader collate drop them?"
                )
            metadata = self.metadata_branch(dense_features, categorical_ids)
            fused = torch.cat([pooled, metadata], dim=1)
        else:
            fused = pooled

        return self.classifier(fused)


class HybridTrainer:
    """Training loop mirror of `RoBERTaClassifier` but aware of metadata tensors.

    Kept as a thin wrapper rather than inheriting from `RoBERTaClassifier`
    because the forward signature differs (extra metadata tensors).
    """

    def __init__(
        self,
        model: HybridClassifier,
        label_smoothing: float = 0.0,
        class_weights: torch.Tensor | None = None,
        dataset_config_path: str = "config/dataset.yaml",
    ) -> None:
        self.model = model
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
        dataset_config = load_dataset_config(dataset_config_path)
        self.label_names = dataset_config["liar"]["label_names"]

    def _loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weight, label_smoothing=self.label_smoothing)
        return loss_fct(logits, labels)

    def _move_batch(self, batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
        moved = {
            "input_ids": batch["input_ids"].to(device, non_blocking=True),
            "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
            "label": batch["label"].to(device, non_blocking=True),
        }
        if self.model.use_metadata:
            moved["dense_features"] = batch["dense_features"].to(device, non_blocking=True)
            moved["categorical_ids"] = batch["categorical_ids"].to(device, non_blocking=True)
        return moved

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
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        optimizer.zero_grad()

        for step_index, batch in enumerate(dataloader, start=1):
            tensors = self._move_batch(batch, device)
            logits = self.model(
                input_ids=tensors["input_ids"],
                attention_mask=tensors["attention_mask"],
                dense_features=tensors.get("dense_features"),
                categorical_ids=tensors.get("categorical_ids"),
            )
            loss = self._loss_fn(logits, tensors["label"]) / max(gradient_accumulation_steps, 1)
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
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        predictions: list[int] = []
        labels: list[int] = []
        logits_rows: list[list[float]] = []
        probability_rows: list[list[float]] = []

        with torch.no_grad():
            for batch in dataloader:
                tensors = self._move_batch(batch, device)
                logits = self.model(
                    input_ids=tensors["input_ids"],
                    attention_mask=tensors["attention_mask"],
                    dense_features=tensors.get("dense_features"),
                    categorical_ids=tensors.get("categorical_ids"),
                )
                loss = self._loss_fn(logits, tensors["label"])
                total_loss += float(loss.item())
                total_batches += 1
                batch_predictions = torch.argmax(logits, dim=1)
                predictions.extend(batch_predictions.cpu().tolist())
                labels.extend(tensors["label"].cpu().tolist())
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


__all__ = ["HybridClassifier", "HybridTrainer", "MetadataBranch"]
