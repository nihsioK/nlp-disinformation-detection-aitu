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

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.disinfo_detection.losses import OrdinalAwareLoss
from src.disinfo_detection.metadata_features import MetadataSpec
from src.disinfo_detection.models_transformers import load_dataset_config
from src.disinfo_detection.trainer import BaseTrainer


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

        self.categorical_embedding = nn.Embedding(
            num_embeddings=spec.total_buckets,
            embedding_dim=categorical_embedding_dim,
        )
        offsets = torch.tensor(spec.field_offsets, dtype=torch.long)
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
    def _pool_cls(encoder_outputs) -> torch.Tensor:
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
        pooled = self._pool_cls(encoder_outputs)
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


class HybridTrainer(BaseTrainer):
    """Training loop wrapper for `HybridClassifier`."""

    def __init__(
        self,
        model: HybridClassifier,
        dataset_config_path: str = "config/dataset.yaml",
        loss_module: OrdinalAwareLoss | None = None,
    ) -> None:
        self.model = model
        dataset_config = load_dataset_config(dataset_config_path)
        self.label_names = dataset_config["liar"]["label_names"]
        self.loss_module = loss_module or OrdinalAwareLoss(
            num_classes=model.num_labels,
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
        if self.model.use_metadata:
            model_kwargs["dense_features"] = batch["dense_features"].to(device, non_blocking=True)
            model_kwargs["categorical_ids"] = batch["categorical_ids"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        return model_kwargs, labels

    def _compute_logits(self, model_kwargs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(**model_kwargs)


__all__ = ["HybridClassifier", "HybridTrainer", "MetadataBranch"]
