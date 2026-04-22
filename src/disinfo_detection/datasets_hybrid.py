"""PyTorch Dataset that emits text tokens + metadata tensors in one batch.

Keeps pre-tokenization (5x faster on MPS than re-tokenizing per step) and
adds aligned metadata tensors so the hybrid trainer can consume a single
batch dict.
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.disinfo_detection.metadata_features import MetadataSpec, tensors_from_dataframe


class HybridLIARDataset(Dataset):
    """Pre-tokenized LIAR dataset with aligned metadata tensors."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int,
        text_column: str = "statement_transformer",
        label_column: str = "label_id",
        metadata_spec: MetadataSpec | None = None,
        include_metadata: bool = True,
    ) -> None:
        spec = metadata_spec or MetadataSpec()
        self.include_metadata = include_metadata

        encodings = tokenizer(
            df[text_column].astype(str).tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = torch.tensor(df[label_column].tolist(), dtype=torch.long)

        if include_metadata:
            dense, categorical = tensors_from_dataframe(df, spec=spec)
            self.dense_features = dense
            self.categorical_ids = categorical
        else:
            self.dense_features = None
            self.categorical_ids = None

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "label": self.labels[index],
        }
        if self.include_metadata:
            item["dense_features"] = self.dense_features[index]
            item["categorical_ids"] = self.categorical_ids[index]
        return item


__all__ = ["HybridLIARDataset"]
