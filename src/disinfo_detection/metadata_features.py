"""Metadata feature extraction for the hybrid model.

The hybrid architecture fuses the RoBERTa `[CLS]` embedding with a metadata
branch. That branch has two groups of features:

1. Dense numeric features already computed by `preprocessing.py`:
   - 5-dim normalized credibility vector (`credibility_0..4`)
   - 4 scalar credibility summaries (`cred_total`, `cred_log_total`,
     `cred_pants_share`, `cred_false_share`)

2. Categorical metadata fields encoded via **feature hashing** so we avoid
   maintaining a vocabulary for speakers/subjects (LIAR has thousands of rare
   speakers; a hashing trick with a moderate bucket size is the standard,
   out-of-core-friendly approach used in the original LIAR paper and most
   follow-ups).

Hashing is deterministic (hashlib.blake2b with a fixed salt) so that training
and evaluation produce identical feature matrices without persisting any
vocabulary.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

CREDIBILITY_VECTOR_COLS = [f"credibility_{index}" for index in range(5)]
CREDIBILITY_SCALAR_COLS = [
    "cred_total",
    "cred_log_total",
    "cred_pants_share",
    "cred_false_share",
]
DEFAULT_CATEGORICAL_FIELDS = ("speaker", "party", "job", "state", "subject", "context")
DEFAULT_NUM_BUCKETS = 256
DEFAULT_HASH_SALT = b"disinfo-aitu-v1"

# The scalar `cred_total` and `cred_log_total` are much larger in magnitude
# than the probabilities/ratios. We standardize on sensible constants so that
# the metadata branch doesn't explode at init. These constants are derived
# from LIAR train statistics (mean ~30, max a few hundred) and are robust
# enough that we don't need to fit a scaler.
SCALAR_NORMALIZERS = {
    "cred_total": 50.0,
    "cred_log_total": 5.0,
    "cred_pants_share": 1.0,
    "cred_false_share": 1.0,
}


@dataclass(frozen=True)
class MetadataSpec:
    """Declarative description of the metadata branch input shape."""

    categorical_fields: tuple[str, ...] = DEFAULT_CATEGORICAL_FIELDS
    num_buckets: int = DEFAULT_NUM_BUCKETS
    hash_salt: bytes = DEFAULT_HASH_SALT

    @property
    def num_categorical(self) -> int:
        return len(self.categorical_fields)

    @property
    def num_dense(self) -> int:
        return len(CREDIBILITY_VECTOR_COLS) + len(CREDIBILITY_SCALAR_COLS)


def _hash_token(value: str, num_buckets: int, salt: bytes) -> int:
    """Return a stable non-negative integer hash in `[0, num_buckets)`.

    We use blake2b with a fixed salt instead of Python's built-in `hash`
    because `hash` is randomized between processes (PYTHONHASHSEED) and would
    produce different features across training and evaluation runs.
    """

    if value is None:
        value = ""
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8, key=salt[:16]).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % num_buckets


def hash_categorical_field(
    series: pd.Series,
    num_buckets: int = DEFAULT_NUM_BUCKETS,
    hash_salt: bytes = DEFAULT_HASH_SALT,
) -> np.ndarray:
    """Hash a pandas Series of strings to integer bucket IDs."""

    ids = np.empty(len(series), dtype=np.int64)
    for index, value in enumerate(series.fillna("").astype(str).tolist()):
        ids[index] = _hash_token(value, num_buckets=num_buckets, salt=hash_salt)
    return ids


def build_categorical_matrix(
    df: pd.DataFrame,
    spec: MetadataSpec = MetadataSpec(),
) -> np.ndarray:
    """Return an `(n_rows, num_categorical)` int64 matrix of hashed IDs."""

    columns = []
    for field in spec.categorical_fields:
        if field not in df.columns:
            # Field missing entirely — hash the empty string so the model sees
            # a consistent "unknown" bucket across the whole split.
            columns.append(np.full(len(df), _hash_token("", spec.num_buckets, spec.hash_salt), dtype=np.int64))
            continue
        columns.append(
            hash_categorical_field(df[field], num_buckets=spec.num_buckets, hash_salt=spec.hash_salt)
        )
    return np.stack(columns, axis=1)


def build_dense_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return an `(n_rows, num_dense)` float32 matrix of credibility features."""

    missing_vector = [column for column in CREDIBILITY_VECTOR_COLS if column not in df.columns]
    missing_scalars = [column for column in CREDIBILITY_SCALAR_COLS if column not in df.columns]
    if missing_vector or missing_scalars:
        raise KeyError(
            "Processed DataFrame is missing required credibility columns: "
            f"{missing_vector + missing_scalars}. Re-run scripts/preprocess.py."
        )

    vector_part = df[CREDIBILITY_VECTOR_COLS].to_numpy(dtype=np.float32, copy=True)
    scalar_part = df[CREDIBILITY_SCALAR_COLS].to_numpy(dtype=np.float32, copy=True)

    # Normalize scalar magnitudes so they live on roughly the same scale as
    # the probability-shaped credibility vector.
    for column_index, column_name in enumerate(CREDIBILITY_SCALAR_COLS):
        divisor = SCALAR_NORMALIZERS.get(column_name, 1.0) or 1.0
        scalar_part[:, column_index] = scalar_part[:, column_index] / divisor

    return np.concatenate([vector_part, scalar_part], axis=1)


def tensors_from_dataframe(
    df: pd.DataFrame,
    spec: MetadataSpec = MetadataSpec(),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a processed DataFrame to (dense_features, categorical_ids) tensors."""

    dense = torch.from_numpy(build_dense_matrix(df)).float()
    categorical = torch.from_numpy(build_categorical_matrix(df, spec=spec)).long()
    return dense, categorical


def describe_feature_layout(spec: MetadataSpec = MetadataSpec()) -> dict[str, int]:
    """Small helper used by tests and logs to sanity-check the layout."""

    return {
        "num_dense": spec.num_dense,
        "num_categorical_fields": spec.num_categorical,
        "num_hash_buckets": spec.num_buckets,
        "categorical_fields": list(spec.categorical_fields),
    }


__all__ = [
    "MetadataSpec",
    "CREDIBILITY_VECTOR_COLS",
    "CREDIBILITY_SCALAR_COLS",
    "DEFAULT_CATEGORICAL_FIELDS",
    "DEFAULT_NUM_BUCKETS",
    "build_dense_matrix",
    "build_categorical_matrix",
    "tensors_from_dataframe",
    "describe_feature_layout",
    "hash_categorical_field",
]
