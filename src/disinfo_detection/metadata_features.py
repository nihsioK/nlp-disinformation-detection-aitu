"""Metadata feature extraction for the hybrid model.

The hybrid architecture fuses the RoBERTa `[CLS]` embedding with a metadata
branch. That branch has two groups of features:

1. Dense numeric features already computed by `preprocessing.py`:
   - 5-dim normalized credibility vector (`credibility_0..4`)
   - 4 scalar credibility summaries (`cred_total`, `cred_log_total`,
     `cred_pants_share`, `cred_false_share`)
   When `MetadataSpec.leakage_corrected` is `True`, the matching
   `credibility_corrected_*` and `cred_*_corrected` columns are used instead.

2. Categorical metadata fields encoded via **feature hashing** so we avoid
   maintaining a vocabulary for speakers/subjects (LIAR has thousands of rare
   speakers). Bucket sizes can be set per field via a dict so high-signal
   fields like `speaker` get more capacity than low-cardinality fields like
   `party`.

Hashing is deterministic (hashlib.blake2b with a fixed salt) so that training
and evaluation produce identical feature matrices without persisting any
vocabulary.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping

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
CREDIBILITY_VECTOR_COLS_CORRECTED = [f"credibility_corrected_{index}" for index in range(5)]
CREDIBILITY_SCALAR_COLS_CORRECTED = [
    "cred_total_corrected",
    "cred_log_total_corrected",
    "cred_pants_share_corrected",
    "cred_false_share_corrected",
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

_BASE_SCALAR_NAMES = (
    "cred_total",
    "cred_log_total",
    "cred_pants_share",
    "cred_false_share",
)


@dataclass(frozen=True)
class MetadataSpec:
    """Declarative description of the metadata branch input shape.

    `num_buckets` accepts either an integer (applied to every field) or a
    mapping `{field_name: bucket_size}` so that fields with many unique values
    (e.g., `speaker`) can be assigned more capacity than near-binary fields
    (e.g., `party`).

    `leakage_corrected` toggles whether the dense feature matrix is read from
    the `credibility_corrected_*` columns (with the row's own verdict removed
    from the speaker's count history) or from the raw LIAR columns.
    """

    categorical_fields: tuple[str, ...] = DEFAULT_CATEGORICAL_FIELDS
    num_buckets: int | Mapping[str, int] = DEFAULT_NUM_BUCKETS
    hash_salt: bytes = DEFAULT_HASH_SALT
    leakage_corrected: bool = False

    def buckets_for(self, field_name: str) -> int:
        """Return the hash-bucket count assigned to `field_name`."""

        if isinstance(self.num_buckets, Mapping):
            return int(self.num_buckets.get(field_name, DEFAULT_NUM_BUCKETS))
        return int(self.num_buckets)

    @property
    def num_categorical(self) -> int:
        return len(self.categorical_fields)

    @property
    def num_dense(self) -> int:
        return len(CREDIBILITY_VECTOR_COLS) + len(CREDIBILITY_SCALAR_COLS)

    @property
    def field_bucket_sizes(self) -> tuple[int, ...]:
        return tuple(self.buckets_for(name) for name in self.categorical_fields)

    @property
    def field_offsets(self) -> tuple[int, ...]:
        offsets: list[int] = []
        running = 0
        for size in self.field_bucket_sizes:
            offsets.append(running)
            running += size
        return tuple(offsets)

    @property
    def total_buckets(self) -> int:
        return sum(self.field_bucket_sizes)

    @property
    def vector_columns(self) -> list[str]:
        return CREDIBILITY_VECTOR_COLS_CORRECTED if self.leakage_corrected else CREDIBILITY_VECTOR_COLS

    @property
    def scalar_columns(self) -> list[str]:
        return CREDIBILITY_SCALAR_COLS_CORRECTED if self.leakage_corrected else CREDIBILITY_SCALAR_COLS


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
    """Return an `(n_rows, num_categorical)` int64 matrix of hashed IDs.

    Each column is hashed against the per-field bucket size declared by
    `spec`, so values across columns can live in different ranges. The model
    re-aligns them via `MetadataBranch.field_offsets` before embedding lookup.
    """

    columns: list[np.ndarray] = []
    for field_name in spec.categorical_fields:
        bucket_size = spec.buckets_for(field_name)
        if field_name not in df.columns:
            # Field missing entirely — hash the empty string so the model sees
            # a consistent "unknown" bucket across the whole split.
            columns.append(
                np.full(len(df), _hash_token("", bucket_size, spec.hash_salt), dtype=np.int64)
            )
            continue
        columns.append(
            hash_categorical_field(df[field_name], num_buckets=bucket_size, hash_salt=spec.hash_salt)
        )
    return np.stack(columns, axis=1)


def build_dense_matrix(
    df: pd.DataFrame,
    spec: MetadataSpec = MetadataSpec(),
) -> np.ndarray:
    """Return an `(n_rows, num_dense)` float32 matrix of credibility features.

    `spec.leakage_corrected` selects between the raw and leakage-corrected
    column families. Scalar columns are normalized through `SCALAR_NORMALIZERS`
    so the metadata branch sees comparable magnitudes regardless of the
    variant used.
    """

    vector_columns = spec.vector_columns
    scalar_columns = spec.scalar_columns

    missing_vector = [column for column in vector_columns if column not in df.columns]
    missing_scalars = [column for column in scalar_columns if column not in df.columns]
    if missing_vector or missing_scalars:
        raise KeyError(
            "Processed DataFrame is missing required credibility columns: "
            f"{missing_vector + missing_scalars}. Re-run scripts/preprocess.py."
        )

    vector_part = df[vector_columns].to_numpy(dtype=np.float32, copy=True)
    scalar_part = df[scalar_columns].to_numpy(dtype=np.float32, copy=True)

    # Same normalizers regardless of variant — the corrected scalars have the
    # same dimensional shape as the raw ones, just with the row's own
    # contribution removed from the speaker's count history.
    for column_index, base_name in enumerate(_BASE_SCALAR_NAMES):
        divisor = SCALAR_NORMALIZERS.get(base_name, 1.0) or 1.0
        scalar_part[:, column_index] = scalar_part[:, column_index] / divisor

    return np.concatenate([vector_part, scalar_part], axis=1)


def tensors_from_dataframe(
    df: pd.DataFrame,
    spec: MetadataSpec = MetadataSpec(),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a processed DataFrame to (dense_features, categorical_ids) tensors."""

    dense = torch.from_numpy(build_dense_matrix(df, spec=spec)).float()
    categorical = torch.from_numpy(build_categorical_matrix(df, spec=spec)).long()
    return dense, categorical


def describe_feature_layout(spec: MetadataSpec = MetadataSpec()) -> dict[str, object]:
    """Small helper used by tests and logs to sanity-check the layout."""

    return {
        "num_dense": spec.num_dense,
        "num_categorical_fields": spec.num_categorical,
        "field_bucket_sizes": dict(zip(spec.categorical_fields, spec.field_bucket_sizes)),
        "total_buckets": spec.total_buckets,
        "categorical_fields": list(spec.categorical_fields),
        "leakage_corrected": spec.leakage_corrected,
    }


__all__ = [
    "MetadataSpec",
    "CREDIBILITY_VECTOR_COLS",
    "CREDIBILITY_SCALAR_COLS",
    "CREDIBILITY_VECTOR_COLS_CORRECTED",
    "CREDIBILITY_SCALAR_COLS_CORRECTED",
    "DEFAULT_CATEGORICAL_FIELDS",
    "DEFAULT_NUM_BUCKETS",
    "build_dense_matrix",
    "build_categorical_matrix",
    "tensors_from_dataframe",
    "describe_feature_layout",
    "hash_categorical_field",
]
