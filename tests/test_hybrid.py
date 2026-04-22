"""Unit tests for the hybrid model pipeline.

These tests avoid downloading real pretrained weights — we patch in a tiny
random-init RoBERTa config so the full forward/backward pass runs in <2s.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import AutoTokenizer, RobertaConfig, RobertaModel

from src.disinfo_detection.datasets_hybrid import HybridLIARDataset
from src.disinfo_detection.metadata_features import (
    CREDIBILITY_SCALAR_COLS,
    CREDIBILITY_VECTOR_COLS,
    MetadataSpec,
    build_categorical_matrix,
    build_dense_matrix,
    hash_categorical_field,
    tensors_from_dataframe,
)
from src.disinfo_detection.models_hybrid import HybridClassifier, HybridTrainer


def _toy_dataframe(rows: int = 4) -> pd.DataFrame:
    speakers_pool = ["alice", "bob", "carol", "dave", "eve", "frank", "grace"]
    parties_pool = ["republican", "democrat", "independent"]
    jobs_pool = ["senator", "governor", "analyst", "journalist"]
    states_pool = ["TX", "CA", "NY", "FL", "WA"]
    subjects_pool = ["economy", "health", "immigration", "education"]
    contexts_pool = ["debate", "tweet", "speech", "interview"]

    data = {
        "statement_transformer": [f"Example statement number {index}." for index in range(rows)],
        "label_id": [index % 6 for index in range(rows)],
        "speaker": [speakers_pool[index % len(speakers_pool)] for index in range(rows)],
        "party": [parties_pool[index % len(parties_pool)] for index in range(rows)],
        "job": [jobs_pool[index % len(jobs_pool)] for index in range(rows)],
        "state": [states_pool[index % len(states_pool)] for index in range(rows)],
        "subject": [subjects_pool[index % len(subjects_pool)] for index in range(rows)],
        "context": [contexts_pool[index % len(contexts_pool)] for index in range(rows)],
    }
    sampled = np.random.dirichlet(np.ones(5), size=rows)
    for index, column in enumerate(CREDIBILITY_VECTOR_COLS):
        data[column] = sampled[:, index].tolist()
    totals = [float(10 + 5 * index) for index in range(rows)]
    data["cred_total"] = totals
    data["cred_log_total"] = [math.log1p(value) for value in totals]
    data["cred_pants_share"] = [0.1 * (index % 3) for index in range(rows)]
    data["cred_false_share"] = [0.2 + 0.1 * (index % 4) for index in range(rows)]
    return pd.DataFrame(data)


def test_hash_is_deterministic_and_bounded():
    series = pd.Series(["alice", "bob", "", None, "alice"])
    ids = hash_categorical_field(series, num_buckets=16)
    assert ids.shape == (5,)
    assert ids.min() >= 0 and ids.max() < 16
    # Same value → same bucket.
    assert ids[0] == ids[4]


def test_build_dense_and_categorical_matrices():
    df = _toy_dataframe(rows=4)
    dense = build_dense_matrix(df)
    categorical = build_categorical_matrix(df)
    assert dense.shape == (4, len(CREDIBILITY_VECTOR_COLS) + len(CREDIBILITY_SCALAR_COLS))
    assert dense.dtype == np.float32
    assert categorical.shape == (4, len(MetadataSpec().categorical_fields))
    assert categorical.dtype == np.int64


def test_build_dense_matrix_raises_when_scalars_missing():
    df = _toy_dataframe(rows=2).drop(columns=["cred_total"])
    with pytest.raises(KeyError):
        build_dense_matrix(df)


def _build_tiny_hybrid(use_metadata: bool = True) -> HybridClassifier:
    tiny_config = RobertaConfig(
        vocab_size=50265,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=130,
        type_vocab_size=1,
        pad_token_id=1,
    )
    encoder = RobertaModel(tiny_config)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    return HybridClassifier(
        model_name="roberta-base",
        pretrained_encoder=encoder,
        tokenizer=tokenizer,
        use_metadata=use_metadata,
        metadata_output_dim=8,
        metadata_embedding_dim=4,
        fusion_hidden_dim=16,
    )


def test_hybrid_forward_and_backward_runs():
    torch.manual_seed(0)
    df = _toy_dataframe(rows=4)
    model = _build_tiny_hybrid(use_metadata=True)
    dataset = HybridLIARDataset(df=df, tokenizer=model.tokenizer, max_length=16)
    batch = {key: torch.stack([dataset[i][key] for i in range(len(dataset))]) for key in dataset[0]}

    logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        dense_features=batch["dense_features"],
        categorical_ids=batch["categorical_ids"],
    )
    assert logits.shape == (4, 6)

    trainer = HybridTrainer(model=model)
    loss = trainer._loss_fn(logits, batch["label"])
    loss.backward()
    # At least the fusion head must have a gradient.
    assert model.classifier[0].weight.grad is not None


def test_hybrid_ablation_matches_text_only_shapes():
    torch.manual_seed(0)
    df = _toy_dataframe(rows=3)
    model = _build_tiny_hybrid(use_metadata=False)
    dataset = HybridLIARDataset(
        df=df,
        tokenizer=model.tokenizer,
        max_length=16,
        include_metadata=False,
    )
    batch = {key: torch.stack([dataset[i][key] for i in range(len(dataset))]) for key in dataset[0]}
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    assert logits.shape == (3, 6)
    assert model.metadata_branch is None


def test_tensors_from_dataframe_aligned_with_rows():
    df = _toy_dataframe(rows=5)
    dense, categorical = tensors_from_dataframe(df)
    assert dense.shape[0] == 5
    assert categorical.shape[0] == 5
    assert dense.dtype == torch.float32
    assert categorical.dtype == torch.int64
