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


def test_hybrid_trainer_collects_prediction_outputs_for_metadata_and_ablation():
    """Hybrid evaluation should expose logits/probabilities for downstream analysis."""

    torch.manual_seed(0)
    df = _toy_dataframe(rows=4)

    metadata_model = _build_tiny_hybrid(use_metadata=True)
    metadata_dataset = HybridLIARDataset(df=df, tokenizer=metadata_model.tokenizer, max_length=16)
    metadata_loader = torch.utils.data.DataLoader(metadata_dataset, batch_size=2, shuffle=False)
    metadata_metrics = HybridTrainer(metadata_model).evaluate(
        metadata_loader,
        torch.device("cpu"),
        return_outputs=True,
    )
    assert metadata_metrics["labels"] == [0, 1, 2, 3]
    assert len(metadata_metrics["predictions"]) == 4
    assert len(metadata_metrics["probabilities"]) == 4
    assert len(metadata_metrics["probabilities"][0]) == 6
    assert len(metadata_metrics["logits"]) == 4
    assert len(metadata_metrics["logits"][0]) == 6

    textonly_model = _build_tiny_hybrid(use_metadata=False)
    textonly_dataset = HybridLIARDataset(
        df=df,
        tokenizer=textonly_model.tokenizer,
        max_length=16,
        include_metadata=False,
    )
    textonly_loader = torch.utils.data.DataLoader(textonly_dataset, batch_size=2, shuffle=False)
    textonly_metrics = HybridTrainer(textonly_model).evaluate(
        textonly_loader,
        torch.device("cpu"),
        return_outputs=True,
    )
    assert textonly_metrics["labels"] == [0, 1, 2, 3]
    assert len(textonly_metrics["predictions"]) == 4
    assert len(textonly_metrics["probabilities"]) == 4
    assert len(textonly_metrics["logits"]) == 4


def test_tensors_from_dataframe_aligned_with_rows():
    df = _toy_dataframe(rows=5)
    dense, categorical = tensors_from_dataframe(df)
    assert dense.shape[0] == 5
    assert categorical.shape[0] == 5
    assert dense.dtype == torch.float32
    assert categorical.dtype == torch.int64


def test_metadata_spec_supports_per_field_bucket_sizes():
    spec = MetadataSpec(
        categorical_fields=("speaker", "party"),
        num_buckets={"speaker": 1024, "party": 32},
    )
    assert spec.field_bucket_sizes == (1024, 32)
    assert spec.field_offsets == (0, 1024)
    assert spec.total_buckets == 1024 + 32


def test_categorical_matrix_respects_per_field_buckets():
    df = _toy_dataframe(rows=8)
    spec = MetadataSpec(
        categorical_fields=("speaker", "party"),
        num_buckets={"speaker": 1024, "party": 4},
    )
    matrix = build_categorical_matrix(df, spec=spec)
    assert matrix.shape == (8, 2)
    assert matrix[:, 0].max() < 1024
    assert matrix[:, 1].max() < 4


def test_metadata_branch_uses_total_buckets_for_embedding_table():
    from src.disinfo_detection.models_hybrid import MetadataBranch

    spec = MetadataSpec(
        categorical_fields=("speaker", "party"),
        num_buckets={"speaker": 1024, "party": 32},
    )
    branch = MetadataBranch(spec=spec, categorical_embedding_dim=4, output_dim=8)
    assert branch.categorical_embedding.num_embeddings == 1024 + 32
    assert tuple(branch.field_offsets.tolist()) == (0, 1024)


def test_dense_matrix_uses_corrected_columns_when_flag_set():
    df = _toy_dataframe(rows=3)
    df["credibility_corrected_0"] = [0.4, 0.4, 0.4]
    df["credibility_corrected_1"] = [0.2, 0.2, 0.2]
    df["credibility_corrected_2"] = [0.2, 0.2, 0.2]
    df["credibility_corrected_3"] = [0.1, 0.1, 0.1]
    df["credibility_corrected_4"] = [0.1, 0.1, 0.1]
    df["cred_total_corrected"] = [10.0, 12.0, 14.0]
    df["cred_log_total_corrected"] = [2.4, 2.6, 2.7]
    df["cred_pants_share_corrected"] = [0.1, 0.05, 0.0]
    df["cred_false_share_corrected"] = [0.3, 0.25, 0.2]
    spec = MetadataSpec(leakage_corrected=True)
    dense = build_dense_matrix(df, spec=spec)
    # First five columns should reflect the corrected credibility vector.
    assert abs(dense[0, 0] - 0.4) < 1e-6
    assert abs(dense[0, 1] - 0.2) < 1e-6


def test_aggregate_seed_summaries_returns_mean_and_std():
    from src.disinfo_detection.evaluation import aggregate_seed_summaries

    summaries = [
        {
            "seed": 1,
            "test_macro_f1": 0.4,
            "test_accuracy": 0.5,
            "test_per_class_f1": {"a": 0.2, "b": 0.6},
            "test_per_class_precision": {"a": 0.3, "b": 0.7},
            "test_per_class_recall": {"a": 0.1, "b": 0.5},
            "test_confusion_matrix_labels": ["a", "b"],
        },
        {
            "seed": 2,
            "test_macro_f1": 0.6,
            "test_accuracy": 0.5,
            "test_per_class_f1": {"a": 0.4, "b": 0.8},
            "test_per_class_precision": {"a": 0.5, "b": 0.9},
            "test_per_class_recall": {"a": 0.3, "b": 0.7},
            "test_confusion_matrix_labels": ["a", "b"],
        },
    ]
    aggregate = aggregate_seed_summaries(summaries)
    assert aggregate["num_seeds"] == 2
    assert aggregate["seeds"] == [1, 2]
    assert abs(aggregate["test_macro_f1"]["mean"] - 0.5) < 1e-9
    assert abs(aggregate["test_macro_f1"]["std"] - 0.1) < 1e-9
    assert abs(aggregate["test_per_class_f1"]["mean"]["a"] - 0.3) < 1e-9
    assert abs(aggregate["test_per_class_f1"]["std"]["b"] - 0.1) < 1e-9
