"""Tests for transformer dataset and wrapper behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from src.disinfo_detection.models_transformers import LIARDataset, RoBERTaClassifier


class DummyTokenizer:
    """Simple tokenizer for deterministic transformer tests.

    Now that `LIARDataset` pre-tokenizes the full batch at construction time
    (rather than per-item), this dummy needs to honor the batch dimension too.
    """

    def __call__(self, text, padding: str, truncation: bool, max_length: int, return_tensors: str):
        del padding, truncation, return_tensors
        batch_size = len(text) if isinstance(text, list) else 1
        input_ids = torch.arange(max_length, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
        attention_mask = torch.ones((batch_size, max_length), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class DummySequenceClassifier(torch.nn.Module):
    """Small classifier that mimics Hugging Face output shape."""

    def __init__(self, vocab_size: int = 256, hidden_size: int = 8, num_labels: int = 6) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        del attention_mask
        pooled = self.embedding(input_ids).mean(dim=1)
        logits = self.linear(pooled)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return SimpleNamespace(logits=logits, loss=loss)


def test_liar_dataset_returns_expected_tensor_fields() -> None:
    """Dataset items should expose token ids, attention mask, and label."""

    dataset = LIARDataset(
        texts=["test statement"],
        labels=[0],
        tokenizer=DummyTokenizer(),
        max_length=8,
    )
    batch = dataset[0]
    assert batch["input_ids"].shape[0] == 8
    assert batch["attention_mask"].shape[0] == 8
    assert int(batch["label"].item()) == 0


def test_roberta_classifier_train_evaluate_and_save(tmp_path: Path) -> None:
    """Transformer wrapper should train, evaluate, and persist with a dummy model."""

    tokenizer = DummyTokenizer()
    model = DummySequenceClassifier()
    classifier = RoBERTaClassifier(
        model_name="dummy-model",
        num_labels=6,
        model=model,
        tokenizer=tokenizer,
    )
    dataset = LIARDataset(
        texts=["a", "b", "c", "d"],
        labels=[0, 1, 0, 1],
        tokenizer=tokenizer,
        max_length=8,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    optimizer = torch.optim.AdamW(classifier.model.parameters(), lr=1e-3)

    train_loss = classifier.train_epoch(dataloader, optimizer, scheduler=None, device=torch.device("cpu"))
    metrics = classifier.evaluate(dataloader, torch.device("cpu"))
    checkpoint_path = tmp_path / "dummy_transformer.pt"
    classifier.save(str(checkpoint_path))

    reloaded = RoBERTaClassifier(
        model_name="dummy-model",
        num_labels=6,
        model=DummySequenceClassifier(),
        tokenizer=tokenizer,
    )
    reloaded.load(str(checkpoint_path))

    assert train_loss >= 0.0
    assert {"accuracy", "macro_f1", "val_loss"} <= set(metrics)
    assert checkpoint_path.exists()
