"""Tests for LIAR preprocessing utilities."""

from __future__ import annotations

import pandas as pd

from src.disinfo_detection.preprocessing import (
    build_credibility_vector,
    clean_text_for_tfidf,
    clean_text_for_transformer,
    preprocess_dataframe,
)


def test_clean_text_for_transformer_removes_urls_and_normalizes_whitespace() -> None:
    """Transformer cleaning should preserve case and punctuation, but strip URLs/HTML."""

    text = " Visit HTTPS://example.com   NOW for <b>Facts</b> "
    cleaned = clean_text_for_transformer(text)
    assert "http" not in cleaned
    # Casing and punctuation are preserved for the RoBERTa BPE tokenizer.
    assert cleaned == "Visit NOW for Facts"


def test_clean_text_for_tfidf_removes_urls_and_stopwords_without_stemming() -> None:
    """TF-IDF cleaning should lowercase, drop URLs and stopwords, keep surface forms."""

    text = "The cats are running to https://example.com faster than dogs."
    cleaned = clean_text_for_tfidf(text)
    tokens = cleaned.split()
    assert "http" not in cleaned
    assert "the" not in tokens
    # We deliberately do NOT stem, so surface forms survive.
    assert "cats" in tokens
    assert "running" in tokens
    assert "dogs" in tokens


def test_clean_text_for_tfidf_preserves_numbers_and_currency() -> None:
    """LIAR statements contain many numerical claims; TF-IDF path must keep them."""

    text = "Taxes rose by $600 billion, a 40% increase in 2008."
    cleaned = clean_text_for_tfidf(text)
    tokens = cleaned.split()
    # Numbers and currency markers survive.
    assert "2008" in tokens
    assert any(tok.startswith("$") for tok in tokens)
    assert any(tok.endswith("%") for tok in tokens)


def test_build_credibility_vector_is_normalized() -> None:
    """Credibility vectors should sum to one when counts are present."""

    row = pd.Series(
        {
            "barely_true_counts": 1,
            "false_counts": 2,
            "half_true_counts": 3,
            "mostly_true_counts": 4,
            "pants_on_fire_counts": 0,
        }
    )
    vector = build_credibility_vector(row)
    assert len(vector) == 5
    assert abs(sum(vector) - 1.0) < 1e-9
    assert vector[0] == 0.1


def test_build_credibility_vector_defaults_to_uniform_distribution() -> None:
    """Rows with zero counts should receive a uniform fallback vector."""

    row = pd.Series(
        {
            "barely_true_counts": 0,
            "false_counts": 0,
            "half_true_counts": 0,
            "mostly_true_counts": 0,
            "pants_on_fire_counts": 0,
        }
    )
    assert build_credibility_vector(row) == [0.2, 0.2, 0.2, 0.2, 0.2]


def test_preprocess_dataframe_adds_expected_columns() -> None:
    """DataFrame preprocessing should add text and credibility features."""

    frame = pd.DataFrame(
        [
            {
                "statement": "This is only a test statement.",
                "barely_true_counts": 1,
                "false_counts": 1,
                "half_true_counts": 1,
                "mostly_true_counts": 1,
                "pants_on_fire_counts": 1,
            }
        ]
    )
    processed = preprocess_dataframe(frame)
    expected_columns = {
        "statement_clean",
        "statement_transformer",
        "statement_raw",
        "credibility_vector",
        "credibility_0",
        "credibility_1",
        "credibility_2",
        "credibility_3",
        "credibility_4",
    }
    assert expected_columns.issubset(processed.columns)
    assert len(processed.loc[0, "credibility_vector"]) == 5
    assert abs(sum(processed.loc[0, "credibility_vector"]) - 1.0) < 0.01
