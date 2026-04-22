"""Preprocessing utilities for LIAR text and metadata features."""

from __future__ import annotations

import html
import re
from typing import Iterable

import pandas as pd
from nltk.stem import PorterStemmer


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<[^>]+>")
NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")
MULTISPACE_PATTERN = re.compile(r"\s+")
TOKEN_PATTERN = re.compile(r"[a-z]+")

DEFAULT_CREDIBILITY_VECTOR = [0.2] * 5
ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "before", "being", "between",
    "both", "but", "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each",
    "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "will", "with", "you", "your", "yours", "yourself", "yourselves",
}
STEMMER = PorterStemmer()


def _normalize_text(text: str) -> str:
    """Normalize raw text input into a safe lowercase string.

    Args:
        text: Raw input text.

    Returns:
        Lowercased text with HTML entities unescaped.
    """

    if pd.isna(text):
        return ""

    normalized = html.unescape(str(text)).lower().strip()
    normalized = URL_PATTERN.sub(" ", normalized)
    normalized = HTML_PATTERN.sub(" ", normalized)
    return MULTISPACE_PATTERN.sub(" ", normalized).strip()


def _lemmatize_tokens(tokens: Iterable[str]) -> list[str]:
    """Normalize tokens for TF-IDF-friendly lexical reduction.

    Args:
        tokens: Tokenized lowercase words.

    Returns:
        Normalized tokens.
    """

    return [STEMMER.stem(token) for token in tokens]


def clean_text_for_tfidf(text: str) -> str:
    """Clean text for TF-IDF features.

    Args:
        text: Raw statement text.

    Returns:
        Lowercased text with URLs and HTML removed, normalized, and filtered for stopwords.
    """

    normalized = _normalize_text(text)
    alpha_only = NON_ALPHA_PATTERN.sub(" ", normalized)
    tokens = TOKEN_PATTERN.findall(alpha_only)
    if not tokens:
        return ""

    filtered_tokens = [token for token in tokens if token not in ENGLISH_STOPWORDS]
    if not filtered_tokens:
        return ""

    return " ".join(_lemmatize_tokens(filtered_tokens))


def clean_text_for_transformer(text: str) -> str:
    """Lightly clean text for RoBERTa input.

    Args:
        text: Raw statement text.

    Returns:
        Lowercased text with URLs removed and whitespace normalized.
    """

    normalized = _normalize_text(text)
    return MULTISPACE_PATTERN.sub(" ", normalized).strip()


def build_credibility_vector(row: pd.Series) -> list[float]:
    """Build a normalized five-dimensional credibility vector.

    Args:
        row: DataFrame row containing LIAR credibility count fields.

    Returns:
        List of normalized credibility values ordered as
        `[barely_true, false, half_true, mostly_true, pants_on_fire]`.
        If the total count is zero, returns a uniform vector.
    """

    columns = [
        "barely_true_counts",
        "false_counts",
        "half_true_counts",
        "mostly_true_counts",
        "pants_on_fire_counts",
    ]
    counts = [float(pd.to_numeric(row.get(column, 0), errors="coerce") or 0.0) for column in columns]
    total = sum(counts)
    if total <= 0:
        return DEFAULT_CREDIBILITY_VECTOR.copy()

    return [count / total for count in counts]


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text and credibility preprocessing to a LIAR DataFrame.

    Args:
        df: Input LIAR DataFrame.

    Returns:
        Copy of the input DataFrame augmented with `statement_clean`,
        `statement_transformer`, `statement_raw`, `credibility_vector`, and
        `credibility_0` through `credibility_4`.
    """

    processed = df.copy()
    processed["statement_clean"] = processed["statement"].apply(clean_text_for_tfidf)
    processed["statement_transformer"] = processed["statement"].apply(clean_text_for_transformer)
    processed["statement_raw"] = processed["statement_transformer"]
    processed["credibility_vector"] = processed.apply(build_credibility_vector, axis=1)

    expanded = pd.DataFrame(
        processed["credibility_vector"].tolist(),
        columns=[f"credibility_{index}" for index in range(5)],
        index=processed.index,
    )
    return pd.concat([processed, expanded], axis=1)
