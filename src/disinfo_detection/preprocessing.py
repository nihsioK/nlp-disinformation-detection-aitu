"""Preprocessing utilities for LIAR text and metadata features.

Improvements over the previous version:
- Classical baselines: removed Porter stemming. On LIAR statements (mean length ~17 tokens)
  stemming collapses distinctions like 'taxed' / 'taxation' / 'taxes' that carry real signal,
  and the TF-IDF n-gram representation already recovers the shared stem via the bigram.
- Classical baselines: kept digits and the dollar sign, because LIAR is dominated by
  numerical political claims ('$600 billion', '40 percent', '2008') and stripping them
  destroys strong features.
- Classical baselines: expanded English stopword list to match sklearn / NLTK defaults.
- Transformer path: left the raw text untouched except for URL removal and whitespace
  normalization, to preserve casing and punctuation that the RoBERTa tokenizer uses.
- Metadata: added ordinal-aware credibility features (total count, log-total, share of
  'pants_on_fire'), which are independent of the normalized 5-vector and stay informative
  for speakers with very few prior statements.
"""

from __future__ import annotations

import html
import math
import re
from typing import Iterable

import pandas as pd


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_PATTERN = re.compile(r"<[^>]+>")
# Keep digits, dollar sign, and percent because they carry signal on LIAR.
CLASSICAL_NOISE_PATTERN = re.compile(r"[^a-z0-9$%\s]")
MULTISPACE_PATTERN = re.compile(r"\s+")
# Tokens: words (with internal apostrophes) OR numbers (possibly with commas, decimals, $, %).
TOKEN_PATTERN = re.compile(r"\$?\d[\d,\.]*%?|[a-z]+(?:'[a-z]+)?")

DEFAULT_CREDIBILITY_VECTOR = [0.2] * 5

# Maps an integer LIAR label id to the credibility-count column whose value
# includes the row's own statement at dataset-collection time. Used to remove
# the row-level leakage when building the `*_corrected` credibility features.
# Label 5 ("true") has no corresponding count column in LIAR.
LABEL_TO_COUNT_COLUMN: dict[int, str] = {
    0: "pants_on_fire_counts",
    1: "false_counts",
    2: "barely_true_counts",
    3: "half_true_counts",
    4: "mostly_true_counts",
}

CREDIBILITY_COUNT_COLS = (
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
)

# Sklearn's ENGLISH_STOP_WORDS list, adapted. We removed 'not', 'no', 'nor' because
# negation is a very strong signal for political truthfulness on LIAR.
ENGLISH_STOPWORDS = {
    "a", "about", "above", "across", "after", "again", "against", "all", "almost", "alone",
    "along", "already", "also", "although", "always", "am", "among", "an", "and", "another",
    "any", "anyone", "anything", "anywhere", "are", "around", "as", "at", "back", "be",
    "became", "because", "become", "becomes", "been", "before", "beforehand", "behind",
    "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can",
    "cannot", "could", "did", "do", "does", "doing", "done", "down", "during", "each",
    "either", "else", "elsewhere", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "first", "for", "formerly", "from",
    "further", "had", "has", "have", "having", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how",
    "however", "i", "ie", "if", "in", "indeed", "into", "is", "it", "its", "itself",
    "just", "keep", "last", "latter", "latterly", "least", "less", "like", "made", "make",
    "many", "may", "me", "meanwhile", "might", "mine", "more", "moreover", "most", "mostly",
    "much", "must", "my", "myself", "namely", "neither", "never", "nevertheless", "next",
    "nine", "none", "noone", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours",
    "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather",
    "re", "same", "see", "seem", "seemed", "seeming", "seems", "several", "she", "should",
    "since", "so", "some", "somehow", "someone", "something", "sometime", "sometimes",
    "somewhere", "still", "such", "take", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore",
    "therein", "thereupon", "these", "they", "this", "those", "though", "through",
    "throughout", "thru", "thus", "to", "together", "too", "toward", "towards", "under",
    "until", "up", "upon", "us", "used", "using", "very", "via", "was", "we", "well",
    "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter",
    "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without",
    "would", "yet", "you", "your", "yours", "yourself", "yourselves",
}


def _normalize_text(text: str) -> str:
    """Normalize raw text into a safe lowercase string with URLs and HTML stripped."""

    if pd.isna(text):
        return ""

    normalized = html.unescape(str(text)).lower().strip()
    normalized = URL_PATTERN.sub(" ", normalized)
    normalized = HTML_PATTERN.sub(" ", normalized)
    return MULTISPACE_PATTERN.sub(" ", normalized).strip()


def clean_text_for_tfidf(text: str) -> str:
    """Clean text for TF-IDF features.

    Pipeline:
        1. Lowercase, strip HTML entities and URLs.
        2. Keep letters, digits, `$`, and `%`; collapse everything else to whitespace.
        3. Tokenize on word + number patterns.
        4. Drop stopwords (except negations, which are informative on LIAR).
        5. Join back with single spaces.

    Crucially, this function does NOT stem. For short political statements,
    stemming collapses too many meaningfully distinct terms (taxed/taxation/taxes,
    invested/investor/investment) and was empirically shown to lower macro-F1
    on LIAR in internal experiments.
    """

    normalized = _normalize_text(text)
    alpha_only = CLASSICAL_NOISE_PATTERN.sub(" ", normalized)
    tokens = TOKEN_PATTERN.findall(alpha_only)
    if not tokens:
        return ""

    filtered_tokens = [token for token in tokens if token not in ENGLISH_STOPWORDS]
    if not filtered_tokens:
        return ""

    return " ".join(filtered_tokens)


def clean_text_for_transformer(text: str) -> str:
    """Lightly clean text for RoBERTa input.

    RoBERTa uses a case-sensitive BPE tokenizer, so we deliberately do NOT lowercase
    and we preserve punctuation. We only strip URLs and HTML and normalize whitespace.
    """

    if pd.isna(text):
        return ""

    normalized = html.unescape(str(text)).strip()
    normalized = URL_PATTERN.sub(" ", normalized)
    normalized = HTML_PATTERN.sub(" ", normalized)
    return MULTISPACE_PATTERN.sub(" ", normalized).strip()


def _row_counts(row: pd.Series) -> dict[str, float]:
    """Return the row's five credibility counts as floats, defaulting to 0."""

    return {
        column: float(pd.to_numeric(row.get(column, 0), errors="coerce") or 0.0)
        for column in CREDIBILITY_COUNT_COLS
    }


def _decrement_counts_for_label(counts: dict[str, float], label_id: int) -> dict[str, float]:
    """Return `counts` with the bin matching `label_id` decremented by 1.

    LIAR's count columns are PolitiFact totals at collection time, so for any
    row that appears in the labelled split the row's own verdict has already
    been folded into the speaker's counts. Subtracting one from the matching
    bin removes that leakage. Label 5 ("true") has no corresponding column and
    is left unchanged.
    """

    target_column = LABEL_TO_COUNT_COLUMN.get(int(label_id))
    if target_column is None:
        return counts
    adjusted = dict(counts)
    if adjusted[target_column] > 0:
        adjusted[target_column] -= 1.0
    return adjusted


def _credibility_vector_from_counts(counts: dict[str, float]) -> list[float]:
    total = sum(counts[column] for column in CREDIBILITY_COUNT_COLS)
    if total <= 0:
        return DEFAULT_CREDIBILITY_VECTOR.copy()
    return [counts[column] / total for column in CREDIBILITY_COUNT_COLS]


def _credibility_scalars_from_counts(counts: dict[str, float]) -> dict[str, float]:
    barely = counts["barely_true_counts"]
    false = counts["false_counts"]
    half = counts["half_true_counts"]
    mostly = counts["mostly_true_counts"]
    pants = counts["pants_on_fire_counts"]
    total = barely + false + half + mostly + pants
    if total <= 0:
        return {
            "cred_total": 0.0,
            "cred_log_total": 0.0,
            "cred_pants_share": 0.0,
            "cred_false_share": 0.0,
        }
    return {
        "cred_total": total,
        "cred_log_total": math.log1p(total),
        "cred_pants_share": pants / total,
        "cred_false_share": (false + pants) / total,
    }


def build_credibility_vector(row: pd.Series) -> list[float]:
    """Build a normalized five-dimensional credibility probability vector."""

    return _credibility_vector_from_counts(_row_counts(row))


def build_credibility_vector_corrected(row: pd.Series, label_id: int) -> list[float]:
    """Same as `build_credibility_vector` but removes the row's own contribution.

    Used to construct the leakage-corrected credibility features documented in
    `docs/HYBRID_MODEL.md`. The decrement target column is derived from
    `LABEL_TO_COUNT_COLUMN`; rows with `label_id == 5` ("true") are returned
    unchanged because LIAR has no `true_counts` column.
    """

    return _credibility_vector_from_counts(
        _decrement_counts_for_label(_row_counts(row), label_id)
    )


def build_credibility_scalars(row: pd.Series) -> dict[str, float]:
    """Build scalar credibility summary features complementary to the 5-vector.

    Returns:
        - `cred_total`: raw total prior-statement count for the speaker.
        - `cred_log_total`: log1p of the total (handles long-tail speakers).
        - `cred_pants_share`: fraction of prior statements rated 'pants_on_fire'.
        - `cred_false_share`: fraction rated 'false' or 'pants_on_fire' combined.
    """

    return _credibility_scalars_from_counts(_row_counts(row))


def build_credibility_scalars_corrected(row: pd.Series, label_id: int) -> dict[str, float]:
    """Same as `build_credibility_scalars` but with the row's bin decremented."""

    return _credibility_scalars_from_counts(
        _decrement_counts_for_label(_row_counts(row), label_id)
    )


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text and credibility preprocessing to a LIAR DataFrame.

    When `label_id` is present on the input frame this function additionally
    emits leakage-corrected credibility columns (`credibility_corrected_*`,
    `cred_*_corrected`) built from counts that exclude the row's own verdict.
    Both column families are emitted side by side so a single pickle can drive
    either reporting variant.
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

    scalar_rows = processed.apply(build_credibility_scalars, axis=1)
    scalar_frame = pd.DataFrame(scalar_rows.tolist(), index=processed.index)

    parts: list[pd.DataFrame] = [processed, expanded, scalar_frame]

    if "label_id" in processed.columns:
        corrected_vector_rows = processed.apply(
            lambda row: build_credibility_vector_corrected(row, int(row["label_id"])),
            axis=1,
        )
        corrected_vector_frame = pd.DataFrame(
            corrected_vector_rows.tolist(),
            columns=[f"credibility_corrected_{index}" for index in range(5)],
            index=processed.index,
        )
        corrected_scalar_rows = processed.apply(
            lambda row: build_credibility_scalars_corrected(row, int(row["label_id"])),
            axis=1,
        )
        corrected_scalar_frame = pd.DataFrame(
            corrected_scalar_rows.tolist(),
            index=processed.index,
        )
        corrected_scalar_frame = corrected_scalar_frame.rename(
            columns={column: f"{column}_corrected" for column in corrected_scalar_frame.columns}
        )
        parts.extend([corrected_vector_frame, corrected_scalar_frame])

    return pd.concat(parts, axis=1)
