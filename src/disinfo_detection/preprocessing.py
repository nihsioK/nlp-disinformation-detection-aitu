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


def build_credibility_vector(row: pd.Series) -> list[float]:
    """Build a normalized five-dimensional credibility probability vector."""

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


def build_credibility_scalars(row: pd.Series) -> dict[str, float]:
    """Build scalar credibility summary features complementary to the 5-vector.

    Returns:
        - `cred_total`: raw total prior-statement count for the speaker.
        - `cred_log_total`: log1p of the total (handles long-tail speakers).
        - `cred_pants_share`: fraction of prior statements rated 'pants_on_fire'.
        - `cred_false_share`: fraction rated 'false' or 'pants_on_fire' combined.
    """

    barely = float(pd.to_numeric(row.get("barely_true_counts", 0), errors="coerce") or 0.0)
    false = float(pd.to_numeric(row.get("false_counts", 0), errors="coerce") or 0.0)
    half = float(pd.to_numeric(row.get("half_true_counts", 0), errors="coerce") or 0.0)
    mostly = float(pd.to_numeric(row.get("mostly_true_counts", 0), errors="coerce") or 0.0)
    pants = float(pd.to_numeric(row.get("pants_on_fire_counts", 0), errors="coerce") or 0.0)
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


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text and credibility preprocessing to a LIAR DataFrame."""

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

    return pd.concat([processed, expanded, scalar_frame], axis=1)
