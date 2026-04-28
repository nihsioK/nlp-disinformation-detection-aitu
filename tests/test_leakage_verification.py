"""Unit tests for the leakage-verification routines."""

from __future__ import annotations

import pandas as pd
import pytest

from src.disinfo_detection.preprocessing import (
    LABEL_TO_COUNT_COLUMN,
    _decrement_counts_for_label,
    _row_counts,
    build_credibility_vector,
    build_credibility_vector_corrected,
)


def _synthetic_row(
    *,
    label_id: int,
    barely_true: float = 0.0,
    false_count: float = 0.0,
    half_true: float = 0.0,
    mostly_true: float = 0.0,
    pants_on_fire: float = 0.0,
) -> pd.Series:
    """Build a synthetic LIAR-style row with the given counts and label."""

    return pd.Series({
        "id": "synthetic.json",
        "label_id": label_id,
        "barely_true_counts": barely_true,
        "false_counts": false_count,
        "half_true_counts": half_true,
        "mostly_true_counts": mostly_true,
        "pants_on_fire_counts": pants_on_fire,
    })


class TestLeakageDelta:
    """Verify that the leave-one-out correction shifts the matching bin by exactly 1."""

    def test_speaker_with_single_pants_fire_rating_has_delta_one(self) -> None:
        row = _synthetic_row(label_id=0, pants_on_fire=1.0)
        counts = _row_counts(row)
        corrected = _decrement_counts_for_label(counts, label_id=0)
        target_col = LABEL_TO_COUNT_COLUMN[0]
        assert counts[target_col] - corrected[target_col] == pytest.approx(1.0)

    def test_corrected_vector_zero_after_removing_only_rating(self) -> None:
        row = _synthetic_row(label_id=0, pants_on_fire=1.0)
        corrected_vector = build_credibility_vector_corrected(row, label_id=0)
        # Single rating removed -> total count is 0 -> default uniform vector returned.
        assert corrected_vector == [0.2, 0.2, 0.2, 0.2, 0.2]

    def test_idempotent_on_zero_count(self) -> None:
        row = _synthetic_row(label_id=0)  # no prior ratings at all
        counts = _row_counts(row)
        corrected = _decrement_counts_for_label(counts, label_id=0)
        target_col = LABEL_TO_COUNT_COLUMN[0]
        # Already zero; correction must not push it negative.
        assert counts[target_col] - corrected[target_col] == pytest.approx(0.0)
        assert corrected[target_col] == 0.0

    def test_true_label_leaves_counts_unchanged(self) -> None:
        row = _synthetic_row(label_id=5, mostly_true=2.0, false_count=1.0)
        counts = _row_counts(row)
        corrected = _decrement_counts_for_label(counts, label_id=5)
        # No `true_counts` column exists in LIAR; correction must be a no-op.
        for col in counts:
            assert counts[col] == corrected[col]

    def test_only_target_column_changes(self) -> None:
        row = _synthetic_row(
            label_id=2,  # barely-true
            barely_true=3.0,
            false_count=2.0,
            half_true=1.0,
            mostly_true=4.0,
            pants_on_fire=1.0,
        )
        counts = _row_counts(row)
        corrected = _decrement_counts_for_label(counts, label_id=2)
        target_col = LABEL_TO_COUNT_COLUMN[2]
        for col in counts:
            expected_delta = 1.0 if col == target_col else 0.0
            assert counts[col] - corrected[col] == pytest.approx(expected_delta)

    def test_released_vector_matches_uncorrected_construction(self) -> None:
        row = _synthetic_row(label_id=1, false_count=2.0, half_true=2.0)
        # The released vector is the renormalised counts as-is.
        released = build_credibility_vector(row)
        assert released == pytest.approx([0.0, 0.5, 0.5, 0.0, 0.0])
