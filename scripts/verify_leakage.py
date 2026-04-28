"""Empirical verification of LIAR credibility-history label leakage.

This script implements two complementary checks:

  1A. Counting-based verification (data side).
      For every statement in train/valid/test whose true label falls in the
      lower five classes (pants-fire/false/barely-true/half-true/mostly-true),
      compare the *released* credibility-count column for that label with the
      *leave-one-out* corrected column produced by
      `preprocessing._decrement_counts_for_label`. Aggregate the per-row delta
      to characterise how systematically the released vector folds the row's
      own verdict into the speaker's running totals.

  1B. Predictions-based verification (model side).
      Compare per-example predictions of the leakage-corrected hybrid against
      the leaky hybrid for each of the three trained seeds. Quantify how often
      the two systems disagree, and within those disagreements how often the
      leaky model recovers the true label that the corrected model missed.

The script writes machine-readable JSON to `reports/leakage_verification.json`
and `reports/leakage_verification_predictions.json`, and prints a Markdown
summary block ready to paste into the dissertation.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.disinfo_detection.data_loader import load_liar
from src.disinfo_detection.preprocessing import (
    LABEL_TO_COUNT_COLUMN,
    _decrement_counts_for_label,
    _row_counts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
LABEL_NAMES = [
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true",
]
TRANSFORMER_SEEDS = (42, 1337, 2024)


# ---------------------------------------------------------------------------
# 1A. Counting-based verification
# ---------------------------------------------------------------------------


def _per_row_delta(row: pd.Series) -> tuple[int, float | None]:
    """Return (label_id, delta) where delta = released[y] - corrected[y].

    `delta` is `None` for label_id == 5 ("true") because LIAR has no
    `true_counts` column to inspect.
    """

    label_id = int(row["label_id"])
    target_col = LABEL_TO_COUNT_COLUMN.get(label_id)
    if target_col is None:
        return label_id, None

    counts = _row_counts(row)
    corrected = _decrement_counts_for_label(counts, label_id)
    return label_id, counts[target_col] - corrected[target_col]


def _delta_table_for_split(df: pd.DataFrame) -> dict:
    """Compute the leakage delta histogram for a single split."""

    per_class = defaultdict(list)
    anomalies = []
    for _, row in df.iterrows():
        label_id, delta = _per_row_delta(row)
        if delta is None:
            continue
        per_class[label_id].append(delta)
        if delta not in (0.0, 1.0):
            anomalies.append({
                "id": str(row.get("id", "")),
                "label_id": label_id,
                "label": LABEL_NAMES[label_id],
                "delta": delta,
            })

    summary_per_class = {}
    for label_id in range(5):
        deltas = per_class.get(label_id, [])
        n = len(deltas)
        n_one = sum(1 for d in deltas if d == 1.0)
        n_zero = sum(1 for d in deltas if d == 0.0)
        n_other = n - n_one - n_zero
        mean_delta = sum(deltas) / n if n else 0.0
        # Population std (ddof=0); fine for a descriptive aggregate.
        if n:
            mean = mean_delta
            var = sum((d - mean) ** 2 for d in deltas) / n
            std_delta = var ** 0.5
        else:
            std_delta = 0.0
        summary_per_class[LABEL_NAMES[label_id]] = {
            "n": n,
            "mean_delta": mean_delta,
            "std_delta": std_delta,
            "n_delta_1": n_one,
            "n_delta_0": n_zero,
            "n_delta_other": n_other,
            "frac_delta_1": (n_one / n) if n else 0.0,
            "frac_delta_0": (n_zero / n) if n else 0.0,
        }

    overall_n = sum(s["n"] for s in summary_per_class.values())
    overall_n_one = sum(s["n_delta_1"] for s in summary_per_class.values())
    overall_n_zero = sum(s["n_delta_0"] for s in summary_per_class.values())
    overall = {
        "n": overall_n,
        "frac_delta_1": (overall_n_one / overall_n) if overall_n else 0.0,
        "frac_delta_0": (overall_n_zero / overall_n) if overall_n else 0.0,
        "n_anomalies": len(anomalies),
    }
    return {
        "per_class": summary_per_class,
        "overall": overall,
        "anomalies": anomalies[:50],  # cap for readability
    }


def run_counting_verification() -> dict:
    """Run 1A over all three LIAR splits and return the combined report."""

    report = {}
    for split in ("train", "valid", "test"):
        df = load_liar(split)
        report[split] = _delta_table_for_split(df)
    return report


# ---------------------------------------------------------------------------
# 1B. Predictions-based verification
# ---------------------------------------------------------------------------


def _read_predictions_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _predictions_path(seed: int, system: str) -> Path:
    return REPORTS_DIR / f"seed_{seed}" / "predictions" / f"{system}_test_predictions.jsonl"


def _label_id_to_name(label_id: int) -> str:
    return LABEL_NAMES[int(label_id)]


def _flip_summary(corrected: list[dict], leaky: list[dict]) -> dict:
    """Compare two prediction lists aligned by `id`."""

    by_id_corr = {row["id"]: row for row in corrected}
    by_id_leak = {row["id"]: row for row in leaky}
    common_ids = sorted(set(by_id_corr) & set(by_id_leak))
    if len(common_ids) != len(by_id_corr) or len(common_ids) != len(by_id_leak):
        raise ValueError(
            f"id mismatch: corrected={len(by_id_corr)}, leaky={len(by_id_leak)}, "
            f"common={len(common_ids)}"
        )

    n = len(common_ids)
    flips = 0
    leaky_recovers = 0
    corrected_recovers = 0
    per_class_total = defaultdict(int)
    per_class_flips = defaultdict(int)
    per_class_leaky_recovers = defaultdict(int)
    per_class_corrected_recovers = defaultdict(int)

    for example_id in common_ids:
        c = by_id_corr[example_id]
        l = by_id_leak[example_id]
        true_id = int(c["label_id"])
        # Sanity: same true label across the two prediction files.
        if int(l["label_id"]) != true_id:
            raise ValueError(f"true label mismatch on id={example_id}")
        cls = _label_id_to_name(true_id)
        per_class_total[cls] += 1
        c_pred = int(c["pred_label_id"])
        l_pred = int(l["pred_label_id"])
        if c_pred != l_pred:
            flips += 1
            per_class_flips[cls] += 1
            if l_pred == true_id and c_pred != true_id:
                leaky_recovers += 1
                per_class_leaky_recovers[cls] += 1
            elif c_pred == true_id and l_pred != true_id:
                corrected_recovers += 1
                per_class_corrected_recovers[cls] += 1

    per_class = {}
    for cls in LABEL_NAMES:
        total = per_class_total.get(cls, 0)
        if total == 0:
            continue
        per_class[cls] = {
            "n": total,
            "flips": per_class_flips.get(cls, 0),
            "frac_flips": per_class_flips.get(cls, 0) / total,
            "leaky_recovers": per_class_leaky_recovers.get(cls, 0),
            "frac_leaky_recovers_of_class": per_class_leaky_recovers.get(cls, 0) / total,
            "corrected_recovers": per_class_corrected_recovers.get(cls, 0),
            "frac_corrected_recovers_of_class": per_class_corrected_recovers.get(cls, 0) / total,
        }

    return {
        "n": n,
        "flips": flips,
        "frac_flips": flips / n if n else 0.0,
        "leaky_recovers": leaky_recovers,
        "frac_leaky_recovers_of_total": leaky_recovers / n if n else 0.0,
        "frac_leaky_recovers_of_flips": leaky_recovers / flips if flips else 0.0,
        "corrected_recovers": corrected_recovers,
        "frac_corrected_recovers_of_total": corrected_recovers / n if n else 0.0,
        "frac_corrected_recovers_of_flips": corrected_recovers / flips if flips else 0.0,
        "per_class": per_class,
    }


def run_predictions_verification() -> dict:
    """Run 1B for every available seed and aggregate."""

    per_seed = {}
    for seed in TRANSFORMER_SEEDS:
        corrected_path = _predictions_path(seed, "hybrid")
        leaky_path = _predictions_path(seed, "hybrid_leaky")
        if not corrected_path.exists() or not leaky_path.exists():
            raise FileNotFoundError(
                f"missing prediction file(s) for seed {seed}: "
                f"{corrected_path} or {leaky_path}"
            )
        corrected = _read_predictions_jsonl(corrected_path)
        leaky = _read_predictions_jsonl(leaky_path)
        per_seed[str(seed)] = _flip_summary(corrected, leaky)

    # Aggregate across seeds: simple averages of fractions.
    keys = ["frac_flips", "frac_leaky_recovers_of_total", "frac_leaky_recovers_of_flips"]
    averaged = {key: 0.0 for key in keys}
    for seed_summary in per_seed.values():
        for key in keys:
            averaged[key] += seed_summary[key]
    n_seeds = len(per_seed)
    for key in keys:
        averaged[key] /= n_seeds

    per_class_avg = defaultdict(lambda: {
        "frac_flips": 0.0,
        "frac_leaky_recovers_of_class": 0.0,
    })
    for cls in LABEL_NAMES:
        present = [s["per_class"].get(cls) for s in per_seed.values() if s["per_class"].get(cls)]
        if not present:
            continue
        per_class_avg[cls]["frac_flips"] = sum(p["frac_flips"] for p in present) / len(present)
        per_class_avg[cls]["frac_leaky_recovers_of_class"] = (
            sum(p["frac_leaky_recovers_of_class"] for p in present) / len(present)
        )

    return {
        "per_seed": per_seed,
        "averaged_across_seeds": {
            **averaged,
            "per_class": dict(per_class_avg),
        },
    }


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------


def _format_counting_md(counting: dict) -> str:
    lines = ["### Counting-based leakage verification (1A)", ""]
    lines.append(
        "| Split | Class | n | mean Δ | % Δ=1 | % Δ=0 | anomalies |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for split in ("train", "valid", "test"):
        sp = counting[split]
        for cls in LABEL_NAMES[:5]:
            row = sp["per_class"][cls]
            lines.append(
                f"| {split} | {cls} | {row['n']} | {row['mean_delta']:.3f} | "
                f"{row['frac_delta_1']*100:.1f} | {row['frac_delta_0']*100:.1f} | "
                f"{row['n_delta_other']} |"
            )
        ov = sp["overall"]
        lines.append(
            f"| {split} | **overall** | {ov['n']} | --- | "
            f"{ov['frac_delta_1']*100:.1f} | {ov['frac_delta_0']*100:.1f} | "
            f"{ov['n_anomalies']} |"
        )
    return "\n".join(lines)


def _format_predictions_md(predictions: dict) -> str:
    lines = ["### Predictions-based leakage verification (1B)", ""]
    lines.append("**Seed-averaged across 42, 1337, 2024:**")
    avg = predictions["averaged_across_seeds"]
    lines.append(
        f"- Flip rate (corrected vs leaky differ on prediction): {avg['frac_flips']*100:.1f}%"
    )
    lines.append(
        f"- Leaky-only label recovery: {avg['frac_leaky_recovers_of_total']*100:.1f}% of all examples "
        f"({avg['frac_leaky_recovers_of_flips']*100:.1f}% of flips)"
    )
    lines.append("")
    lines.append("**Per-class flip / leaky-recovery rates (averaged across seeds):**")
    lines.append("| Class | % flips | % leaky-recovers (of class) |")
    lines.append("|---|---:|---:|")
    for cls, row in avg["per_class"].items():
        lines.append(
            f"| {cls} | {row['frac_flips']*100:.1f} | {row['frac_leaky_recovers_of_class']*100:.1f} |"
        )

    lines.append("")
    lines.append("**Per-seed totals:**")
    lines.append("| Seed | n | flips | % flips | leaky-recovers | corrected-recovers |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for seed, summary in predictions["per_seed"].items():
        lines.append(
            f"| {seed} | {summary['n']} | {summary['flips']} | "
            f"{summary['frac_flips']*100:.1f} | {summary['leaky_recovers']} | "
            f"{summary['corrected_recovers']} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("all", "counting", "predictions"),
        default="all",
        help="which verification block to run.",
    )
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode in ("all", "counting"):
        counting = run_counting_verification()
        out_path = REPORTS_DIR / "leakage_verification.json"
        out_path.write_text(json.dumps(counting, indent=2), encoding="utf-8")
        print(_format_counting_md(counting))
        print()

    if args.mode in ("all", "predictions"):
        predictions = run_predictions_verification()
        out_path = REPORTS_DIR / "leakage_verification_predictions.json"
        out_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
        print(_format_predictions_md(predictions))


if __name__ == "__main__":
    main()
