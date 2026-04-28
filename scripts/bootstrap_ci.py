"""Bootstrap confidence intervals for LIAR test-set prediction artifacts.

The canonical Kaggle archive in this checkout stores the 21 prediction JSONL
files under `reports/seed_{42,1337,2024}/predictions/` (seven systems per
seed). The top-level `reports/predictions/` directory is a primary-seed
convenience copy, so this script prefers the seed-scoped layout and falls back
to seed-suffixed files under `reports/predictions/` if needed.

Seed aggregation follows the dissertation roadmap: each `(system, seed)` pair
is bootstrapped independently on the 1283-example test set; the reported
per-system point estimate and CI bounds are simple averages of the per-seed
point estimates and per-seed percentile bounds. This keeps the table aligned
with the paper's seed-mean reporting convention while documenting that the CI
is a descriptive average of seed-level bootstrap intervals, not a hierarchical
bootstrap over training randomness.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


np.random.seed(0)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_N_RESAMPLES = 10_000
SEEDS = (42, 1337, 2024)
LABEL_NAMES = (
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true",
)
LABEL_IDS = tuple(range(len(LABEL_NAMES)))
SYSTEMS = (
    "baseline_naive_bayes",
    "baseline_svm",
    "baseline_random_forest",
    "transformer",
    "hybrid",
    "hybrid_textonly",
    "hybrid_leaky",
)
SYSTEM_DISPLAY_NAMES = {
    "baseline_naive_bayes": "TF-IDF + Naive Bayes",
    "baseline_svm": "TF-IDF + Linear SVM",
    "baseline_random_forest": "TF-IDF + Random Forest",
    "transformer": "RoBERTa text-only",
    "hybrid": "Hybrid, corrected",
    "hybrid_textonly": "Hybrid text-only ablation",
    "hybrid_leaky": "Hybrid, leaky",
}
COMPARISON_PAIRS = {
    "hybrid_corrected_vs_transformer": ("hybrid", "transformer"),
    "hybrid_corrected_vs_hybrid_textonly": ("hybrid", "hybrid_textonly"),
    "hybrid_leaky_vs_hybrid_corrected": ("hybrid_leaky", "hybrid"),
    "hybrid_leaky_vs_transformer": ("hybrid_leaky", "transformer"),
}


@dataclass(frozen=True)
class PredictionSet:
    """Prediction arrays loaded from one `(system, seed)` JSONL artifact."""

    system: str
    seed: int
    ids: tuple[str, ...]
    y_true: np.ndarray
    y_pred: np.ndarray
    path: Path


def compute_metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    """Compute accuracy, macro-F1, and per-class F1 for one prediction vector."""

    per_class = f1_score(
        y_true,
        y_pred,
        labels=list(LABEL_IDS),
        average=None,
        zero_division=0,
    )
    return {
        "macro_f1": float(np.mean(per_class)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_f1": {
            label: float(value)
            for label, value in zip(LABEL_NAMES, per_class, strict=True)
        },
    }


def discover_prediction_files(
    reports_dir: Path,
    systems: Sequence[str] = SYSTEMS,
    seeds: Sequence[int] = SEEDS,
) -> dict[tuple[str, int], Path]:
    """Find the complete 7-system by 3-seed prediction artifact set.

    Args:
        reports_dir: Repository `reports/` directory.
        systems: Expected stable system identifiers.
        seeds: Expected training seeds.

    Returns:
        Mapping from `(system, seed)` to prediction JSONL path.

    Raises:
        FileNotFoundError: If the complete 21-file prediction set cannot be found.
    """

    seed_scoped = _discover_seed_scoped_files(reports_dir, systems, seeds)
    if len(seed_scoped) == len(systems) * len(seeds):
        return seed_scoped

    top_level = _discover_top_level_seed_files(reports_dir, systems, seeds)
    if len(top_level) == len(systems) * len(seeds):
        return top_level

    missing = [
        f"{system}/seed{seed}"
        for seed in seeds
        for system in systems
        if (system, seed) not in seed_scoped and (system, seed) not in top_level
    ]
    raise FileNotFoundError(
        "Could not find complete prediction artifact set. Missing: "
        + ", ".join(missing)
    )


def aggregate_seed_intervals(per_seed: Mapping[str, dict[str, object]]) -> dict[str, object]:
    """Average per-seed point estimates and CI bounds for a system.

    This intentionally averages each seed's bootstrap `point`, `ci_low`,
    `median`, and `ci_high` values independently, matching the dissertation
    roadmap's requested reporting convention for seed-mean summaries.
    """

    seed_summaries = list(per_seed.values())
    return {
        "macro_f1": _average_interval([s["macro_f1"] for s in seed_summaries]),
        "accuracy": _average_interval([s["accuracy"] for s in seed_summaries]),
        "per_class_f1": {
            label: _average_interval([
                s["per_class_f1"][label]
                for s in seed_summaries
                if label in s["per_class_f1"]
            ])
            for label in _ordered_present_labels(seed_summaries)
        },
    }


def bootstrap_metric_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = DEFAULT_N_RESAMPLES,
) -> dict[str, object]:
    """Bootstrap metric percentile intervals for one `(system, seed)` run."""

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if len(y_true) == 0:
        raise ValueError("cannot bootstrap an empty prediction set.")

    point = compute_metric_bundle(y_true, y_pred)
    macro_values = np.empty(n_resamples, dtype=float)
    accuracy_values = np.empty(n_resamples, dtype=float)
    per_class_values = np.empty((n_resamples, len(LABEL_NAMES)), dtype=float)

    n_examples = len(y_true)
    for sample_index in range(n_resamples):
        indices = np.random.randint(0, n_examples, size=n_examples)
        macro, accuracy, per_class = _metric_values(y_true[indices], y_pred[indices])
        macro_values[sample_index] = macro
        accuracy_values[sample_index] = accuracy
        per_class_values[sample_index] = per_class

    return {
        "macro_f1": _percentile_interval(macro_values, float(point["macro_f1"])),
        "accuracy": _percentile_interval(accuracy_values, float(point["accuracy"])),
        "per_class_f1": {
            label: _percentile_interval(
                per_class_values[:, label_id],
                float(point["per_class_f1"][label]),
            )
            for label_id, label in enumerate(LABEL_NAMES)
        },
    }


def paired_bootstrap_comparison(
    per_seed_arrays: Mapping[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_resamples: int = DEFAULT_N_RESAMPLES,
) -> dict[str, object]:
    """Compute paired seed-mean bootstrap differences for two systems.

    Each mapping value is `(y_true, pred_a, pred_b)` aligned by example ID.
    For each bootstrap sample, the same resampled example indices are applied
    to every seed and the per-seed `metric(A) - metric(B)` values are averaged.
    The resulting distribution estimates uncertainty in the reported seed-mean
    difference due to test-set sampling.
    """

    if not per_seed_arrays:
        raise ValueError("per_seed_arrays must contain at least one seed.")

    lengths = {len(arrays[0]) for arrays in per_seed_arrays.values()}
    if len(lengths) != 1:
        raise ValueError("all seeds must have the same number of aligned examples.")
    n_examples = lengths.pop()
    if n_examples == 0:
        raise ValueError("cannot bootstrap empty paired predictions.")

    point = _paired_point_difference(per_seed_arrays)
    macro_values = np.empty(n_resamples, dtype=float)
    accuracy_values = np.empty(n_resamples, dtype=float)
    per_class_values = np.empty((n_resamples, len(LABEL_NAMES)), dtype=float)

    for sample_index in range(n_resamples):
        indices = np.random.randint(0, n_examples, size=n_examples)
        macro_diffs = []
        accuracy_diffs = []
        per_class_diffs = []
        for y_true, pred_a, pred_b in per_seed_arrays.values():
            macro_a, accuracy_a, per_class_a = _metric_values(
                y_true[indices],
                pred_a[indices],
            )
            macro_b, accuracy_b, per_class_b = _metric_values(
                y_true[indices],
                pred_b[indices],
            )
            macro_diffs.append(macro_a - macro_b)
            accuracy_diffs.append(accuracy_a - accuracy_b)
            per_class_diffs.append(per_class_a - per_class_b)

        macro_values[sample_index] = float(np.mean(macro_diffs))
        accuracy_values[sample_index] = float(np.mean(accuracy_diffs))
        per_class_values[sample_index] = np.mean(np.vstack(per_class_diffs), axis=0)

    return {
        "macro_f1_diff": _difference_interval(macro_values, point["macro_f1_diff"]),
        "accuracy_diff": _difference_interval(accuracy_values, point["accuracy_diff"]),
        "per_class_diff": {
            label: _difference_interval(
                per_class_values[:, label_id],
                point["per_class_diff"][label],
            )
            for label_id, label in enumerate(LABEL_NAMES)
        },
    }


def build_bootstrap_report(
    reports_dir: Path = REPORTS_DIR,
    n_resamples: int = DEFAULT_N_RESAMPLES,
) -> dict[str, object]:
    """Build the complete bootstrap CI report from prediction JSONLs."""

    prediction_sets = load_prediction_sets(reports_dir)
    per_system: dict[str, dict[str, object]] = {}

    for system in SYSTEMS:
        logging.info("Bootstrapping per-seed metrics for %s", system)
        system_report: dict[str, object] = {}
        for seed in SEEDS:
            prediction_set = prediction_sets[(system, seed)]
            system_report[f"seed{seed}"] = bootstrap_metric_intervals(
                prediction_set.y_true,
                prediction_set.y_pred,
                n_resamples=n_resamples,
            )
        seed_only = {
            key: value
            for key, value in system_report.items()
            if key.startswith("seed")
        }
        system_report["averaged"] = aggregate_seed_intervals(seed_only)
        per_system[system] = system_report

    paired_comparisons: dict[str, object] = {}
    for comparison_name, (system_a, system_b) in COMPARISON_PAIRS.items():
        logging.info("Running paired bootstrap for %s", comparison_name)
        paired_comparisons[comparison_name] = paired_bootstrap_comparison(
            _build_pair_arrays(prediction_sets, system_a, system_b),
            n_resamples=n_resamples,
        )

    return {
        "metadata": {
            "n_resamples": n_resamples,
            "seeds": list(SEEDS),
            "systems": list(SYSTEMS),
            "label_names": list(LABEL_NAMES),
            "ci_percentiles": [2.5, 50.0, 97.5],
            "seed_aggregation": (
                "Per-system rows average per-seed point estimates and per-seed "
                "bootstrap percentile bounds."
            ),
            "prediction_layout": (
                "Canonical 21 files read from reports/seed_*/predictions; "
                "top-level reports/predictions is treated as a fallback only."
            ),
        },
        "per_system": per_system,
        "paired_comparisons": paired_comparisons,
    }


def load_prediction_sets(reports_dir: Path) -> dict[tuple[str, int], PredictionSet]:
    """Load all discovered prediction artifacts into arrays."""

    prediction_paths = discover_prediction_files(reports_dir)
    loaded = {
        key: load_prediction_set(path)
        for key, path in sorted(prediction_paths.items(), key=lambda item: (item[0][1], item[0][0]))
    }
    for (expected_system, expected_seed), prediction_set in loaded.items():
        if prediction_set.system != expected_system:
            raise ValueError(
                f"system mismatch in {prediction_set.path}: expected "
                f"{expected_system}, got {prediction_set.system}"
            )
        if prediction_set.seed != expected_seed:
            raise ValueError(
                f"seed mismatch in {prediction_set.path}: expected "
                f"{expected_seed}, got {prediction_set.seed}"
            )
    return loaded


def load_prediction_set(path: Path) -> PredictionSet:
    """Load one prediction JSONL artifact."""

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"prediction file is empty: {path}")

    system = str(records[0]["model"])
    seed = int(records[0]["seed"])
    ids = tuple(str(row["id"]) for row in records)
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate example IDs in {path}")

    y_true = np.array([int(row["label_id"]) for row in records], dtype=int)
    y_pred = np.array([int(row["pred_label_id"]) for row in records], dtype=int)
    return PredictionSet(
        system=system,
        seed=seed,
        ids=ids,
        y_true=y_true,
        y_pred=y_pred,
        path=path,
    )


def write_bootstrap_table(report: Mapping[str, object], output_path: Path) -> None:
    """Write a compact LaTeX table with seed-averaged macro-F1 CIs."""

    per_system = report["per_system"]
    lines = [
        "% Generated by scripts/bootstrap_ci.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Seed-averaged test performance with bootstrap confidence intervals.}",
        "\\label{tab:bootstrap-ci}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "System & Test macro-F1 [95\\% CI] & Test accuracy [95\\% CI] \\\\",
        "\\midrule",
    ]
    for system in SYSTEMS:
        averaged = per_system[system]["averaged"]
        macro = _format_ci_cell(averaged["macro_f1"])
        accuracy = _format_ci_cell(averaged["accuracy"])
        lines.append(f"{SYSTEM_DISPLAY_NAMES[system]} & {macro} & {accuracy} \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _discover_seed_scoped_files(
    reports_dir: Path,
    systems: Sequence[str],
    seeds: Sequence[int],
) -> dict[tuple[str, int], Path]:
    found = {}
    for seed in seeds:
        predictions_dir = reports_dir / f"seed_{seed}" / "predictions"
        for system in systems:
            path = predictions_dir / f"{system}_test_predictions.jsonl"
            if path.exists():
                found[(system, seed)] = path
    return found


def _discover_top_level_seed_files(
    reports_dir: Path,
    systems: Sequence[str],
    seeds: Sequence[int],
) -> dict[tuple[str, int], Path]:
    found = {}
    predictions_dir = reports_dir / "predictions"
    candidates = (
        "{system}_seed{seed}.jsonl",
        "{system}_test_predictions_seed{seed}.jsonl",
        "{system}_seed{seed}_test_predictions.jsonl",
    )
    for seed in seeds:
        for system in systems:
            for pattern in candidates:
                path = predictions_dir / pattern.format(system=system, seed=seed)
                if path.exists():
                    found[(system, seed)] = path
                    break
    return found


def _metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, np.ndarray]:
    per_class = f1_score(
        y_true,
        y_pred,
        labels=list(LABEL_IDS),
        average=None,
        zero_division=0,
    )
    return (
        float(np.mean(per_class)),
        float(accuracy_score(y_true, y_pred)),
        np.asarray(per_class, dtype=float),
    )


def _percentile_interval(values: np.ndarray, point: float) -> dict[str, float]:
    ci_low, median, ci_high = np.percentile(values, [2.5, 50.0, 97.5])
    return {
        "point": float(point),
        "ci_low": float(ci_low),
        "median": float(median),
        "ci_high": float(ci_high),
    }


def _difference_interval(values: np.ndarray, point: float) -> dict[str, float]:
    interval = _percentile_interval(values, point)
    if np.isclose(point, 0.0):
        p_value = 1.0
    elif point > 0:
        p_value = min(1.0, 2.0 * float(np.mean(values <= 0.0)))
    else:
        p_value = min(1.0, 2.0 * float(np.mean(values >= 0.0)))
    interval["p_value"] = float(p_value)
    return interval


def _average_interval(intervals: Sequence[object]) -> dict[str, float]:
    if not intervals:
        raise ValueError("cannot average an empty interval list.")
    keys = ("point", "ci_low", "median", "ci_high")
    return {
        key: float(np.mean([float(interval[key]) for interval in intervals]))
        for key in keys
    }


def _ordered_present_labels(seed_summaries: Sequence[dict[str, object]]) -> list[str]:
    present = {
        label
        for summary in seed_summaries
        for label in summary["per_class_f1"]
    }
    return [label for label in LABEL_NAMES if label in present]


def _build_pair_arrays(
    prediction_sets: Mapping[tuple[str, int], PredictionSet],
    system_a: str,
    system_b: str,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return {
        seed: _align_prediction_pair(
            prediction_sets[(system_a, seed)],
            prediction_sets[(system_b, seed)],
        )
        for seed in SEEDS
    }


def _align_prediction_pair(
    set_a: PredictionSet,
    set_b: PredictionSet,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_id_a = {example_id: idx for idx, example_id in enumerate(set_a.ids)}
    by_id_b = {example_id: idx for idx, example_id in enumerate(set_b.ids)}
    if set(by_id_a) != set(by_id_b):
        missing_a = sorted(set(by_id_b) - set(by_id_a))[:5]
        missing_b = sorted(set(by_id_a) - set(by_id_b))[:5]
        raise ValueError(
            f"ID mismatch for {set_a.system} vs {set_b.system}: "
            f"missing from A={missing_a}, missing from B={missing_b}"
        )

    ordered_ids = tuple(sorted(by_id_a))
    true_labels = np.array([set_a.y_true[by_id_a[example_id]] for example_id in ordered_ids])
    true_labels_b = np.array([set_b.y_true[by_id_b[example_id]] for example_id in ordered_ids])
    if not np.array_equal(true_labels, true_labels_b):
        raise ValueError(f"true-label mismatch for {set_a.system} vs {set_b.system}")

    pred_a = np.array([set_a.y_pred[by_id_a[example_id]] for example_id in ordered_ids])
    pred_b = np.array([set_b.y_pred[by_id_b[example_id]] for example_id in ordered_ids])
    return true_labels, pred_a, pred_b


def _paired_point_difference(
    per_seed_arrays: Mapping[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, object]:
    macro_diffs = []
    accuracy_diffs = []
    per_class_diffs = []
    for y_true, pred_a, pred_b in per_seed_arrays.values():
        macro_a, accuracy_a, per_class_a = _metric_values(y_true, pred_a)
        macro_b, accuracy_b, per_class_b = _metric_values(y_true, pred_b)
        macro_diffs.append(macro_a - macro_b)
        accuracy_diffs.append(accuracy_a - accuracy_b)
        per_class_diffs.append(per_class_a - per_class_b)

    per_class_mean = np.mean(np.vstack(per_class_diffs), axis=0)
    return {
        "macro_f1_diff": float(np.mean(macro_diffs)),
        "accuracy_diff": float(np.mean(accuracy_diffs)),
        "per_class_diff": {
            label: float(per_class_mean[label_id])
            for label_id, label in enumerate(LABEL_NAMES)
        },
    }


def _format_ci_cell(interval: Mapping[str, float]) -> str:
    return (
        f"{interval['point']:.3f} "
        f"[{interval['ci_low']:.3f}, {interval['ci_high']:.3f}]"
    )


def main() -> None:
    """Run the bootstrap analysis and write report artifacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Path to the reports directory containing prediction artifacts.",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=DEFAULT_N_RESAMPLES,
        help="Number of bootstrap resamples per interval.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    report = build_bootstrap_report(
        reports_dir=args.reports_dir,
        n_resamples=args.n_resamples,
    )
    json_path = args.reports_dir / "bootstrap_ci.json"
    table_path = args.reports_dir / "bootstrap_ci_table.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_bootstrap_table(report, table_path)

    logging.info("Wrote %s", json_path)
    logging.info("Wrote %s", table_path)


if __name__ == "__main__":
    main()
