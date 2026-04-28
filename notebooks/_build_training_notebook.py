"""Build notebooks/training.ipynb from a structured cell list.

Run from the repo root:

    python notebooks/_build_training_notebook.py

This keeps the notebook JSON auto-generated so reviewers can see the cell
contents as plain Python strings during code review.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "training.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CELLS: list[dict] = [
    md(
        """# LIAR disinformation detection — GPU training notebook

This notebook runs the full multi-seed training pipeline (baseline → transformer → hybrid
→ hybrid-textonly) for the thesis project `nihsioK/nlp-disinformation-detection-aitu` on a
Kaggle GPU instance (T4 × 2 recommended).

**Before running:**

1. In Kaggle, create a new Notebook → Settings → Accelerator → **`GPU T4 x2`** (recommended).
   **Do NOT pick `GPU P100`** — Kaggle's current PyTorch build does not ship CUDA kernels for
   P100's compute capability (sm_60), and transformer training will crash with
   `cudaErrorNoKernelImageForDevice`. T4 (sm_75) works out of the box. If you are forced onto
   P100, run the optional **Cell 2b** below to reinstall a compatible PyTorch build (adds
   ~3–5 minutes).
2. Settings → Internet → **On** (needed to clone GitHub + download `roberta-base`).
3. Run all cells.
4. Download `/kaggle/working/disinformation_results.zip` from the Kaggle **Output** panel.
   If `CREATE_MODELS_ZIP = True`, download `/kaggle/working/models.zip` separately.

**What this notebook does:**

- Clones the repo, installs the `ml` and `dev` extras.
- Downloads the LIAR dataset.
- Runs preprocessing and the full multi-seed sweep (baseline + transformer + hybrid +
  hybrid-textonly + hybrid-leaky) for `SEEDS = [42, 1337, 2024]`. The leakage-corrected
  hybrid is the defensible thesis default; the leaky variant is reported alongside it
  so reviewers can see the size of the credibility-count leakage gap.
- Collects metrics into `results_summary.json` and `multi_seed_summary.json`.
- Saves per-example TEST prediction JSONL files for significance, calibration, leakage, and
  error analysis.
- Runs the paper post-processing scripts and builds one downloadable results archive:
  `/kaggle/working/disinformation_results.zip`.
- Optionally builds a separate checkpoint archive: `/kaggle/working/models.zip`.
"""
    ),
    md(
        """## 0. Configuration

Edit these if you forked the repo or are experimenting with a different branch.
"""
    ),
    code(
        """import os

GITHUB_USER = "nihsioK"
REPO_NAME = "nlp-disinformation-detection-aitu"
BRANCH = "main"  # change to a feature branch if you're iterating

# Which stages to run. Flip to False to skip individual stages during debugging.
RUN_BASELINE = True
RUN_TRANSFORMER = True
RUN_HYBRID = True            # leakage-corrected hybrid (defensible thesis default)
RUN_HYBRID_TEXTONLY = True   # RQ2 ablation (text-only via the hybrid pipeline)
RUN_HYBRID_LEAKY = True      # prior-art comparison (credibility counts include the row's own verdict)
CREATE_MODELS_ZIP = True     # separate checkpoint archive; not included in disinformation_results.zip

# Canonical multi-seed run for thesis/paper reporting. Must match
# `training.seeds` in config/transformer.yaml and config/hybrid.yaml so
# Kaggle and local headline numbers come from the same seed set.
SEEDS = [42, 1337, 2024]
PRIMARY_SEED = SEEDS[0]

WORKDIR = "/kaggle/working"
REPO_DIR = f"{WORKDIR}/{REPO_NAME}"
"""
    ),
    md(
        """## 1. GPU check + compatibility guard

Sanity-check that Kaggle actually gave us a GPU, and warn early if the GPU's compute
capability is not covered by the shipped PyTorch build. On Kaggle today:

- **T4** (`sm_75`) — works out of the box.
- **P100** (`sm_60`) — **not supported** by the default `torch` wheel. Run Cell 2b below
  to reinstall a compatible build, or (better) switch the accelerator to T4 × 2.
"""
    ),
    code(
        """import subprocess

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout or "No NVIDIA GPU detected.")

try:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        supported = torch.cuda.get_arch_list()
        sm = f"sm_{cap[0]}{cap[1]}"
        print(f"\\nDevice: {name}  capability={sm}")
        print(f"PyTorch was built for: {supported}")
        if sm not in supported:
            print(
                f"\\n>>> WARNING: {sm} is NOT in {supported}. "
                "Transformer training will crash with 'no kernel image is available'.\\n"
                ">>> Fix: either switch Accelerator to 'GPU T4 x2' in the right-hand\\n"
                ">>> sidebar and restart the kernel, or run Cell 2b below to reinstall\\n"
                ">>> a PyTorch build that supports this device."
            )
        else:
            print(f"\\nOK — {sm} is supported by this PyTorch build.")
    else:
        print("\\nNo CUDA device visible to PyTorch. Enable a GPU accelerator.")
except Exception as exc:
    print(f"Could not query PyTorch CUDA state: {exc}")
"""
    ),
    md(
        """## 2b. (Optional) Reinstall PyTorch for P100 / older GPUs

**Skip this cell if the check above printed "OK".** Only run it if you are stuck on a P100
(or any GPU whose capability is not in `torch.cuda.get_arch_list()`).

This pins `torch` / `torchvision` / `torchaudio` to the official `cu121` wheels, which
include kernels for `sm_60` through `sm_90`. After it finishes, **restart the kernel**
(Run → Restart kernel), then skip straight to Cell 3.
"""
    ),
    code(
        """# Uncomment the next three lines only if Cell 1 printed a sm_XX mismatch warning.
# import subprocess, sys
# subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
#                 "torch==2.4.1", "torchvision==0.19.1", "torchaudio==2.4.1",
#                 "--index-url", "https://download.pytorch.org/whl/cu121"], check=True)
# print("Done. Now click Run -> Restart kernel, then run Cell 3 onward.")
print("Cell 2b is opt-in — read the markdown above before enabling.")
"""
    ),
    md(
        """## 2. Clone the repository

We clone into `/kaggle/working` so the files persist for the duration of the session and
are visible in the Output tab when the session ends.
"""
    ),
    code(
        """import os, shutil, subprocess

os.chdir(WORKDIR)
if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)

clone_url = f"https://github.com/{GITHUB_USER}/{REPO_NAME}.git"
subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, clone_url, REPO_DIR], check=True)
os.chdir(REPO_DIR)
print("Cloned:", os.getcwd())
print("HEAD:", subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout.strip())
"""
    ),
    md(
        """## 3. Install dependencies

Kaggle images already ship with `torch`, `transformers`, `scikit-learn`, `pandas`, `numpy`,
`pyyaml`. We just need to install the package itself in editable mode so `src.disinfo_detection`
is importable, plus `nltk` (used by the TF-IDF baseline) and `joblib`.
"""
    ),
    code(
        """import subprocess

# Install package in editable mode without re-downloading torch etc.
subprocess.run(
    ["pip", "install", "--quiet", "--no-deps", "-e", "."],
    cwd=REPO_DIR, check=True,
)
# Ensure required runtime libs are present without upgrading pre-installed GPU torch.
subprocess.run(
    ["pip", "install", "--quiet", "nltk>=3.9", "joblib>=1.4", "pyyaml>=6.0"],
    check=True,
)

# NLTK data (stopwords list used by the TF-IDF preprocessor).
import nltk
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
print("Dependencies ready.")
"""
    ),
    md(
        """## 4. Download the LIAR dataset

Pulls the UCSB zip archive and writes `data/{train,valid,test}.tsv`. Skips any file already
present in the session.
"""
    ),
    code(
        """import subprocess, sys

result = subprocess.run(
    [sys.executable, "scripts/download_data.py"],
    cwd=REPO_DIR, check=True,
)
subprocess.run(["ls", "-la", "data/"], cwd=REPO_DIR)
"""
    ),
    md(
        """## 5. Preprocess

Runs `scripts/preprocess.py` which writes pickled train/valid/test DataFrames to
`data/processed/` with the engineered `statement_clean` / `statement_transformer` / credibility
columns.
"""
    ),
    code(
        """import subprocess, sys

subprocess.run([sys.executable, "scripts/preprocess.py"], cwd=REPO_DIR, check=True)
subprocess.run(["ls", "-la", "data/processed/"], cwd=REPO_DIR)
"""
    ),
    md(
        """## 6. Train all models

Each stage is wrapped in a simple helper so we get per-stage wallclock timing and a clean
pass/fail signal even when training takes a while.
"""
    ),
    code(
        """import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPORTS_DIR = Path(REPO_DIR) / "reports"
PREDICTIONS_DIR = REPORTS_DIR / "predictions"
SEED_RESULTS: dict[int, dict] = {}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def configure_seed(seed: int) -> None:
    baseline_path = Path(REPO_DIR) / "config" / "baseline.yml"
    baseline_cfg = _load_yaml(baseline_path)
    baseline_cfg.setdefault("random_forest", {})["random_state"] = int(seed)
    _write_yaml(baseline_path, baseline_cfg)

    # The training scripts now support an in-script multi-seed loop via
    # `training.seeds`. The Kaggle notebook owns its own outer seed loop, so
    # we collapse `seeds` to a single-element list per seed so each subprocess
    # produces deterministic single-seed artifacts that snapshot_seed() can
    # copy into reports/seed_<N>/ without collisions.
    transformer_path = Path(REPO_DIR) / "config" / "transformer.yaml"
    transformer_cfg = _load_yaml(transformer_path)
    transformer_cfg.setdefault("training", {})["seed"] = int(seed)
    transformer_cfg["training"]["seeds"] = [int(seed)]
    _write_yaml(transformer_path, transformer_cfg)

    hybrid_path = Path(REPO_DIR) / "config" / "hybrid.yaml"
    hybrid_cfg = _load_yaml(hybrid_path)
    hybrid_cfg.setdefault("training", {})["seed"] = int(seed)
    hybrid_cfg["training"]["seeds"] = [int(seed)]
    _write_yaml(hybrid_path, hybrid_cfg)


def prepare_hybrid_textonly_config() -> None:
    src_cfg = Path(REPO_DIR) / "config" / "hybrid.yaml"
    tgt_cfg = Path(REPO_DIR) / "config" / "hybrid_textonly.yaml"
    payload = _load_yaml(src_cfg)
    payload.setdefault("model", {})["use_metadata"] = False
    payload.setdefault("paths", {})["output_dir"] = "models/hybrid_textonly_liar/"
    payload.setdefault("paths", {})["logs_dir"] = "reports/hybrid_textonly_logs/"
    payload.setdefault("paths", {})["best_checkpoint"] = "models/hybrid_textonly_liar/best_model.pt"
    _write_yaml(tgt_cfg, payload)


def prepare_hybrid_leaky_config() -> None:
    # Prior-art comparison: same hybrid model, but credibility counts still
    # include the row's own verdict. Used to disclose the leakage gap.
    src_cfg = Path(REPO_DIR) / "config" / "hybrid.yaml"
    tgt_cfg = Path(REPO_DIR) / "config" / "hybrid_leaky.yaml"
    payload = _load_yaml(src_cfg)
    payload.setdefault("metadata", {})["leakage_corrected"] = False
    payload.setdefault("paths", {})["output_dir"] = "models/hybrid_leaky_liar/"
    payload.setdefault("paths", {})["logs_dir"] = "reports/hybrid_leaky_logs/"
    payload.setdefault("paths", {})["best_checkpoint"] = "models/hybrid_leaky_liar/best_model.pt"
    _write_yaml(tgt_cfg, payload)


def run_stage(name: str, argv: list[str], env: dict | None = None) -> float:
    start = time.time()
    print(f"\\n\\n=== {name} — starting ===")
    result = subprocess.run(argv, cwd=REPO_DIR, env=env)
    elapsed = time.time() - start
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\\n=== {name} — {status} in {elapsed/60:.1f} min ===")
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed")
    return elapsed


def snapshot_seed(seed: int) -> dict:
    seed_dir = REPORTS_DIR / f"seed_{seed}"
    if seed_dir.exists():
        shutil.rmtree(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        "baseline_detailed_metrics.json",
        "transformer_logs/transformer_test_metrics.json",
        "transformer_logs/training_log.csv",
        "hybrid_logs/hybrid_test_metrics.json",
        "hybrid_logs/training_log.csv",
        "hybrid_textonly_logs/hybrid_test_metrics.json",
        "hybrid_textonly_logs/training_log.csv",
        "hybrid_leaky_logs/hybrid_test_metrics.json",
        "hybrid_leaky_logs/training_log.csv",
    ]
    for rel in files_to_copy:
        src = REPORTS_DIR / rel
        if src.exists():
            dst = seed_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    if PREDICTIONS_DIR.exists():
        dst_predictions = seed_dir / "predictions"
        dst_predictions.mkdir(parents=True, exist_ok=True)
        for src in sorted(PREDICTIONS_DIR.glob("*_test_predictions.jsonl")):
            shutil.copy2(src, dst_predictions / src.name)

    per_seed: dict[str, dict] = {}
    baseline_path = seed_dir / "baseline_detailed_metrics.json"
    if baseline_path.exists():
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        for model_name, splits in payload.items():
            if model_name.startswith("_") or not isinstance(splits, dict):
                continue
            test_split = splits.get("test")
            if test_split:
                per_seed[f"baseline_{model_name}"] = {
                    "test_macro_f1": test_split.get("macro_f1"),
                    "test_accuracy": test_split.get("accuracy"),
                }

    metric_paths = {
        "transformer": seed_dir / "transformer_logs" / "transformer_test_metrics.json",
        "hybrid": seed_dir / "hybrid_logs" / "hybrid_test_metrics.json",
        "hybrid_textonly": seed_dir / "hybrid_textonly_logs" / "hybrid_test_metrics.json",
        "hybrid_leaky": seed_dir / "hybrid_leaky_logs" / "hybrid_test_metrics.json",
    }
    for model_name, path in metric_paths.items():
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            per_seed[model_name] = {
                "test_macro_f1": payload.get("test_macro_f1"),
                "test_accuracy": payload.get("test_accuracy"),
            }
    return per_seed


stage_times: dict[str, float] = {}
for seed in SEEDS:
    print(f"\\n\\n######## SEED {seed} ########")
    configure_seed(seed)
    if PREDICTIONS_DIR.exists():
        shutil.rmtree(PREDICTIONS_DIR)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    if RUN_BASELINE:
        stage_times[f"seed_{seed}_baseline"] = run_stage(
            f"[seed {seed}] Baseline (TF-IDF + NB/SVM/RF)",
            [sys.executable, "scripts/train_baseline.py"],
        )
    if RUN_TRANSFORMER:
        stage_times[f"seed_{seed}_transformer"] = run_stage(
            f"[seed {seed}] Text-only RoBERTa",
            [sys.executable, "scripts/train_transformer.py"],
        )
    if RUN_HYBRID:
        stage_times[f"seed_{seed}_hybrid"] = run_stage(
            f"[seed {seed}] Hybrid (text + metadata)",
            [sys.executable, "scripts/train_hybrid.py"],
        )
    if RUN_HYBRID_TEXTONLY:
        prepare_hybrid_textonly_config()
        env = {**os.environ, "HYBRID_CONFIG": "config/hybrid_textonly.yaml"}
        stage_times[f"seed_{seed}_hybrid_textonly"] = run_stage(
            f"[seed {seed}] Hybrid text-only ablation",
            [sys.executable, "scripts/train_hybrid.py"],
            env=env,
        )
    if RUN_HYBRID_LEAKY:
        prepare_hybrid_leaky_config()
        env = {**os.environ, "HYBRID_CONFIG": "config/hybrid_leaky.yaml"}
        stage_times[f"seed_{seed}_hybrid_leaky"] = run_stage(
            f"[seed {seed}] Hybrid leaky (prior-art credibility)",
            [sys.executable, "scripts/train_hybrid.py"],
            env=env,
        )

    SEED_RESULTS[seed] = snapshot_seed(seed)

print("\\n\\nSTAGE TIMES (minutes):")
for stage, seconds in stage_times.items():
    print(f"  {stage:32s} {seconds/60:6.1f}")
print(f"  {'TOTAL':32s} {sum(stage_times.values())/60:6.1f}")
"""
    ),
    md(
        """## 7. Collect the test-split metrics

Every training script writes a `*_test_metrics.json` file with the best-val macro-F1, test
macro-F1, test accuracy, and per-class F1. We roll them up into a single `results_summary.json`
that you can paste straight into the paper's results table.
"""
    ),
    code(
        """import json
from pathlib import Path

import numpy as np

REPORTS_DIR = Path(REPO_DIR) / "reports"


def _read_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"_note": f"not found at {path}"}


primary_dir = REPORTS_DIR / f"seed_{PRIMARY_SEED}"
SUMMARY_FILES = {
    "baseline": primary_dir / "baseline_detailed_metrics.json",
    "transformer": primary_dir / "transformer_logs" / "transformer_test_metrics.json",
    "hybrid": primary_dir / "hybrid_logs" / "hybrid_test_metrics.json",
    "hybrid_textonly": primary_dir / "hybrid_textonly_logs" / "hybrid_test_metrics.json",
    "hybrid_leaky": primary_dir / "hybrid_leaky_logs" / "hybrid_test_metrics.json",
}

summary = {
    "_primary_seed": PRIMARY_SEED,
    "_note": (
        "Single-seed payloads for quick inspection; use multi_seed_summary.json "
        "for reported thesis/paper means."
    ),
}
for name, path in SUMMARY_FILES.items():
    summary[name] = _read_json(path)

summary_path = REPORTS_DIR / "results_summary.json"
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print(f"\\nSaved: {summary_path}")

# Build multi-seed macro-F1/accuracy means from the per-seed snapshots.
per_seed_metrics: dict[str, dict] = {}
for seed in SEEDS:
    seed_dir = REPORTS_DIR / f"seed_{seed}"
    per_model: dict[str, dict] = {}

    baseline_path = seed_dir / "baseline_detailed_metrics.json"
    if baseline_path.exists():
        payload = _read_json(baseline_path)
        for model_name, splits in payload.items():
            if model_name.startswith("_") or not isinstance(splits, dict):
                continue
            test_split = splits.get("test")
            if test_split:
                per_model[f"baseline_{model_name}"] = {
                    "test_macro_f1": test_split.get("macro_f1"),
                    "test_accuracy": test_split.get("accuracy"),
                }

    deep_paths = {
        "transformer": seed_dir / "transformer_logs" / "transformer_test_metrics.json",
        "hybrid": seed_dir / "hybrid_logs" / "hybrid_test_metrics.json",
        "hybrid_textonly": seed_dir / "hybrid_textonly_logs" / "hybrid_test_metrics.json",
        "hybrid_leaky": seed_dir / "hybrid_leaky_logs" / "hybrid_test_metrics.json",
    }
    for model_name, path in deep_paths.items():
        if path.exists():
            payload = _read_json(path)
            per_model[model_name] = {
                "test_macro_f1": payload.get("test_macro_f1"),
                "test_accuracy": payload.get("test_accuracy"),
            }

    per_seed_metrics[str(seed)] = per_model

model_names = sorted({name for metrics in per_seed_metrics.values() for name in metrics})
rows = []
for model_name in model_names:
    macro_values = [
        metrics[model_name]["test_macro_f1"]
        for metrics in per_seed_metrics.values()
        if model_name in metrics and metrics[model_name]["test_macro_f1"] is not None
    ]
    accuracy_values = [
        metrics[model_name]["test_accuracy"]
        for metrics in per_seed_metrics.values()
        if model_name in metrics and metrics[model_name]["test_accuracy"] is not None
    ]
    rows.append(
        {
            "model": model_name,
            "n_seeds": len(macro_values),
            "test_macro_f1_mean": float(np.mean(macro_values)) if macro_values else None,
            "test_macro_f1_std": (
                float(np.std(macro_values, ddof=1)) if len(macro_values) > 1 else 0.0
            ),
            "test_accuracy_mean": float(np.mean(accuracy_values)) if accuracy_values else None,
            "test_accuracy_std": (
                float(np.std(accuracy_values, ddof=1)) if len(accuracy_values) > 1 else 0.0
            ),
        }
    )

multi_seed_payload = {
    "seeds": SEEDS,
    "primary_seed": PRIMARY_SEED,
    "summary": rows,
    "per_seed_metrics": per_seed_metrics,
}
multi_seed_path = REPORTS_DIR / "multi_seed_summary.json"
multi_seed_path.write_text(json.dumps(multi_seed_payload, indent=2), encoding="utf-8")
print(f"Saved: {multi_seed_path}")
print(json.dumps(multi_seed_payload, indent=2))
"""
    ),
    md(
        """## 8. Generate only the paper artifacts

Run the post-training scripts that the paper actually uses. This replaces the old broad
diagnostic gallery and writes only:

- leakage verification JSON reports;
- bootstrap confidence-interval JSON/table reports;
- the three final paper figure pairs under `figures/`.

No `reports/figures_all/` gallery is generated.
"""
    ),
    code(
        """import os
import subprocess
import sys
from pathlib import Path

postprocess_env = {**os.environ, "PYTHONPATH": REPO_DIR}
postprocess_steps = [
    ("Leakage verification", [sys.executable, "scripts/verify_leakage.py"]),
    ("Bootstrap confidence intervals", [sys.executable, "scripts/bootstrap_ci.py"]),
    ("Per-class F1 figure", [sys.executable, "scripts/plot_per_class_f1.py"]),
    (
        "Confusion matrices figure",
        [sys.executable, "scripts/plot_confusion_matrices.py"],
    ),
    (
        "Training curves figure",
        [
            sys.executable,
            "scripts/plot_training_curves.py",
            "--seed",
            str(PRIMARY_SEED),
        ],
    ),
]

for name, argv in postprocess_steps:
    print(f"\n=== {name} ===")
    subprocess.run(argv, cwd=REPO_DIR, env=postprocess_env, check=True)

figure_dir = Path(REPO_DIR) / "figures"
expected_figures = [
    "per_class_f1_grouped.pdf",
    "per_class_f1_grouped.png",
    "confusion_matrices.pdf",
    "confusion_matrices.png",
    "training_curves.pdf",
    "training_curves.png",
]
print("\nFinal paper figures:")
for name in expected_figures:
    path = figure_dir / name
    if path.exists():
        print(f"  {path.relative_to(REPO_DIR)}")
    else:
        print(f"  MISSING: {name}")
"""
    ),
    md(
        """## 9. Create downloadable archives

This writes the paper/reproducibility archive to the Kaggle **Output** panel:
`/kaggle/working/disinformation_results.zip`.

If `CREATE_MODELS_ZIP = True`, it also writes checkpoints separately to:
`/kaggle/working/models.zip`.

It contains only the artifacts needed for the paper: aggregate metric JSONs,
seed-scoped metric JSONs, TEST prediction JSONL files, bootstrap/leakage reports,
the final figure files, and primary-seed training logs for Fig. 3. Model checkpoints
and broad diagnostic figure galleries are intentionally excluded.
"""
    ),
    code(
        """import subprocess
import sys

archive_cmd = [
    sys.executable,
    "scripts/package_artifacts.py",
    "--output",
    "/kaggle/working/disinformation_results.zip",
    "--seeds",
    *[str(seed) for seed in SEEDS],
    "--primary-seed",
    str(PRIMARY_SEED),
]
if CREATE_MODELS_ZIP:
    archive_cmd.extend([
        "--include-models",
        "--models-output",
        "/kaggle/working/models.zip",
    ])

subprocess.run(archive_cmd, cwd=REPO_DIR, check=True)
print("Created /kaggle/working/disinformation_results.zip")
if CREATE_MODELS_ZIP:
    print("Created /kaggle/working/models.zip if models/ exists")
"""
    ),
    md(
        """## Done

- Download `/kaggle/working/disinformation_results.zip` from the Kaggle Output panel.
- If enabled, download `/kaggle/working/models.zip` separately from the Kaggle Output panel.
- The archive contains aggregate metrics, seed-scoped metrics, TEST prediction JSONL files,
  bootstrap/leakage reports, final paper figures, and primary-seed training logs for Fig. 3.
- Model checkpoints remain in the Kaggle session under `models/`; they are intentionally not
  part of the required results archive.
"""
    ),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, indent=1))
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes, {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
