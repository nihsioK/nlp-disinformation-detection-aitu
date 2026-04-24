"""Build notebooks/kaggle_training.ipynb from a structured cell list.

Run from the repo root:

    python notebooks/_build_kaggle_notebook.py

This keeps the notebook JSON auto-generated so reviewers can see the cell
contents as plain Python strings during code review.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "kaggle_training.ipynb"


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
        """# LIAR disinformation detection — Kaggle training notebook

This notebook runs the full training pipeline (baseline → transformer → hybrid → hybrid-textonly)
for the thesis project `nihsioK/nlp-disinformation-detection-aitu` on a Kaggle GPU instance
(T4 × 2 recommended). Total expected runtime: **~60–90 min** for all four models.

**Before running:**

1. In Kaggle, create a new Notebook → Settings → Accelerator → **`GPU T4 x2`** (recommended).
   **Do NOT pick `GPU P100`** — Kaggle's current PyTorch build does not ship CUDA kernels for
   P100's compute capability (sm_60), and transformer training will crash with
   `cudaErrorNoKernelImageForDevice`. T4 (sm_75) works out of the box. If you are forced onto
   P100, run the optional **Cell 2b** below to reinstall a compatible PyTorch build (adds
   ~3–5 minutes).
2. Settings → Internet → **On** (needed to clone GitHub + download `roberta-base`).
3. Add a Kaggle Secret named `GH_PAT` (Add-ons → Secrets → Add Secret) containing a GitHub
   Personal Access Token with `repo` scope — needed to push the JSON metrics back to the
   repo. Skip this secret if you only want to view metrics in the notebook output.
4. Run all cells.

**What this notebook does:**

- Clones the repo, installs the `ml` and `dev` extras.
- Downloads the LIAR dataset.
- Runs preprocessing and all four training scripts sequentially.
- Collects `reports/*_logs/*_test_metrics.json` into a single summary.
- Commits and pushes the JSON metric files back to the `main` branch so the numbers land
  directly in git for the paper draft. Model weights (.pt, ~500 MB each) stay on Kaggle —
  download them manually from the session output if you want local copies.
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

GIT_AUTHOR_NAME = "Daniyar Koishin"
GIT_AUTHOR_EMAIL = "dandevko@gmail.com"

# Commit the JSON test-metric files back to GitHub after training.
# Requires a Kaggle Secret named GH_PAT with repo scope.
PUSH_RESULTS_TO_GITHUB = True

# Which stages to run. Flip to False to skip individual stages during debugging.
RUN_BASELINE = True
RUN_TRANSFORMER = True
RUN_HYBRID = True
RUN_HYBRID_TEXTONLY = True  # RQ2 ablation

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
        """import subprocess, sys, time
from pathlib import Path

def run_stage(name: str, argv: list[str]) -> float:
    start = time.time()
    print(f"\\n\\n=== {name} — starting ===")
    result = subprocess.run(argv, cwd=REPO_DIR)
    elapsed = time.time() - start
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\\n=== {name} — {status} in {elapsed/60:.1f} min ===")
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed")
    return elapsed

stage_times = {}
if RUN_BASELINE:
    stage_times["baseline"] = run_stage("Baseline (TF-IDF + NB/SVM/RF)", [sys.executable, "scripts/train_baseline.py"])
if RUN_TRANSFORMER:
    stage_times["transformer"] = run_stage("Text-only RoBERTa", [sys.executable, "scripts/train_transformer.py"])
if RUN_HYBRID:
    stage_times["hybrid"] = run_stage("Hybrid (text + metadata)", [sys.executable, "scripts/train_hybrid.py"])
if RUN_HYBRID_TEXTONLY:
    # Reuse the hybrid pipeline but with use_metadata=False to produce the RQ2 ablation.
    import shutil
    src_cfg = Path(REPO_DIR) / "config" / "hybrid.yaml"
    tgt_cfg = Path(REPO_DIR) / "config" / "hybrid_textonly.yaml"
    text = src_cfg.read_text().replace("use_metadata: true", "use_metadata: false")
    tgt_cfg.write_text(text)
    # Same paths would overwrite — redirect output_dir / logs_dir for the ablation.
    text2 = tgt_cfg.read_text()
    text2 = text2.replace("models/hybrid_liar/", "models/hybrid_textonly_liar/")
    text2 = text2.replace("reports/hybrid_logs/", "reports/hybrid_textonly_logs/")
    tgt_cfg.write_text(text2)
    env = {**os.environ, "HYBRID_CONFIG": "config/hybrid_textonly.yaml"}
    start = time.time()
    print("\\n\\n=== Hybrid text-only ablation — starting ===")
    result = subprocess.run([sys.executable, "scripts/train_hybrid.py"], cwd=REPO_DIR, env=env)
    elapsed = time.time() - start
    if result.returncode != 0:
        raise RuntimeError("Hybrid text-only ablation failed")
    stage_times["hybrid_textonly"] = elapsed
    print(f"\\n=== Hybrid text-only — OK in {elapsed/60:.1f} min ===")

print("\\n\\nSTAGE TIMES (minutes):")
for stage, seconds in stage_times.items():
    print(f"  {stage:22s} {seconds/60:6.1f}")
print(f"  {'TOTAL':22s} {sum(stage_times.values())/60:6.1f}")
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

METRIC_FILES = {
    "baseline":        "reports/baseline_detailed_metrics.json",
    "transformer":     "reports/transformer_logs/transformer_test_metrics.json",
    "hybrid":          "reports/hybrid_logs/hybrid_test_metrics.json",
    "hybrid_textonly": "reports/hybrid_textonly_logs/hybrid_test_metrics.json",
}

summary = {}
for name, rel in METRIC_FILES.items():
    path = Path(REPO_DIR) / rel
    if path.exists():
        summary[name] = json.loads(path.read_text())
    else:
        summary[name] = {"_note": f"not found at {rel}"}

summary_path = Path(REPO_DIR) / "reports" / "results_summary.json"
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print(f"\\nSaved: {summary_path}")
"""
    ),
    md(
        """## 8. Push JSON metrics back to GitHub

Optional but strongly recommended: commits the results files to the `main` branch so the
numbers are captured in git and immediately available for the paper.

Requires a Kaggle Secret called `GH_PAT` (Add-ons → Secrets) containing a GitHub Personal
Access Token with `repo` scope. We never print the token.
"""
    ),
    code(
        """import subprocess, os

if not PUSH_RESULTS_TO_GITHUB:
    print("PUSH_RESULTS_TO_GITHUB=False — skipping.")
else:
    try:
        from kaggle_secrets import UserSecretsClient
        GH_PAT = UserSecretsClient().get_secret("GH_PAT")
    except Exception as exc:
        raise RuntimeError(
            "Could not load Kaggle Secret 'GH_PAT'. Add it via Add-ons → Secrets, "
            "or set PUSH_RESULTS_TO_GITHUB = False in the config cell."
        ) from exc

    # Configure git identity and the authenticated remote.
    subprocess.run(["git", "config", "user.name", GIT_AUTHOR_NAME], cwd=REPO_DIR, check=True)
    subprocess.run(["git", "config", "user.email", GIT_AUTHOR_EMAIL], cwd=REPO_DIR, check=True)

    auth_remote = f"https://{GITHUB_USER}:{GH_PAT}@github.com/{GITHUB_USER}/{REPO_NAME}.git"
    subprocess.run(["git", "remote", "set-url", "origin", auth_remote], cwd=REPO_DIR, check=True)

    # Stage only the whitelisted JSON metric files.
    files = [
        "reports/results_summary.json",
        "reports/baseline_detailed_metrics.json",
        "reports/transformer_logs/transformer_test_metrics.json",
        "reports/hybrid_logs/hybrid_test_metrics.json",
        "reports/hybrid_textonly_logs/hybrid_test_metrics.json",
    ]
    for f in files:
        p = os.path.join(REPO_DIR, f)
        if os.path.exists(p):
            subprocess.run(["git", "add", "-f", f], cwd=REPO_DIR, check=True)

    diff = subprocess.run(["git", "diff", "--cached", "--stat"], cwd=REPO_DIR, capture_output=True, text=True)
    if not diff.stdout.strip():
        print("No metric file changes to commit.")
    else:
        print(diff.stdout)
        commit_msg = "chore(results): add Kaggle training run metrics"
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=REPO_DIR, check=True)
        subprocess.run(["git", "push", "origin", f"HEAD:{BRANCH}"], cwd=REPO_DIR, check=True)
        print("Pushed to", BRANCH)

    # Scrub the token from the remote URL so it can't leak via the saved notebook state.
    subprocess.run(
        ["git", "remote", "set-url", "origin", f"https://github.com/{GITHUB_USER}/{REPO_NAME}.git"],
        cwd=REPO_DIR,
    )
"""
    ),
    md(
        """## 9. Generate dissertation figures

Produce a diverse set of publication-ready plots from the artefacts written above, without
re-running training. For every deep model (transformer, hybrid, hybrid_textonly) we emit:

- **Loss curve (train + val)** — replaces the train-only plot; both curves on one axis.
- **Training loss only** and **validation loss only** — kept as standalone variants for the
  appendix.
- **Validation macro-F1 by epoch**.
- **Confusion matrix** (raw counts + row-normalised).
- **Per-class F1 / precision / recall bar charts** on the test split.

For the TF-IDF baselines we emit per-class bars and confusion matrices per classifier.

Finally, cross-model summary plots: macro-F1 comparison, accuracy comparison, combined loss
curves, combined validation macro-F1 curves.

All figures land in `reports/figures_all/` as 300-DPI PNG **and** vector PDF (so you can drop
either into LaTeX).
"""
    ),
    code(
        """import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_ROOT = Path(REPO_DIR) / "reports" / "figures_all"
FIG_ROOT.mkdir(parents=True, exist_ok=True)

DL_MODELS = {
    \"transformer\":     {
        \"log\":     \"reports/transformer_logs/training_log.csv\",
        \"metrics\": \"reports/transformer_logs/transformer_test_metrics.json\",
        \"pretty\":  \"RoBERTa (text-only)\",
    },
    \"hybrid\": {
        \"log\":     \"reports/hybrid_logs/training_log.csv\",
        \"metrics\": \"reports/hybrid_logs/hybrid_test_metrics.json\",
        \"pretty\":  \"Hybrid (text + metadata)\",
    },
    \"hybrid_textonly\": {
        \"log\":     \"reports/hybrid_textonly_logs/training_log.csv\",
        \"metrics\": \"reports/hybrid_textonly_logs/hybrid_test_metrics.json\",
        \"pretty\":  \"Hybrid (text-only ablation)\",
    },
}

plt.rcParams.update({\"figure.dpi\": 150, \"savefig.dpi\": 300, \"font.size\": 11})


def _save(fig, stem: str) -> None:
    png = FIG_ROOT / f\"{stem}.png\"
    pdf = FIG_ROOT / f\"{stem}.pdf\"
    fig.tight_layout()
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    print(f\"  wrote {png.relative_to(REPO_DIR)} (+ .pdf)\")


def _line(ax, x, y, label, marker=\"o\"):
    ax.plot(x, y, marker=marker, linewidth=1.8, label=label)


# ---- per-DL-model training diagnostics ----
for key, cfg in DL_MODELS.items():
    log_path = Path(REPO_DIR) / cfg[\"log\"]
    if not log_path.exists():
        print(f\"skip {key}: no training log at {cfg['log']}\")
        continue
    df = pd.read_csv(log_path)
    pretty = cfg[\"pretty\"]

    # Combined loss curve (train + val) — the main loss graph for the thesis.
    fig, ax = plt.subplots(figsize=(7, 4.2))
    _line(ax, df[\"epoch\"], df[\"train_loss\"], \"Train loss\")
    _line(ax, df[\"epoch\"], df[\"val_loss\"],   \"Validation loss\", marker=\"s\")
    ax.set_xlabel(\"Epoch\"); ax.set_ylabel(\"Cross-entropy loss\")
    ax.set_title(f\"{pretty} — training vs validation loss\")
    ax.grid(alpha=0.3); ax.legend()
    _save(fig, f\"{key}_loss_curve\")

    # Training loss only (kept for completeness).
    fig, ax = plt.subplots(figsize=(7, 4.2))
    _line(ax, df[\"epoch\"], df[\"train_loss\"], \"Train loss\")
    ax.set_xlabel(\"Epoch\"); ax.set_ylabel(\"Training loss\")
    ax.set_title(f\"{pretty} — training loss\"); ax.grid(alpha=0.3); ax.legend()
    _save(fig, f\"{key}_train_loss_only\")

    # Validation loss only.
    fig, ax = plt.subplots(figsize=(7, 4.2))
    _line(ax, df[\"epoch\"], df[\"val_loss\"], \"Validation loss\", marker=\"s\")
    ax.set_xlabel(\"Epoch\"); ax.set_ylabel(\"Validation loss\")
    ax.set_title(f\"{pretty} — validation loss\"); ax.grid(alpha=0.3); ax.legend()
    _save(fig, f\"{key}_val_loss_only\")

    # Validation macro-F1 by epoch.
    if \"val_macro_f1\" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        _line(ax, df[\"epoch\"], df[\"val_macro_f1\"], \"Validation macro-F1\", marker=\"^\")
        ax.set_xlabel(\"Epoch\"); ax.set_ylabel(\"Macro-F1\")
        ax.set_title(f\"{pretty} — validation macro-F1 by epoch\")
        ax.grid(alpha=0.3); ax.legend()
        _save(fig, f\"{key}_val_macro_f1\")


def _plot_confusion(matrix, labels, title, stem, normalize=False):
    m = np.array(matrix, dtype=float)
    if normalize:
        row_sums = m.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        m = m / row_sums
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(m, cmap=\"Blues\")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha=\"right\")
    ax.set_yticklabels(labels)
    ax.set_xlabel(\"Predicted\"); ax.set_ylabel(\"True\"); ax.set_title(title)
    fmt = \"{:.2f}\" if normalize else \"{:.0f}\"
    thresh = m.max() / 2.0 if m.size else 0.0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, fmt.format(m[i, j]), ha=\"center\", va=\"center\",
                    color=\"white\" if m[i, j] > thresh else \"black\", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, stem)


def _plot_per_class(values: dict, title: str, ylabel: str, stem: str):
    labels = list(values.keys())
    vals = [values[k] for k in labels]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, vals, color=\"#4C72B0\")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f\"{v:.2f}\",
                ha=\"center\", va=\"bottom\", fontsize=9)
    ax.set_ylim(0, max(max(vals) + 0.1, 0.1))
    ax.set_ylabel(ylabel); ax.set_title(title)
    plt.xticks(rotation=30, ha=\"right\"); ax.grid(alpha=0.3, axis=\"y\")
    _save(fig, stem)


# ---- per-DL-model test-split diagnostics ----
for key, cfg in DL_MODELS.items():
    mj = Path(REPO_DIR) / cfg[\"metrics\"]
    if not mj.exists():
        print(f\"skip {key}: no metrics JSON at {cfg['metrics']}\")
        continue
    payload = json.loads(mj.read_text())
    labels = payload[\"test_confusion_matrix_labels\"]
    cm = payload[\"test_confusion_matrix\"]
    _plot_confusion(cm, labels, f\"{cfg['pretty']} — confusion matrix (test)\",
                    f\"{key}_confusion_matrix\", normalize=False)
    _plot_confusion(cm, labels, f\"{cfg['pretty']} — normalised confusion matrix (test)\",
                    f\"{key}_confusion_matrix_normalised\", normalize=True)
    _plot_per_class(payload[\"test_per_class_f1\"],
                    f\"{cfg['pretty']} — per-class F1\", \"F1\", f\"{key}_per_class_f1\")
    _plot_per_class(payload[\"test_per_class_precision\"],
                    f\"{cfg['pretty']} — per-class precision\", \"Precision\", f\"{key}_per_class_precision\")
    _plot_per_class(payload[\"test_per_class_recall\"],
                    f\"{cfg['pretty']} — per-class recall\", \"Recall\", f\"{key}_per_class_recall\")


# ---- baselines (TF-IDF NB / SVM / RF) ----
baseline_path = Path(REPO_DIR) / \"reports\" / \"baseline_detailed_metrics.json\"
if baseline_path.exists():
    baseline = json.loads(baseline_path.read_text())
    for model_name, splits in baseline.items():
        test_split = splits.get(\"test\") if isinstance(splits, dict) else None
        if not test_split:
            continue
        labels = test_split[\"confusion_matrix_labels\"]
        _plot_confusion(test_split[\"confusion_matrix\"], labels,
                        f\"Baseline {model_name} — confusion matrix (test)\",
                        f\"baseline_{model_name}_confusion_matrix\", normalize=False)
        _plot_confusion(test_split[\"confusion_matrix\"], labels,
                        f\"Baseline {model_name} — normalised confusion matrix (test)\",
                        f\"baseline_{model_name}_confusion_matrix_normalised\", normalize=True)
        _plot_per_class(test_split[\"per_class_f1\"],
                        f\"Baseline {model_name} — per-class F1\", \"F1\",
                        f\"baseline_{model_name}_per_class_f1\")
        _plot_per_class(test_split[\"per_class_precision\"],
                        f\"Baseline {model_name} — per-class precision\", \"Precision\",
                        f\"baseline_{model_name}_per_class_precision\")
        _plot_per_class(test_split[\"per_class_recall\"],
                        f\"Baseline {model_name} — per-class recall\", \"Recall\",
                        f\"baseline_{model_name}_per_class_recall\")


# ---- cross-model comparison (all models on the test split) ----
rows = []
if baseline_path.exists():
    for model_name, splits in json.loads(baseline_path.read_text()).items():
        test_split = splits.get(\"test\") if isinstance(splits, dict) else None
        if test_split:
            rows.append((f\"baseline_{model_name}\", test_split[\"accuracy\"], test_split[\"macro_f1\"]))
for key, cfg in DL_MODELS.items():
    mj = Path(REPO_DIR) / cfg[\"metrics\"]
    if mj.exists():
        p = json.loads(mj.read_text())
        rows.append((key, p[\"test_accuracy\"], p[\"test_macro_f1\"]))

if rows:
    names   = [r[0] for r in rows]
    accs    = [r[1] for r in rows]
    f1s     = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(names, f1s, color=\"#55A868\")
    for bar, v in zip(bars, f1s):
        ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2, f\"{v:.3f}\",
                va=\"center\", fontsize=9)
    ax.set_xlim(0, max(f1s) + 0.05)
    ax.set_xlabel(\"Macro-F1 (test)\"); ax.set_title(\"Model comparison — test macro-F1\")
    ax.grid(alpha=0.3, axis=\"x\")
    _save(fig, \"model_comparison_macro_f1\")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(names, accs, color=\"#C44E52\")
    for bar, v in zip(bars, accs):
        ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2, f\"{v:.3f}\",
                va=\"center\", fontsize=9)
    ax.set_xlim(0, max(accs) + 0.05)
    ax.set_xlabel(\"Accuracy (test)\"); ax.set_title(\"Model comparison — test accuracy\")
    ax.grid(alpha=0.3, axis=\"x\")
    _save(fig, \"model_comparison_accuracy\")

    # Grouped bar: accuracy vs macro-F1 per model.
    idx = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(idx - w / 2, accs, w, label=\"Accuracy\",  color=\"#4C72B0\")
    ax.bar(idx + w / 2, f1s,  w, label=\"Macro-F1\", color=\"#DD8452\")
    ax.set_xticks(idx); ax.set_xticklabels(names, rotation=30, ha=\"right\")
    ax.set_ylabel(\"Score\"); ax.set_title(\"Model comparison — accuracy vs macro-F1 (test)\")
    ax.grid(alpha=0.3, axis=\"y\"); ax.legend()
    _save(fig, \"model_comparison_accuracy_vs_f1\")


# ---- combined DL training curves (all on one axis) ----
combined_loss, combined_val, combined_f1 = [], [], []
for key, cfg in DL_MODELS.items():
    log_path = Path(REPO_DIR) / cfg[\"log\"]
    if log_path.exists():
        df = pd.read_csv(log_path)
        combined_loss.append((cfg[\"pretty\"], df))
        combined_val.append((cfg[\"pretty\"], df))
        combined_f1.append((cfg[\"pretty\"], df))

if combined_loss:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for pretty, df in combined_loss:
        ax.plot(df[\"epoch\"], df[\"train_loss\"], marker=\"o\", label=f\"{pretty} — train\")
        ax.plot(df[\"epoch\"], df[\"val_loss\"],   marker=\"s\", linestyle=\"--\",
                label=f\"{pretty} — val\")
    ax.set_xlabel(\"Epoch\"); ax.set_ylabel(\"Cross-entropy loss\")
    ax.set_title(\"All deep models — training vs validation loss\")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc=\"best\")
    _save(fig, \"combined_loss_curves\")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for pretty, df in combined_val:
        if \"val_loss\" in df.columns:
            ax.plot(df[\"epoch\"], df[\"val_loss\"], marker=\"s\", label=pretty)
    ax.set_xlabel(\"Epoch\"); ax.set_ylabel(\"Validation loss\")
    ax.set_title(\"All deep models — validation loss\")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    _save(fig, \"combined_val_loss\")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for pretty, df in combined_f1:
        if \"val_macro_f1\" in df.columns:
            ax.plot(df[\"epoch\"], df[\"val_macro_f1\"], marker=\"^\", label=pretty)
    ax.set_xlabel(\"Epoch\"); ax.set_ylabel(\"Validation macro-F1\")
    ax.set_title(\"All deep models — validation macro-F1 by epoch\")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    _save(fig, \"combined_val_macro_f1\")

print(f\"\\nAll figures saved under {FIG_ROOT.relative_to(REPO_DIR)}/\")
print(f\"Total files: {sum(1 for _ in FIG_ROOT.iterdir())}\")
"""
    ),
    md(
        """## 10. Zip model checkpoints and figures for download

Model weights (`models/*.pt`, ~500 MB each) and figure files are too large or too numerous to
push to git. We zip both into `/kaggle/working/` so they show up as single downloadable
artefacts in the right-hand **Output** panel.
"""
    ),
    code(
        """import subprocess
from pathlib import Path

subprocess.run(
    [\"zip\", \"-r\", \"/kaggle/working/models.zip\", \"models/\"],
    cwd=REPO_DIR, check=True,
)
print(\"Zipped checkpoints to /kaggle/working/models.zip\")

figures_rel = \"reports/figures_all\"
if (Path(REPO_DIR) / figures_rel).exists():
    subprocess.run(
        [\"zip\", \"-r\", \"/kaggle/working/figures.zip\", figures_rel],
        cwd=REPO_DIR, check=True,
    )
    print(\"Zipped figures     to /kaggle/working/figures.zip\")
else:
    print(\"No figures directory found — skipped figures.zip\")

# Also zip the per-model training logs + JSON metrics so the raw numbers travel with the plots.
subprocess.run(
    [\"zip\", \"-r\", \"/kaggle/working/reports.zip\",
     \"reports/transformer_logs\", \"reports/hybrid_logs\",
     \"reports/hybrid_textonly_logs\", \"reports/baseline_detailed_metrics.json\",
     \"reports/results_summary.json\", \"reports/figures_all\"],
    cwd=REPO_DIR,
)
print(\"Zipped reports     to /kaggle/working/reports.zip — download from the Output tab.\")
"""
    ),
    md(
        """## Done

- Training logs (per-epoch CSV, full console output) stayed on the Kaggle session.
- Test-split JSON metrics were pushed to GitHub (`main` branch) — inspect them in the repo's
  `reports/` folder and paste them into the paper's results table.
- Dissertation figures are under `reports/figures_all/` and bundled in `figures.zip`.
- Model weights are available in the Output tab via `models.zip`.
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
