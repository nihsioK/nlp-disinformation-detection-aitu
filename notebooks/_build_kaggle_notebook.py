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
        """## 9. (Optional) Download model checkpoints

Model weights (`models/*.pt`, ~500 MB each) are too large for git. After this cell runs,
open the **Output** tab on the right-hand side of Kaggle — all files under
`/kaggle/working/nlp-disinformation-detection-aitu/models/` are listed and downloadable.

Or zip them into a single archive for easier download:
"""
    ),
    code(
        """import subprocess
subprocess.run(
    ["zip", "-r", "/kaggle/working/models.zip", "models/"],
    cwd=REPO_DIR, check=True,
)
print("Zipped checkpoints to /kaggle/working/models.zip — download from the Output tab.")
"""
    ),
    md(
        """## Done

- Training logs (per-epoch CSV, full console output) stayed on the Kaggle session.
- Test-split JSON metrics were pushed to GitHub (`main` branch) — you can inspect them in the
  repo's `reports/` folder and paste them directly into the paper's results table.
- Model weights are available in the Output tab for manual download.
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
