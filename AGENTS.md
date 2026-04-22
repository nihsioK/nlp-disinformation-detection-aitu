# AGENTS.md — AI Agent Context & Rules
> This file is the single source of truth for any AI agent (GPT Codex, Claude, etc.) working on this project.
> **Read this entire file before writing a single line of code.**
> Keep this file updated after every completed task — add to the progress log at the bottom.

---

## 1. Project Overview

**Project:** NLP framework for detecting disinformation in social networks
**Author:** Daniyar Koishin — Master's student, Astana IT University (AITU), Kazakhstan
**Goal:** Build a hybrid classifier that combines transformer-based text encoding (RoBERTa) with contextual speaker metadata to classify political statements on a 6-class truthfulness scale.
**Dataset:** LIAR (12.8K labeled political statements with speaker metadata and historical credibility)
**Main contribution:** A parallel dual-branch architecture where Branch 1 encodes text semantics via fine-tuned RoBERTa and Branch 2 encodes speaker credibility history + metadata via an MLP. Both branches are fused for final multi-class classification.
**Evaluation metric:** Macro-averaged F1-score (primary), Accuracy + Confusion Matrix (secondary)

---

## 2. Repository Structure

```
nlp-disinformation-detection-aitu/
├── AGENTS.md                          ← YOU ARE HERE
├── PROJECT_PLAN.md                    ← Full week-by-week task list
├── README.md                          ← Human-readable project intro
├── Makefile                           ← CLI shortcuts (make train-baseline, etc.)
├── pyproject.toml                     ← Dependencies and package metadata
├── .gitignore                         ← data/, models/, __pycache__, .env
│
├── config/
│   ├── dataset.yaml                   ← Paths, column names, label maps
│   ├── baseline.yml                   ← TF-IDF + classical ML hyperparameters
│   └── transformer.yaml               ← RoBERTa training config
│
├── data/                              ← NEVER commit to git (in .gitignore)
│   ├── train.tsv                      ← LIAR raw training split
│   ├── valid.tsv                      ← LIAR raw validation split
│   ├── test.tsv                       ← LIAR raw test split
│   └── processed/                     ← Preprocessed outputs from scripts/preprocess.py
│
├── models/                            ← NEVER commit to git (in .gitignore)
│   ├── baseline_svm.pkl
│   ├── baseline_nb.pkl
│   ├── baseline_rf.pkl
│   ├── roberta_liar/                  ← HuggingFace model checkpoint
│   └── hybrid_model/                  ← Final hybrid model checkpoint
│
├── notebooks/
│   ├── 01_eda_liar.ipynb              ← ✅ COMPLETE
│   ├── 02_preprocessing.ipynb         ← Visualize preprocessing steps
│   ├── 03_baseline_training.ipynb     ← Walk through classical ML training
│   ├── 04_transformer_experiments.ipynb
│   ├── 05_hybrid_model.ipynb
│   └── 06_evaluation_results.ipynb    ← All final results + figures
│
├── reports/
│   └── figures/                       ← All paper figures saved as PNG
│
├── scripts/                           ← Standalone runnable scripts
│   ├── download_data.py               ← Downloads LIAR to data/
│   ├── preprocess.py                  ← Runs full preprocessing pipeline
│   ├── train_baseline.py              ← Trains all classical ML models
│   └── train_transformer.py           ← Fine-tunes RoBERTa
│
├── src/disinfo_detection/             ← Core importable Python package
│   ├── __init__.py
│   ├── data_loader.py                 ← load_liar(split) and related utilities
│   ├── preprocessing.py               ← Text cleaning, feature engineering
│   ├── models_baseline.py             ← TFIDFBaseline class
│   ├── models_transformers.py         ← RoBERTaClassifier class + LIARDataset
│   ├── models_hybrid.py               ← HybridDisinfoClassifier (MAIN CONTRIBUTION)
│   └── evaluation.py                  ← compute_metrics(), plot_confusion_matrix()
│
└── tests/
    └── __init__.py
```

---

## 3. Coding Conventions

### 3.1 Language & Version
- Python **3.12+** (see `.python-version`)
- All code must be compatible with the versions in `pyproject.toml`

### 3.2 Style Rules
- Follow **PEP 8** strictly
- Max line length: **100 characters**
- Use **type hints** on all function signatures
- Use **f-strings** for string formatting (not `.format()` or `%`)
- No wildcard imports (`from module import *` is forbidden)

```python
# ✅ CORRECT
def load_liar(split: str, config_path: str = "config/dataset.yaml") -> pd.DataFrame:

# ❌ WRONG
def load_liar(split, config_path="config/dataset.yaml"):
```

### 3.3 Docstrings
Every function and class **must** have a Google-style docstring:

```python
def compute_metrics(y_true: list, y_pred: list) -> dict:
    """Compute classification metrics for multi-class prediction.

    Args:
        y_true: Ground truth integer labels.
        y_pred: Predicted integer labels.

    Returns:
        Dictionary with keys: accuracy, macro_f1, per_class_f1.

    Example:
        >>> metrics = compute_metrics([0, 1, 2], [0, 2, 2])
        >>> print(metrics["macro_f1"])
    """
```

### 3.4 Config-Driven Code
- **Never hardcode paths, hyperparameters, or label names** in source files
- All configurable values live in `config/*.yaml`
- Load configs using `pyyaml` at the top of scripts

```python
import yaml

with open("config/dataset.yaml") as f:
    cfg = yaml.safe_load(f)

train_path = cfg["liar"]["train_path"]
```

### 3.5 Reproducibility
- Always set random seeds at the top of training scripts:

```python
import random, numpy as np, torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

- Local execution on Apple Silicon is the default project workflow; the same seed policy applies
  when PyTorch runs on `mps`

### 3.6 Device Handling (PyTorch)
- Always use dynamic device detection, never hardcode `"cuda"` or `"cpu"`
- Preferred device order for this repository is: `cuda`, then `mps`, then `cpu`

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = model.to(device)
```

### 3.7 Logging
- Use Python's built-in `logging` module in scripts, **not** `print()`
- Notebooks may use `print()` for readability

```python
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

logger.info(f"Loaded {len(df)} training examples")
```

### 3.8 Saving Models
- Classical ML models: save with `joblib.dump(model, path)`
- PyTorch models: save with `torch.save(model.state_dict(), path)`
- Always save to `models/` directory (which is gitignored)

### 3.9 Error Handling
- Wrap file I/O in try/except with informative messages
- Validate that data files exist before attempting to load them

```python
if not Path(train_path).exists():
    raise FileNotFoundError(f"LIAR train file not found at {train_path}. Run scripts/download_data.py first.")
```

---

## 4. Data Conventions

### 4.1 LIAR Dataset Schema
When loading the raw TSV files, always assign these exact column names in this order:

```python
LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker",
    "job", "state", "party",
    "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts", "context"
]
```

### 4.2 Label Encoding
Always use this exact mapping (order matters for evaluation):

```python
LABEL_MAP = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5
}
LABEL_NAMES = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
```

### 4.3 Credibility Vector
The 6-dimensional credibility vector for each speaker is always constructed in this order:

```python
CREDIBILITY_COLS = [
    "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts"
]
# Normalize by total count to get ratios summing to 1.0
```

### 4.4 Preprocessing Versions
Two versions of text are maintained throughout the project:

| Version | Used for | Description |
|---|---|---|
| `statement_clean` | Classical ML (TF-IDF) | Lowercased, no URLs, lemmatized, no stopwords |
| `statement_raw` | Transformer (RoBERTa) | Lowercased, URLs removed, punctuation kept |

---

## 5. Key Design Decisions

These decisions are final. Do not change them without updating this file.

| Decision | Choice | Reason |
|---|---|---|
| Transformer model | `roberta-base` | Strong performance on political text, manageable size |
| Classification type | Multi-class (6 labels) | Reflects real-world truthfulness nuance vs. binary |
| Primary metric | Macro-F1 | Penalizes failure on rare classes equally |
| Context features | Credibility vector + speaker/party embeddings | Direct implementation of paper Section 3.3 |
| Fusion method | Concatenation + MLP | Simple, interpretable, proven effective |
| Class imbalance strategy | `WeightedRandomSampler` | Dynamic sampling per paper Section 4.1 |
| Fine-tuning strategy | 3-phase phased training | Prevents context branch from overpowering text branch |
| Seed | 42 | Fixed for reproducibility |

---

## 6. Running the Project

### Full pipeline (in order):
```bash
python scripts/download_data.py       # Step 1: Get LIAR data
python scripts/preprocess.py          # Step 2: Clean + feature engineer
python scripts/train_baseline.py      # Step 3: Train SVM, NB, RF
python scripts/train_transformer.py   # Step 4: Fine-tune RoBERTa locally; prefer MPS on Apple Silicon
# Step 5: Hybrid model — run locally first; use cloud only if a specific run exceeds local limits
```

### Makefile shortcuts:
```bash
make install      # creates .venv and installs project deps
make download     # runs download_data.py
make smoke        # verifies local dataset loading
make preprocess   # runs preprocess.py
make baseline     # runs train_baseline.py
make transformer  # runs train_transformer.py
make test         # runs pytest tests/
```

### Local environment setup:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[ml,viz,dev]"
```

### Optional cloud fallback:
- Google Colab or Kaggle may be used only if a specific transformer or hybrid experiment proves
  too slow or too memory-intensive locally
- Cloud notebooks are not the default execution path for this repository

---

## 7. Dependencies

All dependencies are declared in `pyproject.toml`. The project uses **optional dependency groups**:

```toml
[project.optional-dependencies]
ml = [
    "scikit-learn",
    "torch",
    "transformers",
    "datasets",
    "numpy",
    "scipy",
    "joblib",
    "pyyaml",
]
viz = [
    "matplotlib",
    "seaborn",
]
dev = [
    "pytest",
    "jupyter",
]
```

Install everything: `pip install -e ".[ml,viz,dev]"`

---

## 8. What AI Agents Must NOT Do

- ❌ Do not hardcode any file paths — always use config files or `pathlib.Path`
- ❌ Do not add new dependencies without updating `pyproject.toml`
- ❌ Do not modify `LABEL_MAP` or `LIAR_COLUMNS` — everything downstream depends on them
- ❌ Do not use `print()` in scripts (use `logging`)
- ❌ Do not commit files to `data/` or `models/` directories
- ❌ Do not change the 3-phase training strategy without updating Section 5 above
- ❌ Do not write functions longer than ~50 lines — break them up
- ❌ Do not leave TODO comments — either implement it or raise a `NotImplementedError` with a description

---

## 9. Progress Log

> **Instructions for agents:** After completing any task, append an entry here.
> Format: `### [YYYY-MM-DD] — <what was done>`
> Be specific — list files created/modified and key outputs.

---

### [2026-03-06] — Project initialized

**Status:** EDA complete, all other files are empty shells.

**Completed:**
- `notebooks/01_eda_liar.ipynb` — label distribution, text length histogram, VADER sentiment analysis, top speakers, top subjects, sentiment by class

**Created (empty, pending implementation):**
- `src/disinfo_detection/data_loader.py`
- `src/disinfo_detection/preprocessing.py`
- `src/disinfo_detection/models_baseline.py`
- `src/disinfo_detection/models_transformers.py`
- `config/dataset.yaml`, `config/baseline.yml`, `config/transformer.yaml`
- `scripts/download_data.py`, `scripts/preprocess.py`, `scripts/train_baseline.py`, `scripts/train_transformer.py`

**Not yet created (must be built from scratch):**
- `src/disinfo_detection/models_hybrid.py`
- `src/disinfo_detection/evaluation.py`
- `notebooks/05_hybrid_model.ipynb`
- `notebooks/06_evaluation_results.ipynb`

**Known issues:**
- `pyproject.toml` only has `matplotlib`, `nltk`, `pandas`, `seaborn` — all ML dependencies missing
- All config YAML files are empty

**Next task:** Week 1 — update `pyproject.toml`, fill `config/dataset.yaml`, write `data_loader.py`

---

<!-- AGENTS: ADD NEW LOG ENTRIES BELOW THIS LINE -->

### [2026-03-06] — Completed Task 1.1 `pyproject.toml`

**Modified:**
- `pyproject.toml` — replaced placeholder package metadata with the project description, added base dependencies (`numpy`, `pyyaml`), defined optional `ml`, `viz`, and `dev` dependency groups, and added Hatch build-system configuration for packaging `src/disinfo_detection`

**Validation:**
- Parsed `pyproject.toml` locally with Python `tomllib`
- Checked current environment imports for `torch`, `transformers`, and `sklearn`

**Notes:**
- Dependency installation itself was not run in this task

### [2026-03-06] — Task 1.1 validation status documented

**Validation details:**
- `pyproject.toml` parses successfully with Python `tomllib`
- Acceptance import check could not be completed in the current environment because `torch` is not installed
- Attempted isolated `.venv` installation, but dependency download was blocked before `hatchling` and ML packages could be fetched

### [2026-03-06] — Completed Week 1 Tasks 1.1 to 1.4

**Modified:**
- `pyproject.toml` — completed Task 1.1 package metadata, optional dependency groups, and Hatch build configuration
- `config/dataset.yaml` — added LIAR paths, ordered column names, quoted string labels for YAML-safe `label_map`, `label_names`, `credibility_cols`, and `metadata_cols`
- `scripts/download_data.py` — implemented config-driven LIAR download/export script with `logging`, ordered TSV output, `data/` creation, and clean skip behavior when files already exist
- `src/disinfo_detection/data_loader.py` — implemented `load_config`, `load_liar`, `get_label_map`, and `get_splits` with type hints, Google-style docstrings, split validation, file existence checks, and `label_id` generation

**Validation:**
- Parsed `pyproject.toml` successfully with Python `tomllib`
- Loaded `config/dataset.yaml` successfully with `yaml.safe_load`
- Ran `scripts/download_data.py` through the local `.venv`; existing raw files were detected and the script exited cleanly with `Data already exists, skipping.`
- Ran `load_liar("train")` through the local `.venv`; returned shape `(10269, 15)`, included `label_id`, and all `label_id` values were within `0..5`
- Ran `get_splits()` through the local `.venv`; returned train, validation, and test DataFrames with shapes `(10269, 15)`, `(1284, 15)`, and `(1283, 15)`

**Notes:**
- `config/dataset.yaml` uses quoted `"false"` and `"true"` labels to avoid YAML boolean coercion, while preserving the required label strings
- Task 1.1 import acceptance check (`import torch`, `transformers`, `sklearn`) remains blocked in the current environment because those optional ML dependencies are not installed locally

### [2026-03-06] — Updated LIAR downloader to direct HTTP TSV downloads

**Modified:**
- `scripts/download_data.py` — replaced the `load_dataset("liar")` implementation with direct `requests`-based downloads from the Hugging Face TSV URLs for `train`, `valid`, and `test`; kept config-driven local output paths, `data/` creation, existing skip-if-all-files-exist behavior, and `logging`
- `pyproject.toml` — added `requests>=2.32.0` to base dependencies and preserved the updated `pandas>=2.2.2,<3.0.0` constraint

**Validation:**
- Compiled `scripts/download_data.py` successfully with `python -m py_compile`
- Ran `scripts/download_data.py` through the local `.venv`; existing raw files were detected and the script exited cleanly with `Data already exists, skipping.`
- Parsed `pyproject.toml` and confirmed the base dependency list includes `requests` and the new `pandas` version range

### [2026-03-06] — Improved LIAR downloader skip behavior for partial downloads

**Modified:**
- `scripts/download_data.py` — replaced all-or-nothing skip logic with per-file checks so reruns skip only the existing TSV files and continue downloading any missing split files

**Validation:**
- Compiled `scripts/download_data.py` successfully with `python -m py_compile`
- Ran `scripts/download_data.py` through the local `.venv`; confirmed it now logs individual skip messages for `data/train.tsv`, `data/valid.tsv`, and `data/test.tsv`

### [2026-03-06] — Switched LIAR downloader to original UCSB zip source

**Modified:**
- `scripts/download_data.py` — replaced the Hugging Face loader implementation with direct download of `https://www.cs.ucsb.edu/~william/data/liar_dataset.zip`, added in-memory zip handling, archive member lookup by filename, extraction of `train.tsv`, `valid.tsv`, and `test.tsv` to the config-defined local paths, and preserved per-file skip behavior

**Validation:**
- Compiled `scripts/download_data.py` successfully with `python -m py_compile`
- Ran `scripts/download_data.py` through the local `.venv`; existing local split files were skipped individually and the script exited cleanly with `All LIAR split files already exist.`

### [2026-04-03] — Repository review and current-state assessment

**Modified:**
- `AGENTS.md` — added this review log entry after a full repository walkthrough

**Reviewed:**
- Project documents: `AGENTS.md`, `PROJECT_PLAN.md`, `README.md`
- Project metadata and repo hygiene: `pyproject.toml`, `Makefile`, `.gitignore`, `git status --short`
- Configs: `config/dataset.yaml`, `config/baseline.yml`, `config/transformer.yaml`
- Source package: `src/disinfo_detection/*.py`
- Scripts: `scripts/*.py`
- Notebooks: file inventory plus structural inspection of `notebooks/01_eda_liar.ipynb`
- Test surface: `tests/`

**Validation:**
- Confirmed local LIAR split files exist at `data/train.tsv`, `data/valid.tsv`, and `data/test.tsv`
- Ran `load_liar("train")` and `get_splits()` through `.venv/bin/python`; returned shapes `(10269, 15)`, `(1284, 15)`, and `(1283, 15)` with all label ids `0..5`
- Confirmed the active system Python cannot import `pandas`, and `.venv/bin/python -m pytest -q` fails because `pytest` is not installed

**Findings:**
- The repository is still in scaffold stage: only `scripts/download_data.py`, `config/dataset.yaml`, and `src/disinfo_detection/data_loader.py` contain substantive implementation
- `config/baseline.yml`, `config/transformer.yaml`, `src/disinfo_detection/preprocessing.py`, `src/disinfo_detection/models_baseline.py`, `src/disinfo_detection/models_transformers.py`, `scripts/preprocess.py`, `scripts/train_baseline.py`, and `scripts/train_transformer.py` are empty
- `README.md` is only a title line, `Makefile` is empty, and `main.py` is a placeholder script using `print()`
- `notebooks/02_preprocessing.ipynb`, `notebooks/03_baseline_training.ipynb`, and `notebooks/04_transformer_experiments.ipynb` are zero-byte files rather than valid notebooks
- `tests/` has no test modules yet

### [2026-04-03] — Confirmed local PDF plan documents are accessible

**Modified:**
- `AGENTS.md` — added this PDF access check entry

**Reviewed:**
- `MSRW 1 Daniyar Koishin Final.pdf`
- `ИПМ_Daniyar_Koishin_Final.pdf`

**Validation:**
- Confirmed both PDF files exist in the repository root and are readable as local files
- Extracted embedded PDF title metadata from both files:
  - `MSRW 1 Daniyar Koishin Final`
  - `ИПМ_Daniyar_Koishin_Final`

**Notes:**
- Poppler utilities (`pdfinfo`, `pdftotext`) are not installed in the current environment, so deeper extraction or rendered page review would need either those tools or a Python PDF library to be installed first

### [2026-04-03] — Updated repository for local-first Apple Silicon usage

**Modified:**
- `README.md` — replaced placeholder content with local installation, local workflow, and current status documentation
- `Makefile` — added local `.venv` bootstrap, install, smoke, training, test, and clean targets
- `config/baseline.yml` — filled baseline model hyperparameters
- `config/transformer.yaml` — filled transformer training defaults with local runtime settings, including `mps` priority
- `PROJECT_PLAN.md` — updated roadmap assumptions to local-first execution and Apple Silicon device selection
- `AGENTS.md` — updated runtime guidance from generic GPU/cloud assumptions to local Apple Silicon usage

**Validation:**
- Parsed `config/baseline.yml` and `config/transformer.yaml` successfully with `yaml.safe_load`
- Ran `make -n install smoke download preprocess baseline transformer test clean` to verify target structure without executing destructive or networked actions

**Notes:**
- Local-first readiness now exists at the documentation, config, and command-surface level
- Preprocessing, baseline training, transformer training, hybrid modeling, and automated test cases remain implementation work, not just environment work

### [2026-04-03] — Added initial IEEE conference LaTeX paper scaffold

**Modified:**
- `reports/paper/main.tex` — created an IEEE conference paper starter with preamble, title block, abstract, keywords, and placeholders for Introduction, Literature Review, Methodology, Results, and Conclusion
- `reports/paper/references.bib` — created an initial BibTeX placeholder file for future citations

**Validation:**
- Confirmed no prior `.tex` or `.bib` paper scaffold existed in the repository before creation

**Notes:**
- The document uses the `IEEEtran` conference class and is intended to be expanded iteratively as theory, methods, and results are finalized

### [2026-04-03] — Drafted first substantive paper sections from current project state

**Modified:**
- `reports/paper/main.tex` — replaced section placeholders with initial academic prose for the Abstract, Introduction, Literature Review, Methodology, Results, and Conclusion

**Validation:**
- Confirmed the draft remains structurally consistent with the IEEE conference template scaffold already added to the repository

**Notes:**
- The draft intentionally avoids claiming experimental performance that has not yet been produced
- The Results and Conclusion sections currently reflect the verified project state rather than final study outcomes

### [2026-04-03] — Extracted uploaded PDFs and aligned next implementation priorities

**Modified:**
- `AGENTS.md` — added this planning log entry after extracting the uploaded MSRW and master's work plan PDFs

**Validation:**
- Installed `pypdf` into the local `.venv`
- Extracted readable text from `MSRW 1 Daniyar Koishin Final.pdf` and `ИПМ_Daniyar_Koishin_Final.pdf`

**Findings:**
- The MSRW document supports the staged technical progression from classical ML baselines to transformer models and then to context-aware or hybrid methods
- The formal master's work plan explicitly includes a baseline pipeline with preprocessing, TF-IDF, Naive Bayes, SVM, and Random Forest, followed by a transformer-based experimental framework and comparative evaluation
- This matches the current repository roadmap and confirms that preprocessing plus baseline implementation is the immediate critical path

### [2026-04-03] — Implemented and validated the preprocessing pipeline

**Modified:**
- `src/disinfo_detection/preprocessing.py` — implemented text cleaning for TF-IDF and transformer inputs, credibility vector construction, and DataFrame-level preprocessing
- `scripts/preprocess.py` — implemented the end-to-end preprocessing script that loads all LIAR splits and saves processed pickle artifacts
- `tests/test_preprocessing.py` — added unit tests for text cleaning, credibility-vector construction, and DataFrame preprocessing outputs

**Validation:**
- Compiled `src/disinfo_detection/preprocessing.py`, `scripts/preprocess.py`, and `tests/test_preprocessing.py` with `python -m py_compile`
- Ran `.venv/bin/python -m pytest tests/test_preprocessing.py -q`; all 5 tests passed
- Ran `.venv/bin/python scripts/preprocess.py`; generated `data/processed/train.pkl`, `valid.pkl`, and `test.pkl`
- Loaded all three processed pickle files and confirmed they contain `statement_clean`, `statement_transformer`, `statement_raw`, `credibility_vector`, and `credibility_0` through `credibility_4`

**Notes:**
- The preprocessing implementation avoids hidden network dependencies by using local normalization and stemming rather than downloading external NLTK corpora at runtime
- `statement_raw` is currently stored as an alias of `statement_transformer` to keep the repository compatible with both the roadmap terminology and the existing agent conventions

### [2026-04-03] — Implemented and ran the baseline modeling pipeline

**Modified:**
- `src/disinfo_detection/models_baseline.py` — implemented the `TFIDFBaseline` class with TF-IDF vectorization, SVM, Naive Bayes, Random Forest support, probability estimation, and model persistence
- `src/disinfo_detection/evaluation.py` — created reusable metric computation, confusion-matrix plotting, and model-comparison utilities
- `scripts/train_baseline.py` — implemented end-to-end baseline training on processed LIAR data and CSV summary export
- `tests/test_baseline.py` — added unit tests for baseline training/persistence and evaluation metrics

**Validation:**
- Installed `scikit-learn` and `scipy` into the local `.venv`
- Compiled `models_baseline.py`, `evaluation.py`, `train_baseline.py`, and `test_baseline.py` with `python -m py_compile`
- Ran `.venv/bin/python -m pytest tests/test_baseline.py tests/test_preprocessing.py -q`; all 7 tests passed
- Ran `.venv/bin/python scripts/train_baseline.py`; saved model artifacts and `reports/baseline_results.csv`

**Outputs:**
- `models/baseline_svm.pkl`
- `models/baseline_naive_bayes.pkl`
- `models/baseline_random_forest.pkl`
- `reports/baseline_results.csv`

**Observed validation metrics on `valid.pkl`:**
- `svm` — accuracy `0.2539`, macro-F1 `0.2533`
- `naive_bayes` — accuracy `0.2329`, macro-F1 `0.1814`
- `random_forest` — accuracy `0.2399`, macro-F1 `0.2379`

**Notes:**
- The current SVM implementation uses a linear margin-based model that is efficient on sparse TF-IDF features; probability-like outputs are derived from the decision scores for API consistency
- Plotting imports in `evaluation.py` were made lazy so metric-only runs do not trigger unnecessary Matplotlib cache warnings

### [2026-04-03] — Added run history and transformer-stage scaffolding

**Modified:**
- `src/disinfo_detection/evaluation.py` — added run-history appending and transformer training-curve plotting helpers
- `scripts/train_baseline.py` — added baseline run-history logging and made figure generation optional behind `ENABLE_FIGURES=1`
- `src/disinfo_detection/models_transformers.py` — implemented `LIARDataset`, `RoBERTaClassifier`, and transformer config loading
- `scripts/train_transformer.py` — implemented the RoBERTa training script with local device resolution, checkpoint saving, CSV training logs, optional figure generation, and run-history logging
- `tests/test_transformers.py` — added transformer unit tests using dummy tokenizer/model components

**Validation:**
- Installed `torch`, `transformers`, and `tokenizers` into the local `.venv`
- Compiled `evaluation.py`, `models_transformers.py`, `train_transformer.py`, and `test_transformers.py` with `python -m py_compile`
- Ran `.venv/bin/python -m pytest tests/test_preprocessing.py tests/test_baseline.py tests/test_transformers.py -q`; all 9 tests passed
- Re-ran `.venv/bin/python scripts/train_baseline.py`; saved `reports/baseline_run_history.csv` successfully in stable mode

**Notes:**
- Figure generation was the likely cause of the earlier Python crash due local font-cache issues in the current environment
- Baseline and transformer training now default to stable non-plotting mode; enable PNG figure generation explicitly with `ENABLE_FIGURES=1`

### [2026-04-08] — Reviewed groupmate task documents

**Modified:**
- `AGENTS.md` — added this review log entry

**Reviewed:**
- `/Users/doncheck/Downloads/Mariyam tasks 1-3.docx`
- `/Users/doncheck/Downloads/task1-3.docx`
- `/Users/doncheck/Downloads/Asset Alibek Все Задачи Научная Стажировка.docx`

**Findings:**
- Identified the common assignment structure across the documents: Task 1 defines the research problem and background, Task 2 describes data sources and collection/storage methods, and Task 3 presents visual/result analysis with patterns, trends, and anomalies.

**Notes:**
- Used macOS `textutil` to extract document text because `python-docx` was not installed in the local `.venv`.

### [2026-04-08] — Created Task 1-3 LaTeX report

**Modified:**
- `reports/task1_task3/task1_task3_report.tex` — created a self-contained LaTeX report covering Task 1 research problem/background, Task 2 data sources and collection/storage methods, and Task 3 visualization/result analysis for the disinformation detection dissertation
- `AGENTS.md` — added this completion log entry

**Reviewed:**
- `/Users/doncheck/Downloads/MSRW 1 Daniyar Koishin Final.docx`
- `/Users/doncheck/Downloads/ИПМ_Daniyar_Koishin_Final.docx`
- Existing local result files in `reports/baseline_results.csv` and `reports/transformer_logs/training_log.csv`

**Validation:**
- Extracted both dissertation DOCX files with macOS `textutil`
- Checked the generated LaTeX source for non-ASCII characters
- Checked major LaTeX begin/end structure and section structure with `rg`

**Notes:**
- PDF compilation was not run because `pdflatex`, `xelatex`, and `latexmk` are not installed in the current environment
- Task 3 uses real local baseline and RoBERTa validation metrics; the hybrid model is clearly described as planned/future work rather than a completed result

### [2026-04-08] — Added graphics to Task 1-3 LaTeX report

**Modified:**
- `reports/task1_task3/task1_task3_report.tex` — added `graphicx` and embedded generated Task 3 figures
- `reports/task1_task3/figures/baseline_macro_f1.png` — baseline macro-F1 bar chart
- `reports/task1_task3/figures/roberta_macro_f1_by_epoch.png` — RoBERTa validation macro-F1 line chart
- `reports/task1_task3/figures/roberta_loss_curve.png` — RoBERTa training and validation loss chart
- `AGENTS.md` — added this completion log entry

**Validation:**
- Generated figures from existing CSV outputs in `reports/baseline_results.csv` and `reports/transformer_logs/training_log.csv`
- Set `MPLCONFIGDIR=/tmp/codex-mpl-cache` for plotting because the default Matplotlib cache directory was not writable
- Checked that the LaTeX source references the generated figure files

### [2026-04-08] — Revised Task 1-3 report wording

**Modified:**
- `reports/task1_task3/task1_task3_report.tex` — removed explicit Task 1/Task 2/Task 3 and Step labels from the LaTeX content, retitled the report, and revised wording to sound more like a student-written dissertation progress report while preserving the same results and figures
- `AGENTS.md` — added this completion log entry

**Validation:**
- Confirmed no `Task` or `Step` wording remains in the LaTeX source
- Confirmed figure references are still present
- Checked the LaTeX source for non-ASCII characters

### [2026-04-09] — Created internship DOCX document package

**Modified:**
- `output/doc/internship_docs/src_html/04_syllabus_nlp_methods_for_disinformation_detection.html` — prepared the content source for the internship syllabus
- `output/doc/internship_docs/src_html/05_article_title_abstract_journals.html` — prepared the content source for the article title, abstract, and journal selection task
- `output/doc/internship_docs/src_html/06_scopus_zotero_bibliography.html` — prepared the content source for the Scopus query, Zotero, and bibliography task
- `output/doc/internship_docs/src_html/07_project_passport.html` — prepared the content source for the project passport task
- `output/doc/internship_docs/src_html/updated_internship_report.html` — prepared the updated internship report content source
- `output/doc/internship_docs/generate_docx_from_html.py` — added a small standard-library DOCX generator to convert the prepared HTML content into plain Word documents with Times New Roman formatting
- `output/doc/internship_docs/Научная стажировка - задание 4 - Силлабус.docx`
- `output/doc/internship_docs/Научная стажировка - задание 5 - Статья и журналы.docx`
- `output/doc/internship_docs/Научная стажировка - задание 6 - Scopus и библиография.docx`
- `output/doc/internship_docs/Научная стажировка - задание 7 - Паспорт проекта.docx`
- `output/doc/internship_docs/Отчет о научной стажировке - обновленный.docx`
- `AGENTS.md` — added this completion log entry

**Reviewed:**
- `/Users/doncheck/Downloads/OneDrive_1_4-9-2026/науч стаж задание 1.docx`
- `/Users/doncheck/Downloads/OneDrive_1_4-9-2026/науч стаж задание 2.docx`
- `/Users/doncheck/Downloads/OneDrive_1_4-9-2026/науч стаж задание 3.docx`
- `/Users/doncheck/Downloads/OneDrive_1_4-9-2026/науч стаж задание 4.docx`
- `/Users/doncheck/Downloads/OneDrive_1_4-9-2026/науч стаж задание 5.docx`
- `/Users/doncheck/Downloads/OneDrive_1_4-9-2026/науч стаж задание 6.docx`
- `/Users/doncheck/Downloads/OneDrive_1_4-9-2026/науч стаж задание 7.docx`
- `/Users/doncheck/Downloads/OneDrive_1_4-9-2026/Syllabus Kulbossynov Alisher.doc`
- `/Users/doncheck/Downloads/Отчет стажировки.docx`

**Validation:**
- Generated all five DOCX files locally
- Verified the generated report and bibliography DOCX files by extracting text with `textutil`
- Verified that the generated DOCX styles specify `Times New Roman`

**Notes:**
- Direct HTML to DOCX conversion via `textutil` was not reliable in this environment, so the final DOCX files were generated directly in WordprocessingML format using Python standard library only

### [2026-04-22] — Removed large model artifacts from git history head and pushed clean commit

**Modified:**
- `.gitignore` — added `data/` and `models/` to keep generated training artifacts out of version control
- `models/` — removed tracked model artifacts from the index, including `models/roberta_liar/best_model.pt` and the baseline `.pkl` files, while leaving the local files on disk

**Validation:**
- Confirmed `HEAD` no longer contains `models/roberta_liar/best_model.pt`
- Verified `git push origin main:main` completed successfully after the cleanup

**Notes:**
- The previous HTTP 408 was caused by the 498 MB checkpoint being included in the push pack
- The cleaned commit is now on `origin/main`
