# AGENTS.md — Context for AI agents

> Read this file in full before modifying the repository.
> After any substantive change, append a dated entry to the Progress Log at
> the bottom (format: `### [YYYY-MM-DD] — short title`).

---

## 1. Project at a glance

- **Project**: NLP methods for detecting disinformation in social networks
- **Author**: Daniyar Koishin — MSc student, Astana IT University (AITU), Kazakhstan
- **Program**: 7M06105 Computer Science and Engineering, 2025–2027
- **Supervisor**: Tamara Zhukabayeva, PhD, Professor
- **Thesis defense**: June 2027
- **Planned publications**: IEEE SIST 2026 conference paper + one Scopus journal
- **Dataset**: LIAR (12 836 labeled political statements, 6-class truthfulness scale)
- **Main contribution**: hybrid classifier that fuses a fine-tuned RoBERTa text
  encoder with a metadata branch (speaker credibility history + hashed
  categorical context). The text-only branch is a strict baseline; the
  metadata-augmented variant is the novel part of the thesis.
- **Primary metric**: macro-F1 on the TEST split. Secondary: accuracy, per-class F1, confusion matrix.

---

## 2. Repository map

```
nlp-disinformation-detection-aitu/
├── AGENTS.md                          ← this file
├── README.md                          ← human-facing quick start
├── Makefile                           ← make install / download / preprocess / baseline / transformer / hybrid / test
├── pyproject.toml                     ← dependencies (Python 3.12, optional groups: ml, viz, dev)
├── main.tex                           ← paper source for Overleaf / Tectonic
├── references.bib                     ← paper bibliography
├── .python-version                    ← 3.12
├── .gitignore                         ← excludes data/, models/, reports/figures/, .venv, caches
│
├── config/
│   ├── dataset.yaml                   ← LIAR paths, column order, label map, metadata columns
│   ├── baseline.yml                   ← TF-IDF + classical ML hyperparameters
│   ├── transformer.yaml               ← text-only RoBERTa fine-tuning config
│   └── hybrid.yaml                    ← hybrid (text + metadata) fine-tuning config
│
├── data/                              ← GITIGNORED. Created by scripts/download_data.py.
│   ├── train.tsv / valid.tsv / test.tsv
│   └── processed/                     ← .pkl files produced by scripts/preprocess.py
│
├── models/                            ← GITIGNORED. Training writes checkpoints here.
│
├── notebooks/
│   ├── 01_eda_liar.ipynb              ← EDA: label dist, length hist, VADER sentiment, top speakers/subjects
│   ├── training.ipynb                 ← clean remote GPU training notebook
│   └── _build_training_notebook.py    ← generator for training.ipynb
│
├── reports/
│   ├── __init__.py                    ← marker for pytest discovery
│   ├── figures/                       ← PNGs (gitignored contents)
│   ├── transformer_logs/              ← written by train_transformer.py (gitignored contents)
│   └── hybrid_logs/                   ← written by train_hybrid.py (gitignored contents)
│
├── scripts/                           ← runnable CLI entry points
│   ├── download_data.py               ← downloads LIAR from UCSB zip
│   ├── preprocess.py                  ← builds statement_clean / statement_transformer / credibility features
│   ├── train_baseline.py              ← TF-IDF + NB / SVM / RF baselines
│   ├── train_transformer.py           ← text-only RoBERTa
│   ├── train_hybrid.py                ← hybrid (text + metadata), also supports text-only ablation via config
│   ├── verify_leakage.py              ← leakage verification reports
│   ├── bootstrap_ci.py                ← bootstrap confidence intervals
│   ├── package_artifacts.py           ← Kaggle results.zip / models.zip packager
│   ├── import_results_archive.py      ← local importer for disinformation_results.zip
│   ├── package_overleaf.py            ← Overleaf zip packager
│   ├── plot_per_class_f1.py           ← paper per-class F1 figure
│   ├── plot_confusion_matrices.py     ← paper confusion-matrix figure
│   └── plot_training_curves.py        ← paper training-curves figure
│
├── src/disinfo_detection/             ← core importable package
│   ├── data_loader.py                 ← load_liar(split), get_splits(), label maps
│   ├── preprocessing.py               ← clean_text_for_tfidf / clean_text_for_transformer / credibility features
│   ├── evaluation.py                  ← compute_metrics(), append_run_history(), plot_training_history()
│   ├── models_baseline.py             ← TFIDFBaseline (NB / SVM / RF)
│   ├── models_transformers.py         ← LIARDataset + RoBERTaClassifier
│   ├── metadata_features.py           ← deterministic hashing for categorical metadata, dense matrix builder
│   ├── datasets_hybrid.py             ← HybridLIARDataset (text + metadata)
│   └── models_hybrid.py               ← HybridClassifier, MetadataBranch, HybridTrainer (THE THESIS NOVELTY)
│
├── docs/
│   ├── TRAINING_IMPROVEMENTS.md       ← rationale for preprocessing + transformer tuning changes
│   ├── HYBRID_MODEL.md                ← architecture and design decisions for the hybrid model
│   └── KAGGLE_TRAINING.md             ← clean remote GPU training guide
│
└── tests/                             ← pytest suite (16/16 green on main)
    ├── test_preprocessing.py
    ├── test_baseline.py
    ├── test_transformers.py
    └── test_hybrid.py
```

---

## 3. Current state (as of 2026-04-22)

| Component | Status |
|---|---|
| Data loading (`data_loader.py`, `download_data.py`) | ✅ Implemented |
| Preprocessing (`preprocessing.py`, `scripts/preprocess.py`) | ✅ Implemented + tuned (see `docs/TRAINING_IMPROVEMENTS.md`) |
| Classical baselines (`models_baseline.py`, `scripts/train_baseline.py`) | ✅ Implemented |
| Text-only RoBERTa (`models_transformers.py`, `scripts/train_transformer.py`) | ✅ Implemented + tuned |
| Hybrid text + metadata model (`models_hybrid.py`, `metadata_features.py`, `datasets_hybrid.py`, `scripts/train_hybrid.py`) | ✅ Implemented, tested |
| Evaluation utilities (`evaluation.py`) | ✅ Implemented |
| Tests | ✅ 16/16 passing (`make test`) |
| Full end-to-end local training on Apple Silicon | ⏳ Pending (author needs to run `make train-all`) |
| Final numbers in MSRW 2 §5 | ⏳ `[to be filled]` placeholders remain until a full run completes |
| IEEE SIST 2026 paper draft | ⏳ Scaffolded but not aligned with final numbers yet |

---

## 4. How to run

### First-time setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[ml,viz,dev]"
```

Or, equivalently: `make install`.

### Full pipeline

```bash
make download        # downloads LIAR from UCSB into data/
make preprocess      # writes data/processed/{train,valid,test}.pkl
make baseline        # TF-IDF + NB / SVM / RF → reports/
make transformer     # text-only RoBERTa → reports/transformer_logs/
make hybrid          # text + metadata → reports/hybrid_logs/
make hybrid-textonly # RQ2 ablation: hybrid pipeline with use_metadata=false
make test            # runs pytest
```

### Device selection

Every training script picks a device in this priority order: `cuda` → `mps` → `cpu`.
Primary development target is macOS on Apple Silicon (`mps`). Never hardcode a device.

---

## 5. Coding conventions

- **Python 3.12+**, PEP 8, max line length 100 chars.
- **Type hints** on every function signature.
- **Google-style docstrings** on all public functions and classes.
- **No `print()` in scripts** — use `logging`. Notebooks may use `print`.
- **No wildcard imports.** Use `from module import name`.
- **No hardcoded paths or hyperparameters** — put them in `config/*.yaml`.
- **Always set seeds** (`random`, `numpy`, `torch`) at the top of training scripts. Default seed is 42.
- **File I/O** should use `pathlib.Path`, not string concatenation.
- **PyTorch device handling** must check `cuda` → `mps` → `cpu` dynamically.
- **Checkpoints**: `torch.save(model.state_dict(), path)`. Classical: `joblib.dump`. Write to `models/` (gitignored).
- **Never commit** anything under `data/`, `models/`, `reports/figures/*.png`, `reports/*_logs/*.csv`, or virtualenv directories.

---

## 6. Dataset conventions (do not change)

### 6.1 LIAR column order (used when reading TSV)

```python
LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker",
    "job", "state", "party",
    "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts", "context",
]
```

### 6.2 Label map

```python
LABEL_MAP = {
    "pants-fire": 0, "false": 1, "barely-true": 2,
    "half-true": 3, "mostly-true": 4, "true": 5,
}
LABEL_NAMES = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
```

### 6.3 Credibility features (written by `preprocessing.py`)

Per-row columns produced by `preprocess_dataframe`:

- `credibility_0..4` — normalized 5-dim probability vector over the 5 prior-statement count columns.
- `cred_total`, `cred_log_total`, `cred_pants_share`, `cred_false_share` — scalar credibility summaries.
- `statement_clean` — lowercased, URL-stripped, stopword-filtered (negations kept), digits/`$`/`%` kept. Used by TF-IDF.
- `statement_transformer` — case preserved, URLs/HTML stripped, whitespace normalized. Used by RoBERTa.

### 6.4 Reporting split

All final metrics in the thesis and paper are reported on the **TEST split (1 283 examples)**, not the validation split.

---

## 7. Key design decisions (final unless this file is updated)

| Area | Choice | Rationale |
|---|---|---|
| Text encoder | `roberta-base` | Strong on short political text, fits on `mps` with batch 16. |
| Classification | Multi-class (6 labels) | Matches original LIAR task and the thesis RQs. |
| Primary metric | Macro-F1 | Penalizes failure on rare classes (`pants-fire`) equally. |
| Class imbalance | Inverse-sqrt-frequency class weights in CE | Full inverse-freq overshoots toward `pants-fire`; sqrt is the standard compromise. |
| Label smoothing | 0.05 | LIAR labels are ordinal; empirically +0.5–1.0 pp macro-F1. |
| Max token length | 64 | Mean statement length ≈ 17 tokens, p99 ≈ 40. 128 wasted compute. |
| Learning rate (text-only) | 1e-5 | 2e-5 routinely collapses to predicting `half-true` for every example. |
| Learning rate (hybrid) | 1e-5 encoder / 5e-4 head | Head + metadata branch are trained from scratch and tolerate a larger step. |
| Early stopping | Patience 2 on val macro-F1 | Most seeds converge in 3–5 epochs. |
| Metadata categoricals | Feature hashing, 256 buckets per field, blake2b with fixed salt | Avoids persisting a vocab; handles unseen speakers at test time; deterministic across runs. |
| Fusion | Concatenate `[CLS]` + metadata embedding → small MLP (128, GELU) → softmax | Simple and interpretable; a wider head overfits 10k training examples. |
| Ablation | Flip `model.use_metadata: false` in `config/hybrid.yaml` | Runs the exact same code path as the full hybrid — required for an apples-to-apples RQ2 comparison. |

---

## 8. What AI agents MUST NOT do

- ❌ Commit files under `data/`, `models/`, or run-time logs.
- ❌ Hardcode paths or hyperparameters outside `config/`.
- ❌ Modify `LIAR_COLUMNS` or `LABEL_MAP` — downstream depends on their order.
- ❌ Use `print()` inside scripts (notebooks are fine).
- ❌ Add a dependency without updating `pyproject.toml`.
- ❌ Remove the `use_metadata` ablation switch from `HybridClassifier` — the thesis RQ2 depends on it.
- ❌ Report numbers from the validation split as final thesis results — always evaluate on TEST.
- ❌ Add files that are not part of the thesis/paper pipeline (no internship artifacts, no unrelated scratch code).

---

## 9. Progress log

> Append newest entries at the bottom. Keep each entry short: what changed, what was validated, any open follow-ups.

### [2026-03-06] — Project initialized, EDA completed

- `notebooks/01_eda_liar.ipynb` — label distribution, text-length histogram, VADER sentiment, top speakers, top subjects.

### [2026-03-06] — Week 1: data pipeline landed

- `pyproject.toml` filled with base deps and `ml` / `viz` / `dev` optional groups.
- `config/dataset.yaml`, `scripts/download_data.py`, `src/disinfo_detection/data_loader.py` implemented.
- Validated: `load_liar("train")` returns `(10269, 15)`, `get_splits()` returns 10269 / 1284 / 1283 rows.

### [2026-04-03] — Local-first workflow, baseline/transformer configs filled

- Repo reframed for local macOS + Apple Silicon (`mps` device priority).
- `Makefile` expanded with install / smoke / preprocess / baseline / transformer / test targets.
- `config/baseline.yml` and `config/transformer.yaml` filled.

### [2026-04-03] — Classical baselines and text-only RoBERTa implemented

- `src/disinfo_detection/preprocessing.py` — dual-text preprocessing (TF-IDF vs. transformer).
- `src/disinfo_detection/models_baseline.py` — `TFIDFBaseline` with NB / SVM / RF.
- `src/disinfo_detection/models_transformers.py` — `LIARDataset` + `RoBERTaClassifier`.
- Corresponding `scripts/train_baseline.py` and `scripts/train_transformer.py`.

### [2026-04-22] — Training pipeline hardened (PR #1, `fix/training-improvements`)

- Preprocessing: dropped Porter stemming, preserved digits / `$` / `%`, case-preserving variant for transformer, negation-aware stopwords, added 4 scalar credibility features.
- Baseline config: `min_df=3`, `SVM C=0.5`, `NB alpha=0.3`, `RF` unbounded depth, 400 estimators.
- Transformer config: `lr=1e-5`, `max_length=64`, `batch=16 × accum 2`, `dropout=0.2`, `label_smoothing=0.05`, early stopping patience 2.
- `scripts/train_transformer.py`: inverse-sqrt-freq class weights, pre-tokenization, TEST evaluation, per-class F1 JSON dump.
- `docs/TRAINING_IMPROVEMENTS.md` captures the rationale for each change.

### [2026-04-22] — Hybrid text + metadata model (PR #2, `feat/hybrid-model`)

- `src/disinfo_detection/metadata_features.py` — deterministic blake2b feature hashing (256 buckets per field), dense matrix builder, scalar normalization.
- `src/disinfo_detection/models_hybrid.py` — `MetadataBranch`, `HybridClassifier` (with `use_metadata=False` ablation), `HybridTrainer`.
- `src/disinfo_detection/datasets_hybrid.py` — `HybridLIARDataset` packs tokenized text + metadata tensors.
- `scripts/train_hybrid.py` — two-LR optimizer (encoder 1e-5, head 5e-4), class-weighted CE, TEST reporting.
- `config/hybrid.yaml`, `docs/HYBRID_MODEL.md`, `make hybrid` / `make hybrid-textonly` targets.
- `tests/test_hybrid.py` — 6 new tests covering hashing, matrix shapes, forward+backward, ablation mode.
- Test suite now 16/16 green.

### [2026-04-22] — Repository cleanup (`chore/cleanup-docs`)

- Deleted stale artifacts: `output/doc/internship_docs/` (internship is complete and unrelated to the thesis), zero-byte notebooks 02/03/04, `main.py`, `utils/`, `PROJECT_PLAN.md`.
- Rewrote `AGENTS.md` (this file) from scratch so future agents get an accurate, compact view of the project.
- Rewrote `README.md` to reflect the current state.
- No code changes.

### [2026-04-24] — Pre-Kaggle instrumentation plan/code prepared

- Added the pre-Kaggle requirement that every model writes TEST prediction JSONL
  artifacts with probabilities/logits.
- Added environment fingerprints to metrics JSON outputs and updated the Kaggle
  notebook generator for multi-seed artifact collection.
- Replaced `PLAN.md` with a thesis roadmap that makes instrumentation the
  required Phase 0 before expensive Kaggle training.
- Validated with local unit tests and notebook regeneration before the next Kaggle run.

### [2026-04-25] — Kaggle results archive simplified

- Removed the optional GitHub push flow from the Kaggle notebook generator.
- Regenerated the remote training notebook so the final output is one
  required archive: `/kaggle/working/disinformation_results.zip`.
- The archive now contains metrics, prediction JSONL files, per-seed snapshots,
  logs, generated figures, and an `archive_manifest.json`.

### [2026-04-27] — Leakage-corrected credibility, ordinal loss, per-field hash buckets, multi-seed loop

- Preprocessing now emits `credibility_corrected_*` and `cred_*_corrected`
  columns alongside the raw versions; counts have the row's own verdict
  subtracted (label 5 / "true" has no counts column and is left unchanged).
  `metadata.leakage_corrected: true` in `config/hybrid.yaml` is the new
  defensible default; `make hybrid-leaky` reproduces the prior-art leaky
  setup for disclosure.
- `MetadataSpec.num_buckets` accepts a `{field: bucket_count}` dict;
  `MetadataBranch` derives per-field offsets via `field_offsets`. Default
  layout: `speaker=4096`, `subject/job/context=1024`, `state=256`, `party=64`.
- New `src/disinfo_detection/losses.py` provides `OrdinalAwareLoss` (weighted
  CE + squared EMD over softmax CDFs) and a `build_loss_module` config
  factory. Both `RoBERTaClassifier` and `HybridTrainer` route loss through
  the module; configs default to `loss.type: ordinal` with `emd_weight=0.5`.
- `metadata_output_dim` raised from 64 to 128 in `config/hybrid.yaml` so the
  metadata branch carries more weight relative to the 768-d `[CLS]` vector.
- `training.seeds` (list) drives an in-script multi-seed loop in both
  training scripts; per-seed JSON / JSONL / checkpoint artefacts are written
  with a `_seed{N}` suffix and an aggregate `*_test_metrics_multiseed.json`
  reports mean ± std macro-F1 / accuracy / per-class F1.
- `notebooks/_build_training_notebook.py::configure_seed` collapses the YAML
  `seeds` list to a single-element list per outer-loop seed, so the Kaggle
  outer loop continues to drive deterministic single-seed snapshots.
- Tests: 34/34 green (was 16/16). New coverage for corrected credibility
  columns, per-field bucket layout, EMD/ordinal loss properties, and seed
  aggregation.

### [2026-04-28] — Bootstrap confidence intervals added

- Added `scripts/bootstrap_ci.py` to compute 10000-resample bootstrap CIs for
  all 7 systems x 3 seeds and paired bootstrap differences for the planned
  hybrid/text-only/leaky comparisons.
- Added `reports/bootstrap_ci.json` and `reports/bootstrap_ci_table.md` with
  seed-averaged macro-F1 / accuracy intervals for paper tables.
- Added `make bootstrap-ci`, `tests/test_bootstrap_ci.py`, and `.gitignore`
  exceptions for the two bootstrap report artifacts; validated with
  `make bootstrap-ci` and `make test` (44/44 green).

### [2026-04-28] — Long et al. 2017 citation corrected

- Added `long2017fakenews` to `references.bib` and changed the Long et al.
  citations in Related Work, Table I, and Discussion from `shu2017` to the new
  source.
- Compiled `main.tex` with Tectonic; BibTeX finished cleanly and the rendered
  bibliography shows the Long et al. IJCNLP short-paper entry with the ACL
  Anthology URL.

### [2026-04-28] — Grouped per-class F1 figure added

- Added `scripts/plot_per_class_f1.py` to aggregate per-class F1 over seeds
  42, 1337, and 2024 for RoBERTa text-only, leakage-corrected hybrid, and
  leaky hybrid.
- Generated `figures/per_class_f1_grouped.pdf` and `.png`, replaced the old
  three-panel per-class Figure 2 block in `main.tex`, and added `make figures`.
- Validated the rendered PDF page with Poppler and ran `make test` (47/47
  green).

### [2026-04-28] — Side-by-side confusion matrices added

- Added `scripts/plot_confusion_matrices.py` to average leaky and
  leakage-corrected confusion matrices across seeds, row-normalise them, and
  render side-by-side annotated heatmaps.
- Generated `figures/confusion_matrices.pdf` and `.png`, replaced the old
  stacked seed-42 confusion-matrix figure in `main.tex`, and extended
  `make figures`.
- Validated the rendered PDF page with Poppler and ran `make test` (51/51
  green).

### [2026-04-28] — Paper finalisation and repository hygiene

- Removed the old horizontal macro-F1 figure from `main.tex`, added bootstrap
  confidence intervals to Table I, and verified final figure numbering as
  Fig. 1--3 with no unresolved PDF references.
- Applied the requested abstract, leakage, EMD, discussion, and limitations
  text edits; installed and ran `aspell --mode=tex` for manual typo review.
- Updated `README.md`, added `CITATION.cff`, tightened `.gitignore`, and
  verified `make figures`, `make test` (51/51 green), and Tectonic compilation.

### [2026-04-28] — Gitignore paper artifacts tightened

- Reviewed untracked and ignored repository outputs after paper/report generation.
- Added explicit ignore coverage for `paper_figures/` and common LaTeX auxiliary
  outputs while preserving committed paper figures under `figures/`.

### [2026-04-28] — Training notebook and artifact cleanup

- Renamed the remote GPU notebook to `notebooks/training.ipynb` and its generator
  to `notebooks/_build_training_notebook.py`.
- Replaced the old `reports/figures_all/` diagnostic gallery generation with
  only the paper artifacts: leakage reports, bootstrap CIs, and three final
  figure pairs under `figures/`.
- Added `scripts/plot_training_curves.py`, updated `make figures`, and removed
  stale local build/archive clutter from the working tree.

### [2026-04-28] — Reproducible Kaggle-to-Overleaf workflow added

- Renamed the manuscript source to `main.tex` for Overleaf compatibility.
- Added `scripts/package_artifacts.py` so Kaggle writes a minimal
  `disinformation_results.zip` and a separate optional `models.zip`.
- Added `scripts/import_results_archive.py` and `scripts/package_overleaf.py`
  plus `make import-results`, `make package-results-with-models`, and
  `make package-overleaf` for the local archive-to-paper workflow.
