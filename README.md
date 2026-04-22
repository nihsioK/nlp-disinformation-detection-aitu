# nlp-disinformation-detection-aitu

Hybrid NLP framework for detecting disinformation in political statements from
the LIAR dataset. The project combines a fine-tuned `roberta-base` text encoder
with a metadata branch that encodes speaker credibility history and hashed
categorical context features.

This repository is the code base for Daniyar Koishin's master's thesis at
Astana IT University (program 7M06105, 2025–2027). The planned outputs are an
IEEE SIST 2026 conference paper and a Scopus journal article, followed by the
thesis defense in June 2027.

## What is in the repo

| Area | Location |
|---|---|
| Classical baselines (TF-IDF + NB / SVM / RF) | `scripts/train_baseline.py`, `src/disinfo_detection/models_baseline.py` |
| Text-only RoBERTa | `scripts/train_transformer.py`, `src/disinfo_detection/models_transformers.py` |
| **Hybrid text + metadata model (thesis novelty)** | `scripts/train_hybrid.py`, `src/disinfo_detection/models_hybrid.py`, `src/disinfo_detection/metadata_features.py`, `src/disinfo_detection/datasets_hybrid.py` |
| Preprocessing | `scripts/preprocess.py`, `src/disinfo_detection/preprocessing.py` |
| Evaluation utilities | `src/disinfo_detection/evaluation.py` |
| EDA | `notebooks/01_eda_liar.ipynb` |
| Configs | `config/{dataset,baseline,transformer,hybrid}.yaml` |
| Tests | `tests/` (pytest) |
| Design notes | `docs/TRAINING_IMPROVEMENTS.md`, `docs/HYBRID_MODEL.md` |
| Agent/developer rules | `AGENTS.md` |

## Prerequisites

- Python **3.12** (pinned in `.python-version`)
- macOS on Apple Silicon is the primary development target; Linux with CUDA
  also works. Device selection is automatic: `cuda` → `mps` → `cpu`.
- A local virtual environment in `.venv/`.

## Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[ml,viz,dev]"
```

Or, equivalently:

```bash
make install
```

## End-to-end pipeline

```bash
make download         # LIAR train/valid/test.tsv into data/ (≈ 1 MB)
make preprocess       # data/processed/{train,valid,test}.pkl
make baseline         # TF-IDF + NB / SVM / RF
make transformer      # text-only RoBERTa fine-tuning
make hybrid           # text + metadata hybrid (thesis novelty)
make hybrid-textonly  # RQ2 ablation: hybrid code path with use_metadata=false
make test             # pytest, currently 16/16 green
```

All trained models land in `models/` (gitignored). Per-epoch metrics land in
`reports/transformer_logs/` and `reports/hybrid_logs/`. Final TEST metrics are
written as JSON next to the logs.

## Reported metric

Every number in the thesis and the paper is computed on the **TEST split
(1 283 examples)**, not the validation split. Primary metric is macro-F1;
secondary metrics are accuracy, per-class F1, and the confusion matrix.

## Project status

- ✅ Data loading, preprocessing, classical baselines, text-only RoBERTa,
  hybrid text + metadata model, 16 unit tests.
- ⏳ Full local training run on Apple Silicon to populate final numbers in the
  thesis report (MSRW 2 §5 currently has `[to be filled]` placeholders).
- ⏳ IEEE SIST 2026 paper draft alignment with final numbers.

See `AGENTS.md` for the coding conventions, dataset conventions, key design
decisions, and progress log.

## License and citation

Citation and license details will be added once the thesis manuscript and
final results are complete.
