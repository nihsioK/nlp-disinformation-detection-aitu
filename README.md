# nlp-disinformation-detection-aitu

Hybrid NLP framework for detecting disinformation in political statements from
the LIAR dataset. The project combines a fine-tuned `roberta-base` text encoder
with a metadata branch that encodes speaker credibility history and hashed
categorical context features.

This repository is the code base for Daniyar Koishin's master's thesis at
Astana IT University (program 7M06105, 2025–2027). The planned outputs are an
IEEE SIST 2026 conference paper and a Scopus journal article, followed by the
thesis defense in June 2027.

## Research questions

The thesis and the paper are organized around three research questions. The
codebase is structured so that each RQ maps to a single `make` target.

- **RQ1 — Are classical TF-IDF baselines competitive with transformer
  text-only models on LIAR six-class?** Answered by comparing
  `make baseline` (Naive Bayes / SVM / Random Forest) against
  `make transformer` (text-only RoBERTa-base) on the TEST-split macro-F1.
- **RQ2 — Does fusing LIAR metadata (speaker credibility counts, hashed
  categorical context fields) with the RoBERTa encoder improve macro-F1 over
  a text-only transformer of identical capacity?** Answered by comparing
  `make hybrid` (text + metadata) against `make hybrid-textonly`, which runs
  the *same* hybrid code path with `use_metadata: false` so the only
  difference is the metadata branch.
- **RQ3 — Which metadata signals carry the lift: the historical credibility
  counts, or the categorical context fields (speaker / party / job / state /
  subject / context)?** Addressed by toggling `metadata.categorical_fields`
  in `config/hybrid.yaml`; see `docs/HYBRID_MODEL.md` for the planned
  per-field ablations.

## Dataset

All experiments use the **LIAR** dataset (Wang, 2017):

- 12 836 short political statements collected from PolitiFact.
- Six-class ordinal truthfulness label: `pants-fire < false < barely-true <
  half-true < mostly-true < true`.
- Ships with per-speaker metadata: `speaker`, `party`, `job`, `state`,
  `subject`, `context`, and five historical credibility counts
  (`barely_true_counts`, `false_counts`, `half_true_counts`,
  `mostly_true_counts`, `pants_on_fire_counts`).
- Train / valid / test splits: 10 269 / 1 284 / 1 283 statements. The test
  split is the canonical reporting split used by Wang (2017) and
  follow-ups and is the only split quoted in the thesis and the paper
  (see "Reported metric" below).

See `docs/HYBRID_MODEL.md` for a note on LIAR's credibility-count convention
and why the observed metadata-driven macro-F1 lift is consistent with prior
published numbers (Alhindi 2018; Kirilin & Strube 2019).

## What is in the repo

| Area | Location |
|---|---|
| Classical baselines (TF-IDF + NB / SVM / RF) | `scripts/train_baseline.py`, `src/disinfo_detection/models_baseline.py` |
| Text-only RoBERTa | `scripts/train_transformer.py`, `src/disinfo_detection/models_transformers.py` |
| **Hybrid text + metadata model (thesis novelty)** | `scripts/train_hybrid.py`, `src/disinfo_detection/models_hybrid.py`, `src/disinfo_detection/metadata_features.py`, `src/disinfo_detection/datasets_hybrid.py` |
| Preprocessing | `scripts/preprocess.py`, `src/disinfo_detection/preprocessing.py` |
| Evaluation utilities | `src/disinfo_detection/evaluation.py` |
| EDA | `notebooks/01_eda_liar.ipynb` |
| Configs | `config/dataset.yaml`, `config/baseline.yml`, `config/transformer.yaml`, `config/hybrid.yaml` |
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

## License

Code in this repository is released under the [MIT License](LICENSE).
The thesis manuscript, figures, and written content remain under the author's
copyright and are not covered by the MIT License.

## Citation

If you use this code, please cite:

```bibtex
@mastersthesis{koishin2027disinfo,
  author  = {Koishin, Daniyar},
  title   = {Hybrid NLP Framework for Disinformation Detection in Political Statements},
  school  = {Astana IT University},
  year    = {2027},
  address = {Astana, Kazakhstan},
  note    = {Program 7M06105}
}
```

The BibTeX entry will be updated with final title, page numbers, and the
associated IEEE SIST 2026 paper reference once the manuscript is finalized.
