# nlp-disinformation-detection-aitu

Hybrid NLP framework for detecting disinformation in political statements from the LIAR dataset.
The project combines a transformer text branch based on `roberta-base` with a metadata branch that
encodes speaker credibility history and categorical context features.

The repository is now organized for local-first development on macOS with Apple Silicon. Cloud
notebooks such as Colab or Kaggle are optional fallback environments, not the primary execution
path.

## Prerequisites

- Python `3.12` as specified in `.python-version`
- macOS on Apple Silicon is the primary target environment
- A local virtual environment in `.venv`
- For transformer training, PyTorch should use `mps` when available; `cpu` remains the fallback

## Installation

Create and populate a local virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[ml,viz,dev]"
```

If you prefer `make`, the same setup is available through:

```bash
make install
```

## Quick Start

The repository is still partially scaffolded, so only the data acquisition and loader path are
fully implemented today. The local workflow is:

```bash
make install
make download
make smoke
```

As the remaining modules are implemented, the intended local end-to-end pipeline is:

```bash
make download
make preprocess
make baseline
make transformer
make test
```

## Local Execution Notes

- Prefer running everything from the project root with the local `.venv`
- On Apple Silicon, transformer code should select devices in this order: `cuda`, then `mps`,
  then `cpu`
- Start conservatively with batch sizes that fit local memory; increase only after verifying
  stable runs
- Colab/Kaggle should only be used if local runtime or memory proves insufficient for a specific
  experiment

## Project Structure

```text
nlp-disinformation-detection-aitu/
├── config/                    # YAML configs for data, baseline, and transformer runs
├── notebooks/                 # EDA and experiment notebooks
├── reports/                   # Figures, logs, and tabular outputs
├── scripts/                   # Runnable CLI entry points
├── src/disinfo_detection/     # Core package code
├── tests/                     # Pytest-based validation
├── AGENTS.md                  # Agent rules and project decisions
└── PROJECT_PLAN.md            # Week-by-week implementation roadmap
```

## Current Status

- Implemented: `scripts/download_data.py`, `config/dataset.yaml`,
  `src/disinfo_detection/data_loader.py`
- Prepared for local-first usage: `README.md`, `Makefile`, local config scaffolding, and
  Apple Silicon workflow guidance
- Not yet implemented: preprocessing, baseline training, transformer training, hybrid model,
  evaluation utilities, and automated tests

## Results

Final model results are not available yet because the training and evaluation stages are still in
progress. This section will be updated once `reports/final_results.csv` exists.

## Citation

Citation details will be added after the thesis manuscript and final results are complete.
