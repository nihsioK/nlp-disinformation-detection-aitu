# Running the GPU training notebook on Kaggle

This guide documents the clean end-to-end remote run. The notebook trains all
reported systems, runs the paper post-processing scripts, and creates two
downloadable archives: a required results archive and an optional checkpoint
archive. It does not push back to GitHub and does not generate the old
diagnostic figure gallery.

## Notebook

Upload `notebooks/training.ipynb` to Kaggle and run it with:

- Accelerator: `GPU T4 x2` preferred.
- Internet: On.
- Persistence: not required.
- Branch: set in the first code cell, default `main`.

If Kaggle assigns a P100 and the GPU check reports unsupported CUDA kernels,
either switch to T4 or enable the optional PyTorch reinstall cell in the
notebook, restart the kernel, and continue from the clone step.

## What It Runs

The default notebook run executes:

- `scripts/download_data.py`
- `scripts/preprocess.py`
- `scripts/train_baseline.py`
- `scripts/train_transformer.py`
- `scripts/train_hybrid.py` for corrected, text-only, and leaky variants
- `scripts/verify_leakage.py`
- `scripts/bootstrap_ci.py`
- `scripts/plot_per_class_f1.py`
- `scripts/plot_confusion_matrices.py`
- `scripts/plot_training_curves.py`

The canonical seeds are `42`, `1337`, and `2024`.

## Output Archives

The required artifact is `/kaggle/working/disinformation_results.zip`. Download
it from the Kaggle Output panel.

If `CREATE_MODELS_ZIP = True`, the notebook also writes
`/kaggle/working/models.zip`. Keep this separate. It is useful for checkpoint
archival, but it is not needed for paper tables, figures, or Overleaf.

`disinformation_results.zip` intentionally contains only paper/reproducibility
artifacts:

- aggregate summaries: `reports/results_summary.json`, `reports/multi_seed_summary.json`
- bootstrap/leakage reports: `reports/bootstrap_ci*`, `reports/leakage_verification*`
- seed-scoped metric JSON files under `reports/seed_<seed>/`
- seed-scoped TEST prediction JSONL files under `reports/seed_<seed>/predictions/`
- final paper figures under `figures/`
- primary-seed training logs needed for the training-curves figure

It intentionally excludes model checkpoints, broad diagnostic figure galleries,
duplicate top-level prediction copies, raw run-history CSVs, caches, and virtual
environments.

## Local Reproducibility Workflow

After downloading the Kaggle results archive, copy `disinformation_results.zip`
to the repository root and run:

```bash
make import-results
make figures
make package-overleaf
```

`make import-results` safely extracts only `reports/` and `figures/` artifacts
from the archive. `make figures` regenerates deterministic final figures from
the imported result files when possible. `make package-overleaf` creates
`dist/overleaf_submission.zip` containing only `main.tex`, `references.bib`, and
the final figure PDFs needed by Overleaf.

If your downloaded archive is elsewhere, pass it explicitly:

```bash
make import-results RESULTS_ZIP=/path/to/disinformation_results.zip
```

## Regenerating The Notebook

Do not hand-edit `notebooks/training.ipynb`. It is generated from:

```bash
python notebooks/_build_training_notebook.py
```

Edit the generator, run it, and commit both files.
