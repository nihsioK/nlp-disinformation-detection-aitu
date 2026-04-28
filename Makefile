.PHONY: install download smoke preprocess baseline transformer hybrid hybrid-textonly hybrid-leaky train-all test verify-leakage bootstrap-ci figures package-results package-results-with-models import-results package-overleaf paper clean

VENV ?= .venv
PYTHON_BIN ?= python3.12
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
RESULTS_ZIP ?= disinformation_results.zip
MODELS_ZIP ?= models.zip
OVERLEAF_ZIP ?= dist/overleaf_submission.zip

$(PYTHON):
	$(PYTHON_BIN) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: $(PYTHON)
	$(PIP) install -e ".[ml,viz,dev]"

download: $(PYTHON)
	$(PYTHON) scripts/download_data.py

smoke: $(PYTHON)
	$(PYTHON) -c "from src.disinfo_detection.data_loader import load_liar; df = load_liar('train'); print(df.shape)"

preprocess: $(PYTHON)
	$(PYTHON) scripts/preprocess.py

baseline: $(PYTHON)
	$(PYTHON) scripts/train_baseline.py

transformer: $(PYTHON)
	$(PYTHON) scripts/train_transformer.py

hybrid: $(PYTHON)
	$(PYTHON) scripts/train_hybrid.py

# Text-only ablation that reuses the hybrid code path (for RQ2).
# Flips use_metadata off via a one-line sed into a temp config.
hybrid-textonly: $(PYTHON)
	sed 's/use_metadata: true/use_metadata: false/' config/hybrid.yaml > config/hybrid_textonly.yaml
	HYBRID_CONFIG=config/hybrid_textonly.yaml $(PYTHON) scripts/train_hybrid.py

# Leaky comparison: same hybrid pipeline but with credibility counts that still
# include the row's own verdict, matching prior LIAR work (Wang 2017, Alhindi
# 2018, Kirilin & Strube 2019). Keep this run for reporting alongside the
# leakage-corrected default to disclose the size of the leakage gap.
hybrid-leaky: $(PYTHON)
	sed 's/leakage_corrected: true/leakage_corrected: false/' config/hybrid.yaml > config/hybrid_leaky.yaml
	HYBRID_CONFIG=config/hybrid_leaky.yaml $(PYTHON) scripts/train_hybrid.py

train-all: download preprocess baseline transformer hybrid

test: $(PYTHON)
	$(PYTHON) -m pytest tests/ -v

# Empirical leakage verification (counting + predictions). Writes
# reports/leakage_verification{,_predictions}.json and prints a Markdown
# summary block ready to paste into the paper.
verify-leakage: $(PYTHON)
	PYTHONPATH=. $(PYTHON) scripts/verify_leakage.py

bootstrap-ci: $(PYTHON)
	PYTHONPATH=. $(PYTHON) scripts/bootstrap_ci.py

figures: $(PYTHON)
	PYTHONPATH=. $(PYTHON) scripts/plot_per_class_f1.py
	PYTHONPATH=. $(PYTHON) scripts/plot_confusion_matrices.py
	PYTHONPATH=. $(PYTHON) scripts/plot_training_curves.py

package-results: $(PYTHON)
	PYTHONPATH=. $(PYTHON) scripts/package_artifacts.py --output $(RESULTS_ZIP)

package-results-with-models: $(PYTHON)
	PYTHONPATH=. $(PYTHON) scripts/package_artifacts.py --output $(RESULTS_ZIP) --include-models --models-output $(MODELS_ZIP)

import-results: $(PYTHON)
	PYTHONPATH=. $(PYTHON) scripts/import_results_archive.py $(RESULTS_ZIP)

package-overleaf: $(PYTHON)
	PYTHONPATH=. $(PYTHON) scripts/package_overleaf.py --output $(OVERLEAF_ZIP)

paper:
	tectonic main.tex

clean:
	rm -rf data/processed/ models/ reports/figures_all/ reports/paper/ reports/task1_task3/ paper_figures/
	rm -rf .pytest_cache/ notebooks/__pycache__/ scripts/__pycache__/ src/__pycache__/ src/disinfo_detection/__pycache__/ tests/__pycache__/
	rm -f .DS_Store reports/.DS_Store src/.DS_Store reports/archive_manifest.json disinformation_results.zip models.zip
	rm -f main.aux main.bbl main.bcf main.blg main.fdb_latexmk main.fls main.log main.out main.run.xml main.synctex.gz main.toc main.xdv
	rm -f paper.aux paper.bbl paper.bcf paper.blg paper.fdb_latexmk paper.fls paper.log paper.out paper.run.xml paper.synctex.gz paper.toc paper.xdv
