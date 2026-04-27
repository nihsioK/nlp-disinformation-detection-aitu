.PHONY: install download smoke preprocess baseline transformer hybrid hybrid-textonly hybrid-leaky train-all test clean

VENV ?= .venv
PYTHON_BIN ?= python3.12
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

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

clean:
	rm -rf data/processed/ models/ reports/figures/*.png reports/transformer_logs/
