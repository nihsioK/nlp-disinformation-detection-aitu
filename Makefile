.PHONY: install download smoke preprocess baseline transformer train-all test clean

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

train-all: download preprocess baseline transformer

test: $(PYTHON)
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf data/processed/ models/ reports/figures/*.png reports/transformer_logs/
