# PROJECT_PLAN.md — Agent-Executable Roadmap
> Read `AGENTS.md` fully before starting any task here.
> Each task specifies exact files, function signatures, and acceptance criteria.
> After completing any task, append an entry to the Progress Log in `AGENTS.md`.

---

## Phase 0 — Current State

| File | Status |
|---|---|
| `notebooks/01_eda_liar.ipynb` | ✅ Complete |
| `pyproject.toml` | ✅ Local package metadata and dependency groups added |
| `config/dataset.yaml` | ✅ Implemented |
| `config/baseline.yml` | ✅ Filled for local baseline training |
| `config/transformer.yaml` | ✅ Filled for local transformer training |
| `src/disinfo_detection/data_loader.py` | ✅ Implemented |
| `scripts/download_data.py` | ✅ Implemented |
| Remaining `src/disinfo_detection/*.py` | ❌ Mostly empty |
| Remaining `scripts/*.py` | ❌ Mostly empty |
| `models_hybrid.py` | ❌ Does not exist yet |
| `evaluation.py` | ❌ Does not exist yet |

---

## Execution Mode — Local First

- Primary execution environment: local macOS machine with Apple Silicon
- Primary virtual environment: `.venv` in the repository root
- Preferred PyTorch device order: `cuda`, then `mps`, then `cpu`
- Colab/Kaggle are fallback environments only if a specific experiment exceeds local limits

---

## WEEK 1 — Environment + Data Pipeline

**Goal:** Working environment, config files filled, data loads cleanly into DataFrames.

---

### TASK 1.1 — Update `pyproject.toml`

**File:** `pyproject.toml`
**Action:** Replace existing content with the following:

```toml
[project]
name = "nlp-disinformation-detection-aitu"
version = "0.1.0"
description = "Hybrid NLP framework for disinformation detection on social networks"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "matplotlib>=3.10.7",
    "nltk>=3.9.2",
    "pandas>=2.3.3",
    "seaborn>=0.13.2",
    "numpy>=1.26.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
ml = [
    "scikit-learn>=1.5.0",
    "scipy>=1.13.0",
    "joblib>=1.4.0",
    "torch>=2.3.0",
    "transformers>=4.41.0",
    "datasets>=2.19.0",
    "tokenizers>=0.19.0",
]
viz = [
    "matplotlib>=3.10.7",
    "seaborn>=0.13.2",
]
dev = [
    "pytest>=8.0.0",
    "jupyter>=1.0.0",
    "ipykernel>=6.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/disinfo_detection"]
```

**Acceptance criteria:**
- `pip install -e ".[ml,viz,dev]"` runs with no errors
- `python -c "import torch; import transformers; import sklearn; print('OK')"` prints `OK`

---

### TASK 1.2 — Fill `config/dataset.yaml`

**File:** `config/dataset.yaml`

```yaml
liar:
  train_path: data/train.tsv
  valid_path: data/valid.tsv
  test_path: data/test.tsv
  processed_dir: data/processed

  columns:
    - id
    - label
    - statement
    - subject
    - speaker
    - job
    - state
    - party
    - barely_true_counts
    - false_counts
    - half_true_counts
    - mostly_true_counts
    - pants_on_fire_counts
    - context

  label_map:
    pants-fire: 0
    false: 1
    barely-true: 2
    half-true: 3
    mostly-true: 4
    true: 5

  label_names:
    - pants-fire
    - false
    - barely-true
    - half-true
    - mostly-true
    - true

  credibility_cols:
    - barely_true_counts
    - false_counts
    - half_true_counts
    - mostly_true_counts
    - pants_on_fire_counts

  metadata_cols:
    - speaker
    - job
    - state
    - party
```

**Acceptance criteria:**
- `yaml.safe_load(open("config/dataset.yaml"))` runs without error and returns a dict

---

### TASK 1.3 — Write `scripts/download_data.py`

**File:** `scripts/download_data.py`
**Action:** Script that downloads LIAR from HuggingFace and saves three TSV splits to `data/`.

**Requirements:**
- Use `datasets` library: `load_dataset("liar")`
- Save `train`, `validation`, `test` splits to `data/train.tsv`, `data/valid.tsv`, `data/test.tsv`
- Use the column order defined in `config/dataset.yaml`
- Use `logging`, not `print()`
- If files already exist, skip and log "Data already exists, skipping."
- Create `data/` directory if it does not exist

**Acceptance criteria:**
- `python scripts/download_data.py` creates three TSV files in `data/`
- Each file has 14 tab-separated columns with no header row
- Re-running skips the download and exits cleanly

---

### TASK 1.4 — Write `src/disinfo_detection/data_loader.py`

**File:** `src/disinfo_detection/data_loader.py`
**Action:** Implement the following functions with full docstrings.

```python
def load_config(config_path: str = "config/dataset.yaml") -> dict:
    """Load and return the dataset config as a dict."""


def load_liar(split: str, config_path: str = "config/dataset.yaml") -> pd.DataFrame:
    """Load a LIAR split into a DataFrame.

    Args:
        split: One of 'train', 'valid', 'test'.
        config_path: Path to dataset.yaml.

    Returns:
        DataFrame with all 14 LIAR columns plus 'label_id' (int, 0-5).

    Raises:
        FileNotFoundError: If split file does not exist.
        ValueError: If split is not 'train', 'valid', or 'test'.
    """


def get_label_map(config_path: str = "config/dataset.yaml") -> dict:
    """Return string-to-integer label mapping from config."""


def get_splits(config_path: str = "config/dataset.yaml") -> tuple:
    """Return (train_df, valid_df, test_df) tuple."""
```

**Implementation notes:**
- Load TSV with `pd.read_csv(path, sep="\t", header=None, quoting=3)`
- Add `label_id` column by mapping `label` column through `LABEL_MAP`
- Validate `split` argument before attempting file load

**Acceptance criteria:**
```python
df = load_liar("train")
assert df.shape[1] == 15
assert "label_id" in df.columns
assert df["label_id"].between(0, 5).all()
assert len(df) > 10000
```

---

## WEEK 2 — Preprocessing Pipeline

**Goal:** Two versions of cleaned text, credibility vectors, encoded metadata, saved to `data/processed/`.

---

### TASK 2.1 — Fill `config/baseline.yml`

```yaml
tfidf:
  max_features: 50000
  ngram_range: [1, 2]
  min_df: 2
  sublinear_tf: true

svm:
  kernel: linear
  C: 1.0
  max_iter: 5000
  class_weight: balanced

naive_bayes:
  alpha: 1.0

random_forest:
  n_estimators: 200
  max_depth: 20
  min_samples_leaf: 2
  class_weight: balanced
  n_jobs: -1
  random_state: 42
```

---

### TASK 2.2 — Write `src/disinfo_detection/preprocessing.py`

**File:** `src/disinfo_detection/preprocessing.py`
**Action:** Implement the following four functions with full docstrings.

```python
def clean_text_for_tfidf(text: str) -> str:
    """Clean text for TF-IDF: lowercase, remove URLs + HTML, lemmatize, remove stopwords."""


def clean_text_for_transformer(text: str) -> str:
    """Lightly clean text for RoBERTa: lowercase, remove URLs, normalize whitespace only.
    Do NOT lemmatize or remove stopwords — RoBERTa prefers natural text."""


def build_credibility_vector(row: pd.Series) -> list[float]:
    """Return normalized 5-dim credibility vector from speaker's historical claim counts.
    Order: [barely_true, false, half_true, mostly_true, pants_on_fire].
    If all counts are 0, return [0.2, 0.2, 0.2, 0.2, 0.2]."""


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing. Adds columns:
    statement_clean, statement_transformer, credibility_vector,
    credibility_0 through credibility_4."""
```

**Acceptance criteria:**
```python
df = preprocess_dataframe(load_liar("train"))
assert "statement_clean" in df.columns
assert "statement_transformer" in df.columns
assert len(df["credibility_vector"].iloc[0]) == 5
assert abs(sum(df["credibility_vector"].iloc[0]) - 1.0) < 0.01
```

---

### TASK 2.3 — Write `scripts/preprocess.py`

**File:** `scripts/preprocess.py`
**Action:** Load all three raw splits, apply `preprocess_dataframe`, save to `data/processed/` as pickle files.

**Requirements:**
- Save as `data/processed/train.pkl`, `valid.pkl`, `test.pkl` using `pd.DataFrame.to_pickle()`
- Log row counts and processing time per split
- Create `data/processed/` if it does not exist

**Acceptance criteria:**
- `python scripts/preprocess.py` runs without errors
- Three `.pkl` files exist in `data/processed/`
- Each loads with `pd.read_pickle()` and has all expected columns

---

## WEEK 3 — Baseline Models

**Goal:** Train SVM, Naive Bayes, Random Forest on TF-IDF features. Save to `models/`.

---

### TASK 3.1 — Write `src/disinfo_detection/models_baseline.py`

**File:** `src/disinfo_detection/models_baseline.py`
**Action:** Implement `TFIDFBaseline` class.

```python
class TFIDFBaseline:
    """sklearn Pipeline: TfidfVectorizer → classifier.

    Args:
        classifier_type: One of 'svm', 'naive_bayes', 'random_forest'.
        config_path: Path to baseline.yml.
    """

    def fit(self, X_train: list[str], y_train: list[int]) -> "TFIDFBaseline": ...
    def predict(self, X: list[str]) -> list[int]: ...
    def predict_proba(self, X: list[str]) -> np.ndarray: ...

    def get_top_features(self, n: int = 20) -> dict:
        """Random Forest only. Returns {class_name: [(word, importance), ...]}."""

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> "TFIDFBaseline": ...
```

**Acceptance criteria:**
```python
model = TFIDFBaseline("svm")
model.fit(train_texts, train_labels)
preds = model.predict(val_texts)
assert all(0 <= p <= 5 for p in preds)
model.save("models/baseline_svm.pkl")
assert Path("models/baseline_svm.pkl").exists()
```

---

### TASK 3.2 — Write `scripts/train_baseline.py`

**File:** `scripts/train_baseline.py`
**Requirements:**
- Load `data/processed/train.pkl` and `data/processed/valid.pkl`
- Train `TFIDFBaseline` for `svm`, `naive_bayes`, `random_forest`
- After each, evaluate on validation set and log Macro-F1
- Save each model to `models/baseline_{type}.pkl`
- Save summary to `reports/baseline_results.csv` with columns `[model, accuracy, macro_f1]`

---

## WEEK 4 — Evaluation Utilities & Baseline Analysis

**Goal:** Reusable evaluation module + all baseline figures for the paper.

---

### TASK 4.1 — Create `src/disinfo_detection/evaluation.py`

**File:** `src/disinfo_detection/evaluation.py` *(new file — create it)*

```python
def compute_metrics(y_true: list[int], y_pred: list[int],
                    label_names: list[str]) -> dict:
    """Returns: {accuracy, macro_f1, per_class_f1, classification_report}"""


def plot_confusion_matrix(y_true: list[int], y_pred: list[int],
                           label_names: list[str], title: str,
                           save_path: str | None = None) -> None:
    """Normalized seaborn heatmap. Figure size (8,6), colormap Blues, DPI 150."""


def compare_models(results: dict[str, dict],
                   save_path: str | None = None) -> pd.DataFrame:
    """Bar chart + DataFrame of model comparison sorted by macro_f1 descending."""
```

---

## WEEK 5 — Transformer Branch Setup

**Goal:** RoBERTa tokenizer working, Dataset class built, training loop skeleton ready.

---

### TASK 5.1 — Fill `config/transformer.yaml`

```yaml
model:
  name: roberta-base
  max_length: 128
  num_labels: 6
  hidden_dropout_prob: 0.1

training:
  batch_size: 16
  learning_rate: 2.0e-5
  epochs: 5
  warmup_ratio: 0.1
  weight_decay: 0.01
  gradient_clip: 1.0
  seed: 42

paths:
  output_dir: models/roberta_liar/
  logs_dir: reports/transformer_logs/
  best_checkpoint: models/roberta_liar/best_model.pt
```

---

### TASK 5.2 — Write `src/disinfo_detection/models_transformers.py`

**File:** `src/disinfo_detection/models_transformers.py`

```python
class LIARDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapping tokenized LIAR statements.

    Args:
        texts: List of statement_transformer strings.
        labels: List of integer label ids.
        tokenizer: HuggingFace tokenizer.
        max_length: Pad/truncate to this length.

    __getitem__ returns dict: {input_ids, attention_mask, label} — all tensors.
    """


class RoBERTaClassifier:
    """Fine-tunable RoBERTa-base for 6-class classification.

    Uses AutoModelForSequenceClassification from HuggingFace.
    """

    def train_epoch(self, dataloader, optimizer, scheduler, device) -> float:
        """One training pass. Returns average loss."""

    def evaluate(self, dataloader, device) -> dict:
        """Returns compute_metrics() dict on dataloader predictions."""

    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

**Acceptance criteria:**
```python
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
dataset = LIARDataset(["test statement"], [0], tokenizer, max_length=128)
batch = dataset[0]
assert batch["input_ids"].shape[0] == 128
assert "attention_mask" in batch
assert "label" in batch
```

---

### TASK 5.3 — Write `scripts/train_transformer.py`

**Requirements:**
- Load processed data, build `LIARDataset` for train + valid
- `DataLoader` with batch size from config
- Optimizer: `AdamW` from `transformers`
- Scheduler: `get_linear_schedule_with_warmup`
- Select device dynamically in this order: `cuda`, then `mps`, then `cpu`
- Log the resolved device at startup so local runs clearly report whether `mps` is active
- Log per epoch: train loss, val loss, val Macro-F1
- Save best checkpoint by val Macro-F1
- Save training log to `reports/transformer_logs/training_log.csv`

---

## WEEK 6 — Phased Fine-tuning

**Goal:** Implement the 3-phase training strategy from paper Section 4.1.

---

### TASK 6.1 — Refactor `scripts/train_transformer.py` for phased training

**Phase 1 (epochs 1-2):** All layers unfrozen, lr `2e-5`
**Phase 2 (epochs 3-4):** Freeze bottom 6 encoder layers, lr `1e-5`
```python
for layer in model.roberta.encoder.layer[:6]:
    for param in layer.parameters():
        param.requires_grad = False
```
**Phase 3 (epoch 5):** Unfreeze all, lr `5e-6`

**Acceptance criteria:**
- Training log CSV has columns `[epoch, phase, train_loss, val_loss, val_macro_f1]`
- Val Macro-F1 in Phase 3 ≥ Phase 1

---

## WEEK 7 — Context Branch

**Goal:** Speaker credibility + metadata encoding branch implemented and tested standalone.

---

### TASK 7.1 — Add encoder helpers to `preprocessing.py`

```python
def fit_label_encoders(df: pd.DataFrame, cols: list[str]) -> dict:
    """Fit LabelEncoder per categorical column. Include 'unknown' category."""


def encode_metadata(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Add speaker_id, job_id, state_id, party_id integer columns."""
```

---

### TASK 7.2 — Create `src/disinfo_detection/models_hybrid.py`

**File:** `src/disinfo_detection/models_hybrid.py` *(new file — create it)*

```python
class ContextBranch(nn.Module):
    """Encodes speaker metadata + credibility history into 32-dim vector.

    Architecture:
    speaker_embedding(n_speakers, 32) + party_embedding(n_parties, 8)
    → concat with credibility_vec (5-dim)
    → Linear(45, 64) → ReLU → Dropout(0.3) → Linear(64, 32)
    """

    def forward(self, speaker_id, party_id, credibility_vec) -> torch.Tensor:
        """Returns (batch_size, 32) tensor."""


class HybridDisinfoClassifier(nn.Module):
    """Dual-branch hybrid: RoBERTa (768) + ContextBranch (32) → fused MLP → 6 classes.

    Fusion MLP: Linear(800,256) → ReLU → Dropout(0.4)
                → Linear(256,64) → ReLU → Dropout(0.2)
                → Linear(64, num_labels)
    """

    def forward(self, input_ids, attention_mask,
                speaker_id, party_id, credibility_vec) -> torch.Tensor:
        """Returns logits of shape (batch_size, num_labels)."""
```

**Acceptance criteria:**
```python
ctx = ContextBranch(n_speakers=3500, n_parties=10)
assert ctx(torch.tensor([0,1]), torch.tensor([0,1]), torch.randn(2,5)).shape == (2, 32)

hybrid = HybridDisinfoClassifier(roberta_model, ctx)
logits = hybrid(torch.randint(0,1000,(2,128)), torch.ones(2,128,dtype=torch.long),
                torch.tensor([0,1]), torch.tensor([0,1]), torch.randn(2,5))
assert logits.shape == (2, 6)
```

---

## WEEK 8 — Hybrid Training Loop

**Goal:** Full hybrid model trains end-to-end with phased strategy and class balancing.

---

### TASK 8.1 — Add `LIARHybridDataset` to `models_hybrid.py`

```python
class LIARHybridDataset(torch.utils.data.Dataset):
    """Combined text + context dataset for hybrid model.

    Each item returns: input_ids, attention_mask, speaker_id, party_id,
                       credibility_vec, label — all tensors.
    """
```

---

### TASK 8.2 — Add `train_hybrid_model()` to `models_hybrid.py`

```python
def train_hybrid_model(model, train_loader, val_loader, device,
                       config_path="config/transformer.yaml",
                       save_dir="models/hybrid_model/") -> pd.DataFrame:
    """3-phase training for HybridDisinfoClassifier.

    Phase 1 (ep 1-2): Freeze RoBERTa, train context + fusion only
    Phase 2 (ep 3-4): Unfreeze top 4 RoBERTa layers, lr=1e-5
    Phase 3 (ep 5):   Unfreeze all, lr=5e-6, WeightedRandomSampler active

    Returns training log DataFrame: [epoch, phase, train_loss, val_loss, val_macro_f1]
    """
```

---

## WEEK 9 — Full Evaluation & Ablation Study

**Goal:** All final numbers, ablation study, all paper figures generated.

---

### TASK 9.1 — Ablation study in `notebooks/06_evaluation_results.ipynb`

Run and record all 4 configurations:

| Config | Setup | Expected Macro-F1 |
|---|---|---|
| A | Context branch only (no text) | ~50-55% |
| B | RoBERTa only (no context) | ~58-65% |
| C | RoBERTa + credibility vector, no metadata embeddings | ~63-67% |
| D | Full hybrid (RoBERTa + credibility + metadata) | ~68-72% |

---

### TASK 9.2 — Generate all paper figures to `reports/figures/`

| Filename | Content |
|---|---|
| `fig1_label_distribution.png` | Label class bar chart |
| `fig2_credibility_by_class.png` | Credibility vector heatmap per truth class |
| `fig3_tfidf_features.png` | Random Forest top-20 feature importances |
| `fig4_baseline_confusion_svm.png` | SVM confusion matrix |
| `fig5_transformer_training_curves.png` | Loss + F1 per epoch (all 3 phases marked) |
| `fig6_ablation_results.png` | Bar chart comparing configs A, B, C, D |
| `fig7_hybrid_confusion.png` | Final hybrid model confusion matrix |
| `fig8_model_comparison.png` | All 6 models Macro-F1 bar chart |

**All figures must:** be PNG, 150 DPI, minimum 12pt font, include title and axis labels, use `seaborn colorblind` palette.

---

### TASK 9.3 — Save `reports/final_results.csv`

Columns: `[model, macro_f1, accuracy, pants_fire_f1, false_f1, barely_true_f1, half_true_f1, mostly_true_f1, true_f1]`
Rows: `naive_bayes`, `svm`, `random_forest`, `roberta_only`, `context_only`, `hybrid_full`

---

## WEEK 10 — Cleanup & Documentation

---

### TASK 10.1 — Update `README.md`

Required sections:
1. Project description (2-3 sentences)
2. Prerequisites (Python version, local Apple Silicon / MPS recommendation)
3. Installation instructions
4. Quick start (local-first pipeline)
5. Project structure
6. Results table (copied from `final_results.csv`)
7. Citation placeholder

---

### TASK 10.2 — Update `Makefile`

```makefile
.PHONY: download preprocess baseline transformer test clean

download:
	python scripts/download_data.py

preprocess:
	python scripts/preprocess.py

baseline:
	python scripts/train_baseline.py

transformer:
	python scripts/train_transformer.py

train-all: download preprocess baseline transformer

test:
	pytest tests/ -v

clean:
	rm -rf data/processed/ models/ reports/figures/*.png
```

---

### TASK 10.3 — End-to-end validation

Run `make train-all` locally on a clean environment (no existing `data/` or `models/`).

**Acceptance criteria:**
- Runs without errors
- `reports/final_results.csv` exists with 6 rows
- At least 5 PNG figures in `reports/figures/`
- Hybrid Macro-F1 exceeds RoBERTa-only by ≥ 3 percentage points

---

## Appendix — Expected Performance Benchmarks

| Model | Expected Macro-F1 |
|---|---|
| Naive Bayes | ~38% |
| SVM | ~44% |
| Random Forest | ~42% |
| RoBERTa (text only) | ~62% |
| Context only | ~52% |
| **Hybrid (full)** | **~68-72%** |

*Estimates from literature. Actual results may vary ±3-5%.*
