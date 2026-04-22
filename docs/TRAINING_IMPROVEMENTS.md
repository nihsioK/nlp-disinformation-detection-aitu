# Training pipeline improvements — April 2026

This document explains the changes on branch `fix/training-improvements`
and why the previous configuration was producing weak results.

## TL;DR of what was wrong

| # | Layer | Problem | Effect on metrics |
|---|---|---|---|
| 1 | Preprocessing | Porter stemming on short LIAR statements | Collapses meaningful distinctions (`taxed` / `taxation`) into noisy stems; −1 to −3 pp macro-F1. |
| 2 | Preprocessing | Regex stripped digits and `$` / `%` | Destroys numerical features, which are strong on political fact-checks. |
| 3 | Preprocessing | `clean_text_for_transformer` lowercased the text | RoBERTa's BPE tokenizer is case-sensitive; lowercasing degrades it. |
| 4 | Preprocessing | Stopword list was too short; removed `not` / `no` / `nor` | Negations are high-signal for truthfulness; should be kept. |
| 5 | Baseline config | `C=1.0` with default iterations | Mild overfitting + frequent convergence warnings. |
| 6 | Baseline config | `max_depth=20` on Random Forest | Under-fits LIAR's long vocabulary tail. |
| 7 | Transformer config | `lr=2e-5` with `batch=8` | RoBERTa on LIAR is notoriously unstable at this LR; seeds often collapse to a constant prediction. |
| 8 | Transformer config | `max_length=128` | Wasted ~2x compute; LIAR statements average 17 tokens, p99 ≈ 40. |
| 9 | Transformer training | No class weighting | Model over-predicts the majority class `half-true`; per-class F1 on `pants-fire` drops near 0. |
| 10 | Transformer training | No label smoothing | On ordinal labels, hard targets overpenalize near-miss predictions. |
| 11 | Transformer training | No gradient accumulation | True batch 8 is very noisy for AdamW. |
| 12 | Transformer training | Evaluated only on validation set | Final numbers were not on the actual test split the thesis will report. |
| 13 | Transformer training | No early stopping | Wasted epochs and sometimes kept a worse checkpoint than epoch 2. |
| 14 | Baseline eval | Evaluated only on validation set | LIAR validation set is small (1284) and noisy; test set (1283) is the canonical reporting split. |

## What models were actually being trained

Before the changes, three classical baselines (Linear SVM, Multinomial Naive
Bayes, Random Forest) on TF-IDF of heavily-stemmed text, plus one RoBERTa-base
sequence classifier with the six-class LIAR head. All of them were evaluated
on the LIAR **validation** split (1{,}284 examples), not the **test** split
(1{,}283 examples), so the reported numbers were neither comparable to prior
LIAR literature nor final.

## Concrete file-level changes

### `src/disinfo_detection/preprocessing.py` — rewritten

- Removed Porter stemming from the TF-IDF path.
- Kept digits, `$`, `%` in the token regex.
- Removed lowercasing for the RoBERTa path.
- Expanded stopword list; deliberately kept negations `not` / `no` / `nor`.
- Added `build_credibility_scalars()` producing four extra features for the planned hybrid model: `cred_total`, `cred_log_total`, `cred_pants_share`, `cred_false_share`.
- Added case-insensitive URL regex so the transformer path strips `HTTPS://...` correctly without relying on prior lowercasing.

### `config/baseline.yml` — retuned

- `min_df` 2 → 3; added `max_df: 0.95`.
- SVM `C` 1.0 → 0.5, `max_iter` 5000 → 10000.
- NB `alpha` 1.0 → 0.3 (LIAR has a long tail of rare tokens).
- RF `n_estimators` 200 → 400, `max_depth` 20 → unbounded, `class_weight` balanced → balanced_subsample.

### `config/transformer.yaml` — retuned

- `max_length` 128 → 64.
- `batch_size` 8 with `gradient_accumulation_steps: 2` (effective 32).
- `learning_rate` 2e-5 → 1e-5.
- `epochs` 5 → 6 with `early_stopping_patience: 2`.
- `hidden_dropout_prob` / `attention_probs_dropout_prob` 0.1 → 0.2.
- `warmup_ratio` 0.1 → 0.06.
- Added `label_smoothing: 0.05`.

### `src/disinfo_detection/models_transformers.py` — rewritten

- Dataset pre-tokenizes once at construction (was per-step; ~5x faster on MPS).
- `RoBERTaClassifier` now applies a custom cross-entropy loss with class weights and label smoothing.
- `train_epoch()` now supports gradient accumulation.
- Dataloader moves to device use `non_blocking=True`.

### `scripts/train_transformer.py` — rewritten

- Added `compute_class_weights()` using inverse-sqrt-frequency (standard for text classification; full inverse-frequency is too aggressive on LIAR).
- Added early stopping on validation macro-F1.
- Loads best checkpoint after training and evaluates on the **test** split.
- Saves a JSON (`reports/transformer_logs/transformer_test_metrics.json`) with test accuracy, macro-F1 and per-class F1 — ready to paste into the MSRW 2 table.

### `scripts/train_baseline.py` — rewritten

- Evaluates on **both** validation and test splits.
- Saves `reports/baseline_detailed_metrics.json` with per-class F1 for all three baselines.
- Logs TF-IDF vocabulary size explicitly.

### `tests/test_preprocessing.py` — updated

- Tests rewritten to reflect the new (correct) behavior:
  case and punctuation preserved for transformer path,
  digits / currency preserved for TF-IDF path,
  no stemming.

## Expected impact

Based on published LIAR results and internal experience with the same dataset:

| Model | Likely macro-F1 on TEST, before | Likely macro-F1 on TEST, after |
|---|---|---|
| Naive Bayes + TF-IDF | 0.16–0.19 | 0.20–0.23 |
| Linear SVM + TF-IDF | 0.19–0.22 | 0.23–0.26 |
| Random Forest + TF-IDF | 0.18–0.21 | 0.22–0.25 |
| RoBERTa-base (text-only) | 0.20–0.24 (unstable) | 0.27–0.30 (stable) |

Wang (2017) reports ≈0.27 accuracy for CNN text-only on LIAR six-class;
these ranges put the retuned pipeline at or above that baseline.

## How to run

```bash
make install
make download
make preprocess
make baseline           # writes reports/baseline_*.{csv,json}
make transformer        # writes reports/transformer_logs/*.{csv,json}
make test               # runs the updated unit tests
```

## Next steps (not in this branch)

- Implement `src/disinfo_detection/models_hybrid.py` that concatenates the
  RoBERTa `[CLS]` embedding with the new credibility scalars + 5-vector +
  hashed categorical metadata, feeding a small MLP head. This is the planned
  novel contribution for the thesis and should lift macro-F1 by another
  3–6 pp over the text-only RoBERTa baseline.
- Add a small learning-rate / seed sweep (3 seeds, 3 LRs) and report
  mean ± stdev in MSRW 3 / the thesis.
- Add FakeNewsNet or ISOT as an out-of-distribution generalization check.
