# Hybrid text + metadata model

This document describes the hybrid classifier added in `feat/hybrid-model` —
the primary novel contribution of the thesis (МСРВ 2 / ИПМ, Section 4).

## Motivation

Text-only RoBERTa on the LIAR dataset plateaus around macro-F1 ≈ 0.27–0.30
(Wang 2017; subsequent work including Alhindi 2018, Kirilin & Strube 2019).
The per-statement text is very short (~17 tokens) and often devoid of
features distinguishing "half-true" from "mostly-true" without external
signal. The LIAR dataset, however, ships metadata for every statement —
speaker, party, job, state, subject, context, and prior-statement history —
which prior work has shown can lift macro-F1 by 2–6 points when fused
correctly.

This module operationalizes that fusion end-to-end with an ablation-ready
design so the thesis can report fair `text-only` vs. `text + metadata`
numbers out of the same code path.

## Architecture

```
statement     -> RoBERTa encoder -> [CLS] (768d)  ─┐
                                                    │
credibility_vector (5d)                             │
cred_total / cred_log_total / cred_pants_share /    │
  cred_false_share (4d normalized)                  ├─> concat -> MLP (128d, GELU, dropout) -> softmax (6)
hashed(speaker, party, job, state, subject,         │
  context) -> shared Embedding table -> (96d)       │
  (6 fields × 16d)                                  │
metadata branch Linear(109 -> 64, GELU, dropout) ───┘
```

- **Feature hashing** for categorical fields (bucket size 256 per field,
  deterministic blake2b + fixed salt) — avoids carrying any vocab, handles
  unseen speakers at test time gracefully, and matches the bucket layout used
  in the original LIAR paper's release-ready pipeline.
- **Scalar normalization** via hard-coded divisors derived from LIAR train
  statistics (no learned scaler to persist).
- **Shared embedding table with per-field offsets** keeps parameters low
  (256×6 buckets × 16d ≈ 25k params for the categorical part).
- **Two-LR optimizer**: encoder at 1e-5 (same as text-only), head & metadata
  branch at 5e-4 since they're trained from scratch.
- **`use_metadata: false`** flips the identical module into the text-only
  ablation for RQ2.

## How to run

```bash
# Full hybrid
make hybrid

# Text-only ablation (for apples-to-apples RQ2 comparison)
make hybrid-textonly

# Or directly with a custom config:
HYBRID_CONFIG=config/my_variant.yaml python scripts/train_hybrid.py
```

Outputs:

- `reports/hybrid_logs/training_log.csv` — per-epoch metrics
- `reports/hybrid_logs/hybrid_run_history.csv` — appended across runs
- `reports/hybrid_logs/hybrid_test_metrics.json` — TEST-split metrics
- `models/hybrid_liar/best_model.pt` — best-checkpoint weights

## Design decisions worth noting in the thesis

1. **Why raw `[CLS]`, not `pooler_output`?** RoBERTa's pooler adds a
   randomly-initialized tanh layer that underperforms on small-dataset
   classification. Using `last_hidden_state[:, 0]` is the standard fix.
2. **Why inverse-sqrt frequency weights?** Full inverse-frequency on LIAR
   pushes the model to over-predict `pants-fire` (rarest class). Inverse-sqrt
   is the common compromise used in the text-classification literature.
3. **Why label smoothing 0.05?** LIAR labels are ordinal (pants-fire < false
   < barely-true < … < true). Smoothing softens the hard one-hot target and
   empirically improves macro-F1 by 0.5–1.0 points.
4. **Why a tiny fusion MLP?** On LIAR the text encoder carries most of the
   capacity; a wide fusion head just overfits the ~10k training examples.
5. **Why feature hashing over learned vocab?** LIAR has thousands of rare
   speakers (~3k unique) and an even longer tail of subject combinations.
   Hashing removes the need to serialize a vocab while keeping collision
   rates low at bucket size 256 per field.

## Tests

`tests/test_hybrid.py` covers:

- Hashing determinism & bucket bounds
- Dense/categorical matrix shapes and dtypes
- End-to-end forward+backward on a 4-row toy batch with a
  tiny-init RoBERTa config (runs in <2s, no downloads)
- Ablation mode (`use_metadata=False`) produces correct shapes and no
  metadata branch parameters

Run: `make test`
