# Running the training pipeline on Kaggle

This guide walks you end-to-end through running every training stage
(`baseline`, `transformer`, `hybrid`, `hybrid-textonly`) on a free Kaggle
GPU and pushing the resulting metric JSON files back into this
repository. The whole pipeline takes **~60–90 minutes** on a P100.

Use this instead of training locally when you want to keep your
laptop/desktop free or when you do not have a CUDA GPU.

> TL;DR: open `notebooks/kaggle_training.ipynb` on Kaggle, enable GPU +
> Internet, add a `GH_PAT` secret, hit **Run all**.

---

## 1. Prerequisites (one-time)

### 1.1 Kaggle account

1. Create a free account at <https://www.kaggle.com/>.
2. Go to **Settings** → **Phone verification** and verify your phone.
   Without this, Kaggle will not let you enable GPU or Internet in
   notebooks.

### 1.2 GitHub Personal Access Token (PAT)

The notebook pushes JSON metrics to `main` after training so the paper
run is captured in git. That push needs a token.

1. Visit <https://github.com/settings/tokens>.
2. **Generate new token (classic)**.
3. **Note**: `kaggle-training-push`.
4. **Expiration**: 30 days is plenty.
5. **Scope**: only `repo` (full control of private repositories).
6. Click **Generate token** and **copy it once** — GitHub will never
   show it again.

### 1.3 Kaggle secret

1. On Kaggle, open any notebook and click **Add-ons** → **Secrets**.
2. **Add a new secret**.
   - Label: `GH_PAT`
   - Value: paste the token from step 1.2.
3. Enable the secret (toggle on) for the training notebook.

The notebook reads this via `UserSecretsClient().get_secret("GH_PAT")`
and never prints the value.

---

## 2. Create the training notebook

1. On Kaggle, click **Create** → **New Notebook**.
2. In the top bar: **File** → **Import notebook** and upload
   `notebooks/kaggle_training.ipynb` from this repo.
3. Open the right sidebar and configure the session:
   - **Accelerator**: **`GPU T4 x2`** (recommended). **Do NOT pick `GPU P100`** — Kaggle's
     current PyTorch build does not ship CUDA kernels for P100's compute capability (`sm_60`),
     so transformer training crashes with `cudaErrorNoKernelImageForDevice`. T4 (`sm_75`)
     works out of the box. If Kaggle only gives you a P100, see §7 "P100 GPU".
   - **Persistence**: `No persistence` is fine; we re-clone each run.
   - **Environment**: `Always use latest environment`.
   - **Internet**: **ON** (required — we clone GitHub and `pip install`).
4. Under **Add-ons** → **Secrets** make sure `GH_PAT` is attached.

Kaggle quota reminder: you get ~30 GPU hours/week for free. A full run
is well under 2 hours, so you can do many iterations.

---

## 3. Configure the run (first code cell)

The very first cell of the notebook defines which stages to run. The
defaults are sensible, but open it and confirm:

```python
GITHUB_USER = "nihsioK"
REPO_NAME   = "nlp-disinformation-detection-aitu"
BRANCH      = "main"

RUN_BASELINE          = True
RUN_TRANSFORMER       = True
RUN_HYBRID            = True
RUN_HYBRID_TEXTONLY   = True

PUSH_RESULTS_TO_GITHUB = True   # set False if you just want to smoke-test
ZIP_CHECKPOINTS        = False  # True if you want to download models/ manually
```

If you only want to re-run one stage (e.g. you iterated on the hybrid
config), set the other `RUN_*` flags to `False`.

---

## 4. Run the notebook

Click **Run All**. Cells execute top-to-bottom:

| # | Cell | What happens |
|---|------|--------------|
| 1 | Config | Reads the flags above. |
| 2 | GPU check | `nvidia-smi` — verify the P100 is attached. |
| 3 | Clone | `git clone --depth 1 --branch main` into `/kaggle/working/repo`. |
| 4 | Install | `pip install --no-deps -e .` (torch/transformers already installed by Kaggle). Downloads NLTK `stopwords`/`punkt`. |
| 5 | Dataset | Downloads LIAR (`make download-liar`) into `data/raw/`. |
| 6 | Preprocess | `make preprocess` → `data/processed/*.csv`. |
| 7 | Baseline | `make baseline`. |
| 8 | Transformer | `make transformer`. |
| 9 | Hybrid | `make hybrid`. |
| 10 | Hybrid text-only | Generates a temp config with `use_metadata: false` and redirected `logs_dir`/`models_dir`, then trains — this is the RQ2 ablation. |
| 11 | Collect | Aggregates all `*_test_metrics.json` into `reports/results_summary.json`. |
| 12 | Push | Commits JSON files to `main` via `GH_PAT`. |
| 13 | Zip (optional) | Zips `models/` so you can download checkpoints from the **Output** tab. |

Each training cell prints wall-clock time so you know where the minutes
went.

---

## 5. What gets pushed back to the repo

Only JSON metric files are pushed. Model weights (hundreds of MB) are
**not** pushed — grab them from Kaggle's **Output** tab if you need
them locally.

Files committed to `main`:

```
reports/results_summary.json
reports/baseline_detailed_metrics.json
reports/transformer_logs/transformer_test_metrics.json
reports/hybrid_logs/hybrid_test_metrics.json
reports/hybrid_textonly_logs/hybrid_test_metrics.json
```

Each JSON contains macro-F1, test accuracy, and per-class F1 on the
TEST split (1283 examples). These are the numbers that go into the
IEEE paper.

The commit message is `chore(results): add Kaggle training run metrics`.

---

## 6. Downloading the trained models

Checkpoints are written to `models/` inside the Kaggle session only.

- **Individual files**: open the **Output** tab on the right, browse to
  `repo/models/...`, click **Download**.
- **Everything at once**: set `ZIP_CHECKPOINTS = True` in the config
  cell. The final cell zips `models/` into
  `/kaggle/working/models.zip` — then download the single archive from
  **Output**.

---

## 7. Troubleshooting

**Transformer crashes with `CUDA error: no kernel image is available for execution on the device` / `cudaErrorNoKernelImageForDevice` (P100 GPU)**
You are on a P100 (`sm_60`) but Kaggle's default `torch` wheel only contains kernels for
`sm_70+`. Two fixes:

- **Preferred**: in the right sidebar, change **Accelerator** to `GPU T4 x2` and restart
  the kernel. T4 has `sm_75` and just works.
- **Fallback** (if you really need P100): enable Cell 2b in the notebook to reinstall
  `torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1` from the official `cu121` index,
  which includes `sm_60` kernels. Then **Run → Restart kernel** and continue from Cell 3.

**`RuntimeError: Could not load Kaggle Secret 'GH_PAT'`**
You either didn't add the secret or didn't attach it to the notebook.
Go to **Add-ons** → **Secrets** and toggle `GH_PAT` on.

**`fatal: Authentication failed for 'https://github.com/...'`**
The token expired, was revoked, or lacks `repo` scope. Generate a new
PAT and update the Kaggle secret.

**Transformer cell dies with `CUDA out of memory`**
Unlikely on a P100 with our config (`batch 16 × accum 2`, `max_length
64`). If it happens, open `config/transformer.yaml` and drop
`batch_size` to 8.

**`make: *** No rule to make target 'baseline'`**
The repo didn't clone correctly. Re-run cell 3 (clone) and cell 4
(install).

**No GPU detected in cell 2**
The session was created without an accelerator. In the right sidebar,
change **Accelerator** to **GPU P100** and **restart** the kernel.

**Kaggle kicks you off after 20 min of inactivity**
Normal. As long as cells are executing, the session stays alive. If
you close the browser tab, the run continues — come back later and
check the **Output** tab.

---

## 8. Regenerating the notebook

Do **not** hand-edit `notebooks/kaggle_training.ipynb`. It is generated
from `notebooks/_build_kaggle_notebook.py`:

```bash
python notebooks/_build_kaggle_notebook.py
```

Edit the Python file, re-run the generator, commit both files.
