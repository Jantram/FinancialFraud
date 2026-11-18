# Fin-JEPA on In-House Fraud Detection Dataset

Chronological training, validation-based checkpoint selection, and multi-seed test evaluation of [**Fin-JEPA**](https://github.com/cedricwyh/fin-jepa) on an in-house transaction fraud dataset (`43,716` charges across `3,992` customers).

## Overview

This project adapts the `cedricwyh/fin-jepa` Joint-Embedding Predictive Architecture (`PriceEncoder` + causal `TransformerPredictor` + `SIGReg` regularization) to customer-level transaction sequences and benchmarks it against supervised tabular baselines (`LogReg`, `XGBoost`) and an untrained encoder control.

### Key Findings

- **Strongest overall baseline:** `LogReg` on `29` standardized numeric features achieves **0.223 PR-AUC** (95% CI `[0.172, 0.298]`) and **0.952 ROC-AUC**, catching **52/127** frauds (`40.9%` recall) at a `1.0%` alert budget.
- **EMA vs. non-EMA pretraining:** Validation checkpoint selection (`VAL_WIN`) picks epoch `35` for `A_repo` (no EMA) and epoch `5` for `E_ema` (EMA target encoder). On the test set, `E_ema` (`0.033` bootstrap PR-AUC, `0.845` ROC-AUC) substantially outperforms `A_repo` (`0.007` bootstrap PR-AUC, `0.592` ROC-AUC), though both remain on par with or below the untrained encoder (`0.025–0.030` PR-AUC) and `LogReg`.

---

## Dataset & Chronological Splits

- **Source files:** `sequences.csv` (`43,716 x 51`), `label_timeline.csv` (`43,716 x 15`), `_ground_truth.csv` (`43,716 x 3`).
- **Time span:** `2026-01-05` to `2026-07-03` (train/test boundary at `2026-04-23 00:18:05`).
- **Observed labels:** `efw` and `dispute` channels with a `14`-day maturity window.

| Split | Date Range | Rows | True Frauds | Purpose |
| :--- | :--- | ---: | ---: | :--- |
| `SSL_TRAIN` | `2026-01-05` → `2026-03-08` | 12,984 | 155 | Self-supervised pretraining & scaler fitting |
| `HEAD_FIT_V` | `2026-01-05` → `2026-02-22` | 10,070 | 153 (`111` labelled) | Probe training for validation selection |
| `VAL_WIN` | `2026-03-09` → `2026-04-22` | 10,007 | 14 (`10` `ato`, `4` `normal`) | Checkpoint & variant selection (no test leakage) |
| `TEST` | `2026-04-23` → `2026-07-03` | 20,725 | 127 (`96` `emerge`, `24` `ato`, `7` `normal`) | Final held-out evaluation |

---

## Feature Engineering & Architecture

- **Input representation (`192` features/charge):**
  - `29` numeric features: `10` binary flags kept raw (`0/1`) and `19` continuous features standardized on the training split only and clipped to `±5.0`.
  - `12` categorical features one-hot encoded (`163` columns, including unknown buckets; `0` unseen categories in test).
  - Causality verified (`max |diff| = 0.0e+00` when truncating future rows).
- **Causal sequence windows (`L = 32`):**
  - `3,049` pretraining windows (`10,349` next-step pairs/epoch).
  - Test windows include train-period customer history (`77.9%` of test charges use train context; `100%` of `20,725` test charges are scored).
- **Model (`278,912` parameters):**
  - `PriceEncoder`: `54,016` params (`embed_dim=64`)
  - `TransformerPredictor`: `224,896` params (causal attention + learned positional embeddings up to `256` steps)
  - `SIGReg`: anti-collapse regularization (`weight = 0.1`)
- **Ablation variants:**
  - `A_repo`: shared online/target encoder (`ema=False`), selected at **epoch 35**.
  - `E_ema`: exponential moving average target encoder (`ema=True`), selected at **epoch 5**.

---

## Notebook Structure (`inhouse-finjepa-cleaned.ipynb`)

1. **Step 1 — Environment, Repository & Data Load:** Seeds, `cedricwyh/fin-jepa` clone, integrity checks, and split/typology audits.
2. **Step 2 — Feature Engineering & Fin-JEPA Input Matrix:** Train-only scaling, one-hot encoding, causality check, and drift inspection.
3. **Step 3 — Label Timeline, Chronological Splits & Class Imbalance:** `14`-day label maturity handling, class weighting (capped at `30.0`), and `SSL_TRAIN` / `HEAD_FIT_V` / `VAL_WIN` / `TEST` definitions.
4. **Step 4 — Causal Sequence Windows & Architecture Check:** Window length comparison (`L ∈ {4, 8, 16, 32, 64}`), `L=32` window construction, and causal masking verification.
5. **Step 5 — Training Machinery & One-Batch Sanity Check:** `SIGReg` + prediction loss, gradient flow checks, and untrained latent diagnostics.
6. **Step 6 — Pretraining & Validation Checkpoint Selection (`A_repo` vs. `E_ema`):** 100-epoch pretraining on `3,049` windows and checkpoint selection via grouped-CV average precision on `z_h`.
7. **Step 7 — Final Refit, Multi-Seed Test Evaluation & Baselines:** Full train refit across seeds `[42, 1042, 2042]` and evaluation against `LogReg`, `XGBoost`, and untrained encoder controls.
8. **Step 8 — Confusion Matrices, ROC/PR Curves & Bootstrap CIs:** Alert-budget confusion matrices (`0.5%`, `1.0%`, `2.0%`), fraud-type recall, `95%` bootstrap CIs, and ROC/PR curves.

---

## Test Results Summary

Evaluated on `20,725` test charges (`127` frauds; chance PR-AUC = `0.0061`). Bootstrap metrics below use seed-averaged rank-normalised scores.

| Model | Features / Mode | Head | PR-AUC (95% CI) | ROC-AUC | Recall @ 1% | Precision @ 1% |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `raw features` | `numeric` (29 cols) | `LogReg` | **0.223** `[0.172, 0.298]` | **0.952** | **0.409** (52/127) | **0.251** |
| `Fin-JEPA [E_ema @ ep 5]` | `z_h` (`[z, h]`) | `LinearProbe` | 0.033 `[0.027, 0.044]` | 0.845 | 0.016 (2/127) | 0.010 |
| `raw features` | `all` (192 cols) | `XGBoost` | 0.030 `[0.024, 0.040]` | 0.867 | 0.031 (4/127) | 0.019 |
| `Fin-JEPA untrained` | `z` (exploratory) | `LinearProbe` | 0.029 `[0.022, 0.037]` | 0.864 | 0.039 (5/127) | 0.024 |
| `Fin-JEPA untrained` | `z_h` (`[z, h]`) | `LinearProbe` | 0.025 `[0.019, 0.033]` | 0.818 | 0.024 (3/127) | 0.014 |
| `Fin-JEPA [A_repo @ ep 35]` | `z_h` (`[z, h]`) | `LinearProbe` | 0.007 `[0.006, 0.008]` | 0.592 | 0.000 (0/127) | 0.000 |

---

## Usage

1. Upload `inhouse-finjepa-cleaned.ipynb` to Kaggle (or a local environment with a CUDA GPU) and attach the `in-house-dataset` containing `sequences.csv`, `label_timeline.csv`, and `_ground_truth.csv`.
2. Ensure internet access is enabled in Step 1 to clone `https://github.com/cedricwyh/fin-jepa` (or set `FRAUD_DATA_DIR` if running locally).
3. Run Steps 1–8 sequentially. Step 7 saves `finjepa_final_results.csv` and `finjepa_test_scores.npz` to the working directory.
