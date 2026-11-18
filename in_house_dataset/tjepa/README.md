# Fraud detection on a time-split payments dataset with T-JEPA

Early-warning fraud detection on an in-house payments dataset using self-supervised tabular representation learning ([T-JEPA](https://github.com/jose-melo/t-jepa)). The notebook compares T-JEPA with simple baselines under a strict time-based split with delayed, partial labels, and reports every result against the hidden ground truth of the test period.

## Repository contents

| File | Purpose |
|---|---|
| `inhouse-tjepa-clean.ipynb` | The full pipeline: data audit, features, labels and validation, T-JEPA pretraining, ablation, final evaluation and metrics. Saved outputs are included. |
| `README.md` | This file. |

The dataset is not part of this repository (see [Data](#data)).

## Task and data

- **Goal:** rank charges by fraud risk so that a small review budget (for example the top 1% of charges) catches as much fraud as possible.
- **Data:** 43,716 charges from 3,992 customers, 2026-01-05 to 2026-07-03. The dataset card describes the data as synthetic.
- **Split:** time-based at 2026-04-23: 22,991 train and 20,725 test charges. Fraud rate 0.74% in train and 0.61% in test. The data are never split randomly.
- **Fraud types:** `ring` (140 frauds, train only), `ato` (17 train, 24 test), `emerge` (96, test only) and a few `normal` frauds (12 train, 7 test). The dominant test fraud type never appears in training, so the benchmark measures robustness to novel fraud.
- **Labels:** delayed, partial labels from `label_timeline.csv` (`efw`, `review`, `dispute`, `refund`, each with its own arrival lag) are the only labels used for training. The true labels in `_ground_truth.csv` are used for evaluation only.

## Method

1. **Features.** 29 numeric and 12 categorical features per charge, including history features computed only from earlier charges (velocity, running amount statistics, card reuse across customers). Charges with little or no history are kept and flagged, not dropped.
2. **Pretraining without labels.** The T-JEPA encoder is trained on train-period charges only. Part of each charge's feature tokens is hidden, and the model predicts the latent representation of the hidden tokens from the visible ones.
3. **Fraud head.** The frozen encoder produces an embedding per charge. A class-weighted logistic regression (or a small MLP) is trained on the labels that had already arrived at the split date (`efw` and `dispute`), using charges old enough to have reliable labels.
4. **Scoring.** Every test charge receives a fraud score. The top 0.5%, 1% and 2% of scores form the review queue.

### Evaluation protocol

- Preprocessing is fitted on training data only.
- A label counts as known only if it had arrived by the split date; charges younger than 14 days are excluded from supervision.
- The model and the epoch are selected on validation data only (chronological window plus a customer-grouped cross-check inside the training period). The test period is used once, in the final evaluation.
- The primary result (linear probe on the `cls_mean` embedding) was fixed before looking at test results.
- Metrics: PR-AUC and ROC-AUC, recall and precision at fixed alert budgets, confusion matrices, recall per fraud type, and bootstrap confidence intervals.

### Compared models

- T-JEPA **A**: the repository's objective, unchanged.
- T-JEPA **C**: A with a deterministic target encoder and targets standardised per feature position (a documented ablation).
- An **untrained encoder** with the same architecture and heads (does pretraining help at all?).
- **Logistic regression** and **XGBoost** on the raw features.

## Results

Saved run, test period: 20,725 charges, 127 frauds (96 `emerge`, 24 `ato`, 7 `normal`). Chance PR-AUC is 0.0061. Mean ± standard deviation over 3 seeds; T-JEPA rows use the pre-registered linear probe on the `cls_mean` embedding.

| Model | PR-AUC | ROC-AUC | Recall @ 1% alerts | Recall `emerge` | Recall `ato` |
|---|---|---|---|---|---|
| Logistic regression, raw features | 0.223 | 0.952 | 0.409 | 0.542 | 0.000 |
| T-JEPA C, epoch 20 | 0.174 ± 0.012 | 0.925 ± 0.007 | 0.344 ± 0.026 | 0.427 ± 0.060 | 0.111 ± 0.104 |
| T-JEPA A, epoch 10 | 0.152 ± 0.076 | 0.906 ± 0.044 | 0.278 ± 0.125 | 0.330 ± 0.197 | 0.153 ± 0.129 |
| Untrained encoder | 0.115 ± 0.043 | 0.884 ± 0.028 | 0.247 ± 0.072 | 0.281 ± 0.120 | 0.181 ± 0.104 |
| XGBoost, raw features | 0.029 ± 0.004 | 0.846 ± 0.004 | 0.034 ± 0.010 | 0.035 ± 0.013 | 0.042 ± 0.000 |

What the results show, and what they do not:

- A **logistic regression on the raw features is the strongest model**: best on PR-AUC, ROC-AUC and recall at the 1% and 2% budgets (tied at 0.5%). T-JEPA did not beat it. The 95% bootstrap intervals for PR-AUC overlap (logistic regression 0.223 [0.172, 0.298], T-JEPA C 0.185 [0.136, 0.250], seed-averaged scores).
- **Pretraining helps only modestly and not consistently.** On the primary metric both pretrained variants beat the untrained encoder, but the seed variation is large. With an MLP head on the flattened embeddings, the untrained encoder scored higher (0.184 ± 0.007 versus 0.139 ± 0.011 for A and 0.120 ± 0.038 for C).
- **XGBoost fails under this shift.** Trees trained on `ring` and `ato` fraud learn rules that do not transfer to `emerge`.
- **Different models catch different fraud types.** At the 1% budget the raw linear model catches 52 of 96 `emerge` frauds but none of the 24 `ato` frauds; the T-JEPA models catch 2 to 5 of the 24 `ato` frauds (very small counts).

## Reproducing

1. **Data.** Provide `sequences.csv`, `label_timeline.csv` and `_ground_truth.csv`. On Kaggle, attach them as a dataset; elsewhere set the environment variable `FRAUD_DATA_DIR` to the folder that contains them.
2. **Environment.** Python 3 with PyTorch, NumPy, pandas, SciPy, scikit-learn, XGBoost and Matplotlib (the saved run used PyTorch 2.11 on Kaggle). `einops` and `tabulate` are installed automatically. Internet access is needed once, to clone the T-JEPA repository at a pinned commit.
3. **Run the notebook from top to bottom.** All settings are in the configuration block of the first code cell. Set `QUICK_TEST = True` for a short smoke test (4 epochs, 1 seed; the results are not meaningful).
4. **Runtime.** The saved outputs come from a full run on **CPU, which took about 4.3 hours**. A GPU is recommended.
5. **Outputs** are written to the working directory: `tjepa_final_results.csv`, `tjepa_test_scores.npz` (test scores per model and seed, for later comparison with other models) and the selected checkpoint.

## Data

The dataset is in-house and is not distributed with this repository; do not commit it. The notebook expects one row per charge in `sequences.csv` (time-ordered per customer, with a `split` column), delayed labels in `label_timeline.csv` joined on `charge_id = id`, and true labels in `_ground_truth.csv` (evaluation only). A suggested `.gitignore`:

```
*.csv
*.parquet
work/
*.pt
*.npz
```

## Limitations

- The test period is dominated by a fraud type (`emerge`, 96 of 127 frauds) that never occurs in training. Differences between methods partly reflect robustness to novel fraud and not only representation quality.
- There are only 127 test frauds and 3 seeds, so confidence intervals are wide and small differences should not be over-read.
- The chronological validation window contains only 14 frauds (4 in the 7 to 23 April window), so validation PR-AUC cannot discriminate between epochs. Epoch selection therefore uses a customer-grouped cross-check inside the training period.
- Results are sensitive to the run: in a separate GPU run, variant C selected epoch 10 instead of 20 and the final numbers differed slightly.
- Hyper-parameters were not tuned. The selected epochs (10 to 20) mean the encoder is only lightly pretrained; longer pretraining reduced fraud separability in the cross-check (representation collapse).
- Not yet done: a clipping ablation (`CLIP = None`), additional behavioural features, a larger seed budget, and a comparison with Fin-JEPA.
- The "best T-JEPA combination" in the last section of the notebook is chosen on the test set and is exploratory only.

## Attribution

The T-JEPA encoder, predictor, masking and optimiser modules come from [jose-melo/t-jepa](https://github.com/jose-melo/t-jepa) (MIT licence, copyright 2024 Hugo Thimonier). They are cloned at run time at commit `98dda99` and are not copied into this repository. See that repository for the method's reference publication.
