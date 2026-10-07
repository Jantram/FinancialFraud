# Results — Fin-JEPA on AMLSim

These results come from the experiment implemented in `finjepa (1).ipynb`.

## Self-supervised pretraining

- Epochs: 20
- Batch size: 256
- Optimizer: AdamW
- Learning rate: 1e-3
- Learning-rate schedule: CosineAnnealingLR
- Best validation loss: 4.58

The reported validation loss combines the prediction loss and the SIGReg regularization term. The individual SIGReg component is on a separate scale and should not be compared directly across splits.

## Downstream fraud detection

The downstream task is **account-level fraud detection**.

| Method | Test ROC-AUC | Test Average Precision |
|---|---:|---:|
| MLP probe — model embedding only (64-dim) | 0.665 | 0.322 |
| MLP probe — embedding + prediction error (65-dim) | 0.697 | 0.340 |
| XGBoost — embedding + prediction error + handcrafted features (76-dim) | **0.8574** | **0.6775** |

The hybrid XGBoost model combines Fin-JEPA-derived features with handcrafted account-level transaction statistics.

## Test-set size

- Test accounts: 1,497
- Fraud accounts: 247

## Precision at top-N flagged accounts

Results below use the XGBoost risk scores on the test accounts.

| Top N | Fraud caught | Precision | % of all fraud caught |
|---:|---:|---:|---:|
| 20 | 20 | 100.0% | 8.1% |
| 50 | 49 | 98.0% | 19.8% |
| 100 | 90 | 90.0% | 36.4% |
| 150 | 116 | 77.3% | 47.0% |
| 200 | 130 | 65.0% | 52.6% |
| 300 | 160 | 53.3% | 64.8% |
| 500 | 192 | 38.4% | 77.7% |

## XGBoost configuration

The final hybrid XGBoost model used:

- 500 estimators
- Maximum tree depth: 4
- Learning rate: 0.03
- Subsample: 0.8
- Column subsampling: 0.8
- Class-imbalance weighting (`scale_pos_weight`)
- Early stopping: 30 rounds
- Random seed: 42

## Feature composition

The final feature vector contains 76 features:

- 64 pooled Fin-JEPA embedding features
- 2 prediction-error features:
  - `error_mean`
  - `error_max`
- 6 basic handcrafted account features
- 4 additional transaction/account features

The most important features reported by the XGBoost model include counterparty-diversity features such as:

1. `log_n_partners_out`
2. `log_n_partners_in`

followed by a mixture of Fin-JEPA embedding dimensions, prediction-error features, and handcrafted transaction statistics.

## Key observation

The final hybrid model performed substantially better than the embedding-only and embedding-plus-error MLP probes.

However, this result should **not** be interpreted as proof that the Fin-JEPA representation alone provides the entire improvement. The final XGBoost model also uses handcrafted account-level features. Additional ablation experiments would be required to isolate the incremental contribution of the learned Fin-JEPA features.

## Reproducibility

The executable experiment is provided in:

`finjepa (1).ipynb`

The notebook contains the AMLSim loading, preprocessing, account-level splitting, window construction, Fin-JEPA pretraining, downstream probes, XGBoost evaluation, and result generation.

`data_pipeline.py` documents the preprocessing contract used by the notebook, while `model.py` contains the Fin-JEPA model implementation used by the experiment.
