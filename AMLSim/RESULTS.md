# Results — Fin-JEPA on AMLSim

Two runs of the same experiment are reported:

- **v2 run:** `finjepa-amlsim-v2.ipynb` (clean version: data processing, training and inference only)
- **Original run:** `finjepa (1).ipynb` (full experiment with the exploratory probes)

Results differ slightly between runs because of random initialization and GPU nondeterminism.

## Self-supervised pretraining

- Epochs: 20
- Batch size: 256
- Optimizer: AdamW
- Learning rate: 1e-3
- Learning-rate schedule: CosineAnnealingLR
- Best validation loss (original run): 4.58

The reported validation loss combines the prediction loss and the SIGReg regularization term. The individual SIGReg component is on a separate scale and should not be compared directly across splits. The downstream steps use the final-epoch weights.

## Downstream fraud detection

The downstream task is **account-level fraud detection**.

| Method | Test ROC-AUC | Test Average Precision |
|---|---:|---:|
| MLP probe — model embedding only (64-dim), original run | 0.665 | 0.322 |
| MLP probe — embedding + prediction error, original run | 0.697 | 0.340 |
| XGBoost — embedding + prediction error + handcrafted features (76-dim), original run | 0.8574 | 0.6775 |
| **XGBoost — same features, v2 run** | **0.8518** | **0.6773** |

The hybrid XGBoost model combines Fin-JEPA-derived features with handcrafted account-level transaction statistics. The MLP probes are only in the original notebook.

## v2 run: test metrics at threshold 0.5

- Test accounts: 1,497
- Fraud accounts: 247 (16.5%, which is also the PR-AUC baseline)

| Metric | Value |
|---|---:|
| ROC-AUC | 0.8518 |
| PR-AUC (Average Precision) | 0.6773 |
| Precision (fraud) | 0.4258 |
| Recall (fraud) | 0.7085 |
| F1 (fraud) | 0.5319 |
| Accuracy | 0.7943 |

Confusion matrix:

| | Predicted normal | Predicted fraud |
|---|---:|---:|
| True normal | 1,014 | 236 |
| True fraud | 72 | 175 |

## Precision at top-N flagged accounts

Results use the XGBoost risk scores on the test accounts.

| Top N | v2: fraud caught | v2: precision | v2: % of all fraud caught | Original: fraud caught | Original: precision |
|---:|---:|---:|---:|---:|---:|
| 20 | 20 | 100.0% | 8.1% | 20 | 100.0% |
| 50 | 49 | 98.0% | 19.8% | 49 | 98.0% |
| 100 | 91 | 91.0% | 36.8% | 90 | 90.0% |
| 150 | 117 | 78.0% | 47.4% | 116 | 77.3% |
| 200 | 128 | 64.0% | 51.8% | 130 | 65.0% |
| 300 | 158 | 52.7% | 64.0% | 160 | 53.3% |
| 500 | 189 | 37.8% | 76.5% | 192 | 38.4% |

## XGBoost configuration

- 500 estimators (maximum)
- Maximum tree depth: 4
- Learning rate: 0.03
- Subsample: 0.8
- Column subsampling: 0.8
- Class-imbalance weighting (`scale_pos_weight`)
- Early stopping: 30 rounds on validation AUC
- Random seed: 42

## Feature composition

The final feature vector contains 76 features:

- 64 pooled Fin-JEPA embedding features
- 2 prediction-error features: `error_mean`, `error_max`
- 6 basic handcrafted account features
- 4 additional transaction/account features

The most important features include counterparty-diversity features such as `log_n_partners_out` and `log_n_partners_in`, followed by a mixture of Fin-JEPA embedding dimensions, prediction-error features and handcrafted transaction statistics.

## Key observation

The final hybrid model performed substantially better than the embedding-only and embedding-plus-error MLP probes.

However, this result should **not** be interpreted as proof that the Fin-JEPA representation alone provides the improvement. The final XGBoost model also uses handcrafted account-level features. Additional ablation experiments, such as XGBoost on the handcrafted features alone, would be required to isolate the contribution of the learned Fin-JEPA features.

## Reproducibility

The clean executable experiment is `finjepa-amlsim-v2.ipynb`. It contains AMLSim loading, preprocessing, account-level splitting, window construction, Fin-JEPA pretraining, account feature building, XGBoost training and test evaluation. The original, longer experiment is in `finjepa (1).ipynb`.

`data_pipeline.py` documents the preprocessing contract used by the notebooks, and `model.py` contains the Fin-JEPA model implementation.
