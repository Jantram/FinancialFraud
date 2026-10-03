# Results — Fin-JEPA on AMLSim

## Self-supervised pretraining
20 epochs, batch size 256, AdamW (lr=1e-3, cosine schedule).
Best val loss: 4.58 (pred_loss: 0.598, sigreg_loss: separate scale, not comparable across splits).

## Downstream fraud detection (account-level)
| Method | Test AUC | Test Avg. Precision |
|---|---|---|
| MLP probe, model embedding only (64-dim) | 0.665 | 0.322 |
| MLP probe, embedding + prediction error (66-dim), 500 epochs w/ early stopping | 0.697 | 0.340 |
| XGBoost, embedding + error + hand-crafted features (76-dim) | **0.857** | **0.678** |

## Precision at top-N flagged accounts (XGBoost model, test set, 247/1497 fraud accounts)
| Top N | Fraud caught | Precision | % of all fraud caught |
|---|---|---|---|
| 20 | 20 | 100.0% | 8.1% |
| 100 | 90 | 90.0% | 36.4% |
| 500 | 192 | 38.4% | 77.7% |

## Most important features (XGBoost)
1. log_n_partners_out (# distinct accounts sent TO)
2. log_n_partners_in (# distinct accounts received FROM)
3-15. Mix of Fin-JEPA learned embedding dims, error_max, amt_std, amt_max, sent_ratio

## Key finding
Hand-crafted account statistics (counterparty diversity especially) outperformed the raw
self-supervised embedding alone, but the two were complementary — combining them beat either
alone.
