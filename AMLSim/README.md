# Fin-JEPA for Fraud Detection on AMLSim

Fin-JEPA, a self-supervised sequence model adapted from the public
[fin-jepa](https://github.com/cedricwyh/fin-jepa) repository, applied to bank
transaction histories from the IBM AMLSim dataset. The model is pretrained
without labels; its embeddings are then combined with hand-crafted account
statistics to detect fraudulent accounts.

## Files

| File | Description |
|---|---|
| `finjepa-amlsim-v2.ipynb` | **v2 (clean):** data processing, pretraining, account features, XGBoost and test evaluation only |
| original notebook | Full experiment history, kept unchanged |
| `model.py`, `data_pipeline.py` | Model code and data-pipeline notes |
| `RESULTS.md` | Results summary |

## Method

1. **Data:** sent and received transactions are merged into one time-ordered history per
   account. Row features: normalized log amount, direction (sent / received), normalized
   time gap.
2. **Split:** accounts are split 70 / 15 / 15 (random, seed 42). Normalization uses
   training accounts only.
3. **Windows:** 15 context + 5 target transactions, stride 5. Each window inherits its
   account's fraud label.
4. **Pretraining (no labels):** a shared per-transaction encoder, a causal transformer
   predictor (4 layers, 4 heads, dim 64) and the SIGReg regularizer. Loss = prediction MSE
   + 0.1 x SIGReg. AdamW, learning rate 1e-3, cosine schedule, 20 epochs, batch size 256.
5. **Account features (76):** mean window embedding (64), prediction error mean and max (2),
   and 10 hand-crafted statistics (transaction count, amount statistics, sent ratio,
   number of distinct counterparties in total / incoming / outgoing, active span, time-gap
   standard deviation).
6. **Classifier:** XGBoost (up to 500 trees, depth 4, learning rate 0.03, class-imbalance
   weighting, early stopping on validation AUC).

## Results (v2 run, test accounts: 1,497, of which 247 fraud)

| Metric | Value |
|---|---|
| ROC-AUC | 0.8518 |
| PR-AUC (Average Precision) | 0.6773 |
| Precision (fraud, threshold 0.5) | 0.4258 |
| Recall (fraud, threshold 0.5) | 0.7085 |
| F1 (fraud, threshold 0.5) | 0.5319 |
| Accuracy | 0.7943 |

The PR-AUC baseline (random ranking) equals the fraud share, about 0.165.

Confusion matrix at threshold 0.5:

| | Predicted normal | Predicted fraud |
|---|---|---|
| True normal | 1,014 | 236 |
| True fraud | 72 | 175 |

Precision when only the top-N highest-risk accounts are investigated:

| Top N | Fraud caught | Precision | % of all fraud caught |
|---|---|---|---|
| 20 | 20 | 100.0% | 8.1% |
| 50 | 49 | 98.0% | 19.8% |
| 100 | 91 | 91.0% | 36.8% |
| 150 | 117 | 78.0% | 47.4% |
| 200 | 128 | 64.0% | 51.8% |
| 300 | 158 | 52.7% | 64.0% |
| 500 | 189 | 37.8% | 76.5% |

## Notes

- Hand-crafted counterparty counts are the most important features; the pretrained
  embeddings alone give much weaker results in the original experiments.
- Results vary slightly between runs because of GPU nondeterminism.
- The downstream steps use the final-epoch pretrained weights.

## How to run

Open `finjepa-amlsim-v2.ipynb` on Kaggle, attach the *IBM AMLSim example dataset*,
enable a GPU and run all cells.
