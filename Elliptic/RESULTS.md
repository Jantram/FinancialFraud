# Results — T-JEPA on Elliptic

Official temporal split. Train 24,923 / Val 4,971 / Test 16,670 labeled transactions.

## Test performance @ threshold 0.50
| Metric | Value |
|---|---|
| Accuracy | 0.8603 |
| Fraud Precision | 0.2823 |
| Fraud Recall | 0.7452 |
| Fraud F1 | 0.4094 |
| ROC-AUC | 0.8878 |
| PR-AUC | 0.4246 |

## Best F1 threshold: 0.90
Precision 0.4266, Recall 0.6168, F1 0.5043. Confusion matrix: [[14689, 898], [415, 668]].

## Important caveat
The self-supervised context encoder was trained for only 1 epoch before embeddings were
generated for the downstream MLP (`args.exp_train_total_epochs = 1` in the notebook). The
strong ROC-AUC may partly reflect that the flattened (166x64) embeddings still retain a lot of
near-raw feature information, rather than deeply learned representations. This is flagged as a
limitation to revisit -- a longer self-supervised pretraining run on Elliptic is a natural next
step for a fairer comparison against Fin-JEPA (pretrained 20 epochs on AMLSim).
