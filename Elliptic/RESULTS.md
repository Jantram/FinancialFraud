# Results - T-JEPA on Elliptic

## Dataset split

The notebook uses a chronological temporal split based on the `time_step` field.

| Split | Time steps | Transactions | Fraud |
|---|---:|---:|---:|
| Train | <= 34 | 29,894 | 3,462 |
| Validation | 35-39 | 5,486 | 447 |
| Test | >= 40 | 11,184 | 636 |

The corrected label mapping is:

- `1` = Illicit / Fraud
- `2` = Licit / Legitimate

## T-JEPA pretraining

The notebook trains the self-supervised T-JEPA model for **20 epochs** before generating embeddings.

The experiment uses:

- Random seed: `42`
- Hidden dimension: `256`
- Embedding dimension: `128`
- Feature masking probability: `30%`
- Batch size: `512`
- Learning rate: `1e-3`
- Weight decay: `1e-4`

## Downstream fraud classifier

The corrected downstream classifier is trained using the T-JEPA embeddings.

The classifier uses:

- Input: T-JEPA embeddings
- Hidden layers: `128 -> 64`
- Dropout: `0.3` and `0.2`
- BCEWithLogitsLoss with class-imbalance weighting
- Adam optimizer
- Learning rate: `0.001`
- Weight decay: `1e-4`
- Batch size: `256`
- Maximum epochs: `30`
- Early stopping based on validation Average Precision
- Best validation Average Precision: `0.6502`

## Final test performance

Results below are reported at the default classification threshold of `0.50`.

| Metric | Value |
|---|---:|
| Accuracy | 0.8881 |
| Fraud Precision | 0.2809 |
| Fraud Recall | 0.6211 |
| Fraud F1 | 0.3869 |
| ROC-AUC | 0.8421 |
| PR-AUC / Average Precision | 0.4980 |

### Confusion matrix

```text
[[9537, 1011],
 [ 241,  395]]```

### Classification report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Licit (0) | 0.98 | 0.90 | 0.94 | 10,548 |
| Illicit/Fraud (1) | 0.28 | 0.62 | 0.39 | 636 |

## Generated outputs

The notebook saves the following corrected outputs:

- `tjepa_elliptic_pr_curve_corrected.png`
- `tjepa_precision_recall_table_corrected.csv`
- `tjepa_elliptic_metrics_corrected.csv`
- `tjepa_elliptic_test_predictions_corrected.csv`
- `tjepa_elliptic_classifier_corrected.pth`

## Reproducibility

The notebook sets random seeds to `42` for Python, NumPy, and PyTorch.

The experiment is designed to run in a Kaggle environment with the Elliptic dataset available at the dataset path specified in the notebook.