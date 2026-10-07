# T-JEPA on Elliptic Bitcoin Fraud Detection

This repository contains the code and notebook used for the T-JEPA-based
fraud detection experiment on the Elliptic Bitcoin transaction dataset.

The experiment adapts a JEPA-style self-supervised learning approach to
learn transaction representations and then uses those embeddings for
downstream fraud classification.

## Repository Contents

- `t-jepa-elliptic-fraud-detection(2).ipynb` - Main experiment notebook.
- `elliptic_dataset.py` - Dataset-related code used for the Elliptic experiment.
- `RESULTS.md` - Documented dataset split, training configuration, and final results.

## Dataset

The experiment uses the Elliptic Bitcoin transaction dataset.

The notebook expects the following files:

- `elliptic_txs_features.csv`
- `elliptic_txs_classes.csv`
- `elliptic_txs_edgelist.csv`

The notebook is configured for a Kaggle environment and expects the dataset
under the Kaggle input directory specified in the notebook.

## Data Preparation

The transaction features and class information are merged using `txId`.

Only transactions with known classes are used:

- `1` = Illicit / Fraud
- `2` = Licit / Legitimate

The transaction records are ordered by `time_step`.

Feature values are standardized using `StandardScaler`. The scaler is fitted
only on the training data and then applied to the validation and test data.

## Temporal Dataset Split

The experiment uses a chronological split based on the `time_step` field.

| Split | Time steps | Transactions | Fraud |
|---|---:|---:|---:|
| Train | <= 34 | 29,894 | 3,462 |
| Validation | 35-39 | 5,486 | 447 |
| Test | >= 40 | 11,184 | 636 |

This prevents future time steps from being used to train the model.

## T-JEPA Pretraining

The notebook trains a self-supervised T-JEPA model before the downstream
fraud classification stage.

The model uses:

- Random seed: `42`
- Hidden dimension: `256`
- Embedding dimension: `128`
- Feature masking probability: `30%`
- Batch size: `512`
- Learning rate: `1e-3`
- Weight decay: `1e-4`
- Pretraining epochs: `20`

The model uses a context encoder, target encoder, predictor, and time-step
embedding. The target encoder is updated using exponential moving average
updates.

The self-supervised objective is based on predicting target representations
from masked transaction features.

## Downstream Fraud Classifier

After T-JEPA pretraining, transaction embeddings are generated using the
context encoder.

A binary fraud classifier is then trained on these embeddings.

The classifier uses:

- Input: T-JEPA embeddings
- Hidden layers: `128 -> 64`
- Dropout: `0.3` and `0.2`
- Loss: `BCEWithLogitsLoss` with class-imbalance weighting
- Optimizer: Adam
- Learning rate: `0.001`
- Weight decay: `1e-4`
- Batch size: `256`
- Maximum epochs: `30`
- Early stopping based on validation Average Precision

## Results

The final test results are documented in [`RESULTS.md`](RESULTS.md).

At the default classification threshold of `0.50`:

| Metric | Value |
|---|---:|
| Accuracy | 0.8881 |
| Fraud Precision | 0.2809 |
| Fraud Recall | 0.6211 |
| Fraud F1 | 0.3869 |
| ROC-AUC | 0.8421 |
| PR-AUC / Average Precision | 0.4980 |

Best validation Average Precision:

`0.6502`

## Generated Outputs

The notebook generates the following experiment outputs:

- `tjepa_elliptic_pr_curve_corrected.png`
- `tjepa_precision_recall_table_corrected.csv`
- `tjepa_elliptic_metrics_corrected.csv`
- `tjepa_elliptic_test_predictions_corrected.csv`
- `tjepa_elliptic_classifier_corrected.pth`

## Reproducibility

The notebook sets the random seed to `42` for Python, NumPy, and PyTorch.

The experiment is designed to run in a Kaggle environment with the Elliptic
dataset available at the dataset path specified in the notebook.

The complete experiment workflow is available in:

`t-jepa-elliptic-fraud-detection(2).ipynb`

## Reference

The experiment is based on the T-JEPA repository:

https://github.com/jose-melo/t-jepa