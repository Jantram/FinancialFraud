# Fin-JEPA on IEEE-CIS Fraud Detection

This repository contains the executable notebook and documentation for adapting Fin-JEPA to the IEEE-CIS Fraud Detection dataset.

## Repository Contents

- `ieee-cis_fin-jepa.ipynb` — complete executable experiment, including Markdown documentation, code cells, diagnostics, training, feature extraction, classification, and evaluation.
- `README.md` — experiment overview and reproducibility notes.
- `RESULTS.md` — results recorded from the current notebook execution.

## Method Overview

The experiment adapts Fin-JEPA to IEEE-CIS transaction data.

The workflow is:

1. Load IEEE-CIS transaction and identity data.
2. Construct a pseudo-customer identity because the dataset does not provide a direct customer/account ID.
3. Merge transaction and identity information.
4. Sort transactions chronologically.
5. Create a chronological 70% / 15% / 15% train/validation/test split from the labeled training data.
6. Remove columns with more than 80% missing values.
7. Encode categorical variables and standardize features using training-set statistics.
8. Clip standardized features to `[-10, 10]` after identifying an extreme value in `V107`.
9. Build pseudo-customer transaction sequences.
10. Pretrain Fin-JEPA using self-supervised prediction and SIGReg regularization.
11. Extract learned representations and prediction-error features.
12. Train downstream XGBoost fraud classifiers.
13. Evaluate ROC-AUC and Average Precision on the held-out chronological test split.

## Pseudo-Customer Construction

IEEE-CIS does not provide a direct customer/account identifier suitable for sequential modeling.

The notebook constructs a pseudo-customer ID from:

- `card1`
- `card2`
- `card3`
- `card5`
- `addr1`

Transactions sharing these five attributes are treated as belonging to the same pseudo-customer.

This is an approximation and should not be interpreted as a true customer identity. A pseudo-ID can aggregate multiple real people who happen to share these attributes.

## Data Split

The public IEEE-CIS competition did not provide the original test labels for local evaluation.

Therefore, the labeled `train_transaction.csv` data is split chronologically by `TransactionDT`:

- 70% earliest transactions — training
- 15% next transactions — validation
- 15% latest transactions — test

This avoids randomly mixing earlier and later transactions across partitions.

## Data Processing

The notebook:

- merges transaction and identity information;
- drops columns with more than 80% missing values;
- label-encodes selected categorical variables;
- standardizes features using training-set statistics;
- clips standardized values to `[-10, 10]`.

An extreme value of `100,000,000` was identified in feature `V107` during preprocessing. Clipping standardized features was used to limit the effect of extreme values.

## Fin-JEPA

The notebook uses a Fin-JEPA-style encoder/predictor architecture with SIGReg regularization.

The model learns representations from pseudo-customer transaction sequences rather than treating every transaction as an independent sample.

The learned representations are later combined with prediction-error features for downstream fraud classification.

## Downstream Classification

XGBoost is used as the downstream fraud classifier.

The notebook evaluates the learned representation and prediction-error features and also contains a later stage for adding handcrafted pseudo-customer behavioral features.

The exact metrics reported in this repository should always be taken from the executed notebook output and `RESULTS.md`.

## Reproducibility

The primary reproducibility artifact is:

`ieee-cis_fin-jepa.ipynb`

The notebook contains the complete executable workflow and Markdown explanations for the individual stages.

The dataset itself is not included in this repository because the IEEE-CIS dataset is distributed through Kaggle.

## Reference Implementation

Fin-JEPA reference implementation:

https://github.com/cedricwyh/fin-jepa

## Important Evaluation Note

The IEEE-CIS experiment uses a constructed pseudo-customer identity and a chronological split of the labeled training data. Therefore, its results should not be interpreted as directly equivalent to an evaluation using the hidden Kaggle competition test set.

The repository documents the experiment as actually executed rather than presenting the hidden competition test score.
