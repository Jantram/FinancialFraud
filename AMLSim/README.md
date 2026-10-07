# Fin-JEPA on AMLSim

This repository contains the AMLSim fraud-detection experiment using a
Fin-JEPA-based self-supervised representation-learning approach.

The experiment adapts the sequence-learning setup of the public Fin-JEPA
repository to the IBM AMLSim synthetic anti-money-laundering dataset.

Original Fin-JEPA repository:
https://github.com/cedricwyh/fin-jepa

## Repository contents

| File | Purpose |
|---|---|
| `finjepa (1).ipynb` | **Main executable experiment.** Contains dataset loading, preprocessing, account splitting, window construction, Fin-JEPA pretraining, downstream fraud detection, XGBoost evaluation, and result generation. |
| `model.py` | Fin-JEPA model implementation used by the experiment. |
| `data_pipeline.py` | Documentation of the AMLSim preprocessing and windowing contract used in the notebook. |
| `RESULTS.md` | Reported experiment results and evaluation details. |

## Dataset

The experiment uses the IBM AMLSim synthetic anti-money-laundering dataset.

The notebook expects these AMLSim files:

- `accounts.csv`
- `transactions.csv`
- `alerts.csv`

The dataset itself is not included in this repository.

The notebook was originally run in a Kaggle environment and uses the dataset path configured there. When reproducing the experiment in another environment, update the dataset path in the notebook accordingly.

## AMLSim preprocessing

Transactions are converted into account-level time-ordered histories.

For each transaction:

- outgoing/sent transactions are assigned `DIRECTION = 1`
- incoming/received transactions are assigned `DIRECTION = 0`
- transaction amount is transformed using `log1p`
- the time gap from the previous transaction is calculated

The amount and time-gap features are standardized using statistics calculated from the training accounts only.

The model uses three input features:

1. normalized log transaction amount
2. transaction direction
3. normalized time gap

## Train / validation / test split

The split is performed at the **account level**, not at the individual transaction level.

Accounts are divided using:

- 70% training
- 15% validation
- 15% test
- `random_state = 42`

This keeps the same account from appearing across different splits.

## Sequence windowing

Each account's transaction history is divided into sliding windows:

- Context length: 15 transactions
- Prediction length: 5 transactions
- Stride: 5 transactions

Only accounts/windows satisfying the required sequence length are used.

## Fin-JEPA configuration

The AMLSim experiment uses:

- embedding dimension: 64
- encoder layers: 3
- predictor layers: 4
- predictor heads: 4
- SIGReg projections: 128
- SIGReg weight: 0.1
- batch size: 256
- training epochs: 20
- optimizer: AdamW
- learning rate: `1e-3`
- learning-rate schedule: cosine annealing
- random seed: 42

The best model is selected using validation loss.

## Downstream fraud detection

The learned representations are evaluated for account-level fraud detection using several downstream approaches.

The experiment includes:

1. MLP using the 64-dimensional Fin-JEPA embedding
2. MLP using the embedding plus prediction error
3. XGBoost using Fin-JEPA features, prediction-error features, and handcrafted account-level features

The final XGBoost feature vector contains 76 features.

See `RESULTS.md` for the exact reported metrics.

## Results

The final hybrid XGBoost model achieved:

- Test ROC-AUC: **0.8574**
- Test Average Precision: **0.6775**

The test set contains 1,497 accounts, including 247 fraud accounts.

The hybrid result combines learned Fin-JEPA features with handcrafted account-level statistics. Therefore, the result should not be interpreted as measuring the contribution of Fin-JEPA alone. Additional ablation experiments would be required for that comparison.

## Reproducibility

For reproduction, start with the jupyter notebook.

The notebook is the authoritative executable source for this experiment. It contains the complete AMLSim data preparation, model training, downstream evaluation, and result-generation workflow.

`data_pipeline.py` is provided as a documentation/reference file rather than as a second independent implementation of the preprocessing pipeline.

## Reference

Fin-JEPA:

https://github.com/cedricwyh/fin-jepa
