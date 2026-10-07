"""
Custom dataset adapter for the Elliptic Bitcoin transaction dataset.

This file documents the Elliptic-specific dataset integration used with the
T-JEPA framework.

The original public T-JEPA repository does not contain this dataset adapter.
This file was added for the Elliptic Bitcoin fraud-detection experiment.

Dataset files:
- elliptic_txs_features.csv
- elliptic_txs_classes.csv

The dataset is joined using the transaction ID (txId). Only transactions with
known labels are retained:

- 1 = Illicit / Fraud
- 2 = Licit / Legitimate

The experiment uses a chronological temporal split based on the `time_step`
field rather than a random train/test split.

Temporal split used by the current experiment:
- Training: time_step <= 34
- Validation: time_step 35-39
- Test: time_step >= 40

This file is kept in the repository to document the Elliptic-specific dataset
integration used by the project.
"""