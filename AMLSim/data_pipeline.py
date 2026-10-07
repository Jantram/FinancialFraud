"""
AMLSim data preprocessing reference for the Fin-JEPA experiment.

NOTE:
The executable preprocessing and window-building code is contained in
`finjepa (1).ipynb`, which is the authoritative reproducibility source
for this experiment.

This file documents the exact preprocessing contract used by the notebook.

Pipeline:
1. Load AMLSim `accounts.csv`, `transactions.csv`, and `alerts.csv`.
2. Convert sent and received transactions into a single account-level
   transaction history.
3. Sort each account's history by `TIMESTAMP`.
4. Create the following model features:
   - `LOG_AMOUNT_NORM`: normalized log1p(transaction amount)
   - `DIRECTION`: 1 for sent/outgoing, 0 for received/incoming
   - `TIME_GAP_NORM`: normalized time gap since the previous transaction
5. Compute normalization statistics using training accounts only.
6. Split accounts into train/validation/test sets using a 70/15/15 split
   with `random_state=42`.
7. Build sliding windows independently for each account:
   - context length: 15 transactions
   - prediction length: 5 transactions
   - stride: 5 transactions

The notebook should be used as the executable source when reproducing
the experiment. This file is documentation only and does not implement
a second, independent preprocessing pipeline.
"""
