"""
Data pipeline for AMLSim -> Fin-JEPA input format.

The original repo's data.py (stock price windowing) is not public. This file is a from-scratch
replacement, following the same (context, target) windowing contract the model expects, applied
to bank-account transaction histories instead of daily stock prices.

Entity analogy: ACCOUNT (AMLSim) <-> STOCK TICKER (original Fin-JEPA)
Sequence analogy: transaction, ordered by TIMESTAMP <-> daily price row, ordered by date

Pipeline steps:
1. Combine sent + received transactions per account into one time-ordered history
   (direction flag distinguishes outgoing vs incoming).
2. Feature engineering: log1p(amount), direction (0/1), time gap since previous transaction.
   Standardized using TRAIN-split statistics only.
3. Account-level train/val/test split (70/15/15) -- NOT time-based, to prevent leakage of the
   same account's future into training.
4. Sliding window per account: context_len=15, pred_len=5, stride=5.
