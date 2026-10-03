# Fin-JEPA on IEEE-CIS Fraud Detection

Adaptation of [cedricwyh/fin-jepa](https://github.com/cedricwyh/fin-jepa) to the IEEE-CIS
Fraud Detection dataset (Kaggle competition). See fin-jepa-amlsim for the base model code
(model.py is unchanged from that repo).

## Key adaptation: pseudo-customer identity
IEEE-CIS has no real customer/account ID. Following common practice from the original Kaggle
competition, we construct a pseudo-customer ID from (card1, card2, card3, card5, addr1) --
transactions sharing all five values are treated as belonging to the same underlying
card/customer. This lets Fin-JEPA build per-entity sequences the same way it does for AMLSim
accounts. Limitation: very high-count pseudo-IDs (max ~9,900 transactions) likely represent
several different real people who happen to share these attributes, not one true customer --
flagged honestly as a source of label noise, not hidden.

## Official split
No public test-set labels exist for IEEE-CIS (Kaggle competition, hidden leaderboard scoring).
Used a chronological split of the labeled train_transaction.csv instead: 70% earliest / 15% /
15% latest by TransactionDT. This is the closest available approximation to an "official" split
given the constraint, and avoids the unfairness of a random split (train accidentally containing
future information relative to test).

## Data pipeline notes
- Dropped 55 columns >80% missing.
- Categorical columns label-encoded, then standardized (mean/std) -- this was NOT done in an
  earlier iteration and caused a large train/val loss gap during self-supervised pretraining
  (see RESULTS.md, "SIGReg diagnostic" note).
- All standardized features clipped to +/-10 std devs after discovering a corrupted/outlier
  value in column V107 (100,000,000 -- clearly not a genuine value) that was distorting
  training. Clipping resolved this without needing to hand-identify every bad column.

See RESULTS.md for full metrics.
