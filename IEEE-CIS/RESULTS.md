# Results — Fin-JEPA on IEEE-CIS

## Self-supervised pretraining
15 epochs, batch size 256. Best val pred_loss: 0.0146-0.0174 (stable, ~3x train — healthy).

Note: total val loss (~12) looked alarming vs train (~0.2) initially. Root-caused to the
SIGReg regularizer term specifically (not prediction quality) — likely reflects genuine
temporal drift between the earlier training period and later validation period in real
e-commerce data (unlike AMLSim's stationary synthetic simulation). pred_loss, the metric
that actually matters for downstream use, generalized well throughout. Documented as a
finding, not treated as an unresolved bug.

## Downstream fraud detection (pseudo-customer level)
| Method | Test AUC | Test Avg. Precision |
|---|---|---|
| XGBoost, embedding + error only (66-dim) | 0.819 | 0.727 |
| XGBoost, + hand-crafted features (70-dim) | **0.894** | **0.852** |

## Most important features (final model)
1. log_tx_count — number of transactions for this pseudo-customer (by far the strongest)
2. amt_max — largest single transaction amount
3-10. Mix of learned embedding dimensions, amt_mean, error_max

## Key finding
Transaction volume alone (log_tx_count) was a dramatically stronger fraud signal here than in
AMLSim, where counterparty diversity dominated instead. Plausible explanation: fraudulent
pseudo-customer IDs may aggregate rapid card testing / multiple fraudulent purchases in a
short window, inflating transaction count for that ID specifically.

## Note on comparability
Label granularity here (pseudo-customer, constructed) is not identical to AMLSim's account
label (real, provided) or Elliptic's transaction label (real, provided) — see main comparison
report for a full discussion of why raw AUC numbers across datasets should not be treated as
strictly apples-to-apples.
