# Results — Fin-JEPA on IEEE-CIS

## Current Notebook Evaluation

The current committed notebook contains an executed final evaluation of the Fin-JEPA + XGBoost experiment.

The recorded final test output is:

| Metric | Current notebook result |
|---|---:|
| ROC-AUC | **0.8174** |
| Average Precision | **0.7386** |

The unrounded values recorded by the notebook are:

- ROC-AUC: `0.8174411288555209`
- Average Precision: `0.7385614726327865`

The test-set verification in the notebook reports:

- Test samples: `3,315`
- Normal: `2,106`
- Fraud: `1,209`
- Minimum predicted probability: `0.03636`
- Maximum predicted probability: `0.98318`
- Mean predicted probability: `0.36389`

The notebook also verifies that prediction and label lengths match, labels are binary, and prediction probabilities are finite.

## Downstream Model

The final recorded downstream experiment uses XGBoost with the Fin-JEPA-derived representation and prediction-error features.

The notebook reports:

- `scale_pos_weight = 3.94`
- 500 boosting estimators
- Validation AUC increases through training and reaches approximately `0.8443` at the final displayed boosting iteration.

The exact final held-out test metrics are the values reported above.

## Feature Importance

The current notebook's final feature-importance output begins with:

1. `error_max`
2. `error_mean`
3. `model_fp_60`
4. `model_fp_61`
5. `model_fp_14`

These are XGBoost feature importances and represent predictive association within the trained classifier; they should not be interpreted as causal effects.

## Earlier Recorded Result

An earlier version of the experiment was documented with approximately:

- ROC-AUC: `0.894`
- Average Precision: `0.852`

Those numbers are **not the current final output recorded in the committed notebook** and should not be presented as the current repository result without rerunning and validating that experiment.

This distinction is intentional so that `RESULTS.md` remains consistent with the executable notebook currently stored in the repository.

## Data and Evaluation Caveats

### Pseudo-customer identity

The pseudo-customer ID is constructed from:

`card1 + card2 + card3 + card5 + addr1`

It is not a ground-truth customer identifier and may combine multiple real users.

### Evaluation split

IEEE-CIS competition test labels are not publicly available for local evaluation. The notebook therefore uses a chronological 70/15/15 split of the labeled training data based on `TransactionDT`.

### Data quality

An extreme value (`100,000,000`) was identified in `V107`. The experiment clips standardized features to `[-10, 10]` to reduce the influence of extreme values.

### Comparability

IEEE-CIS uses a constructed pseudo-customer-level formulation, so its metrics should not be treated as a directly apples-to-apples comparison with AMLSim account-level or Elliptic transaction-level evaluation.

## Reproducibility

The executable source is:

`ieee-cis_fin-jepa.ipynb`

The notebook contains the complete code, Markdown documentation, diagnostics, training stages, feature extraction, and evaluation.
