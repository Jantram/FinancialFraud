# In-House JEPA Fraud Analysis

## Overview

This repository contains the documented, executed notebook for JEPA-based fraud detection on the in-house synthetic payment-fraud dataset. The experiment compares **Fin-JEPA** with a corrected, source-aligned **T-JEPA** baseline.

The complete notebook is:

`inhouse_jepa_analysis.ipynb`

The notebook preserves the original executable code and outputs and adds Markdown cells documenting each major stage for reproducibility.

## Dataset

The executed notebook loads:

- `label_timeline.csv`
- `sequences.csv`
- `_ground_truth.csv`
- `dataset_card.txt`

The loaded tables contain **43,716 transaction rows**. The experiment uses the dataset-provided `split` column and keeps the supplied time-based train/test separation.

The notebook's original input path is:

`/kaggle/input/datasets/abhradwiplala/in-house-dataset`

When reproducing outside Kaggle, update the path in the notebook to the location of the dataset.

## Reproducibility and split

Random seed: **42**.

Executed split:

| Split | Transactions |
|---|---:|
| Train | 22,991 |
| Test | 20,725 |

The notebook does not replace this official split with a random train/test split.

## Features and preprocessing

The model uses **35 features**:

- 23 numeric features
- 12 categorical features

The executed preprocessing produces a `43,716 × 35` feature matrix with no remaining NaN or infinite values. The leakage-safe train/test matrices are:

- `X_train`: **22,991 × 35**
- `X_test`: **20,725 × 35**

## Fin-JEPA

Fin-JEPA creates chronological customer windows with:

- Context length: **4** transactions
- Prediction length: **1** transaction
- Stride: **1**
- Minimum customer history: **3** transactions
- Embedding dimension: **64**
- Training epochs: **20**
- Learning rate: **1e-4**
- Weight decay: **1e-5**

Executed window counts:

- Train windows: **11,670**
- Test windows: **9,632**

The Fin-JEPA test population contains **107 fraud** and **9,525 non-fraud** transactions (fraud rate **1.1109%**).

### Final Fin-JEPA result

| Metric | Result |
|---|---:|
| ROC-AUC | **0.9259** |
| PR-AUC / Average Precision | **0.2104** |
| Fraud cases | **107** |
| Non-fraud cases | **9,525** |

Best training checkpoint: **epoch 19**, training loss **0.0350864**.

## T-JEPA

The notebook first contains an initial T-JEPA implementation. It explicitly identifies that implementation as architecturally incorrect. Its earlier result of **ROC-AUC 0.5230 / PR-AUC 0.0109 is not the final T-JEPA result**.

The corrected source-aligned T-JEPA uses:

- 35 features
- Embedding dimension: **128**
- 4 attention heads
- 3 encoder layers
- 2 predictor layers
- EMA decay: **0.996**
- 30 training epochs

Corrected model parameter count:

- Total parameters: **1,613,440**
- Trainable parameters: **1,013,376**

Best corrected checkpoint: **epoch 30**, training loss **0.036538**.

### Final corrected T-JEPA result

The corrected model is evaluated on the **same 9,632 transactions** used by Fin-JEPA:

| Metric | Result |
|---|---:|
| ROC-AUC | **0.7534** |
| PR-AUC / Average Precision | **0.0287** |
| Fraud cases | **107** |
| Non-fraud cases | **9,525** |

## Fin-JEPA vs T-JEPA

Both models are compared on the same 9,632-transaction population.

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| **Fin-JEPA** | **0.9259** | **0.2104** |
| **Corrected T-JEPA** | **0.7534** | **0.0287** |

The notebook also generates precision-at-recall tables and a common precision-recall curve.

## Reproducing the experiment

1. Obtain the supplied in-house dataset.
2. Place it at the expected input location or update the path variables in the notebook.
3. Open `inhouse_jepa_analysis.ipynb`.
4. Run the cells in order in a CUDA-capable environment if reproducing the original execution.
5. Keep the provided time-based `split` column.
6. Check the final common-test evaluation cells for the reported metrics.

## Repository contents

```text
inhouse-jepa-fraud-analysis/
├── .gitignore
├── README.md
└── inhouse_jepa_analysis.ipynb
```

## Important result note

Only the final executed metrics are reported as the final comparison. The notebook itself records the earlier incorrect T-JEPA result and explicitly rejects it, so it should not be presented as the final baseline.
