# In-house Fin-JEPA and T-JEPA Fraud Analysis

Self-supervised fraud detection on the in-house transaction dataset using two JEPA-based anomaly scorers trained entirely without fraud labels.

---

## Overview

Two complementary architectures are trained and evaluated:

**Fin-JEPA** (sequence-level) — encodes customer transaction histories as sliding windows and trains a causal Transformer to predict the latent representation of the next transaction. Prediction error at inference time is the anomaly score.

**T-JEPA** (feature-level) — treats a single transaction as a set of feature tokens and learns to reconstruct masked features from visible context features. Reconstruction error over target features is the anomaly score.

Both models are trained on train-split data with no fraud labels. Evaluation uses the official test split against held-out ground truth labels across four fraud typologies: ring, ATO (account takeover), emerge, and normal.

---

## Dataset

The notebook expects the in-house dataset from Kaggle at the path below. Edit `BASE` / `DATA_DIR` in the first cells if running elsewhere.

```
/kaggle/input/datasets/abhradwiplala/in-house-dataset/
    sequences.csv          # one row per transaction; features + seq_pos + customer_id
    label_timeline.csv     # noisy real-time labels (EFW, review, dispute, refund)
    _ground_truth.csv      # held-out true fraud labels + typology
```

`label_timeline` is intentionally partial and noisy — it reflects what a real system sees at transaction time. The gap between its positive rate and the ground-truth fraud rate is the core modelling challenge the dataset is designed to test.

---

## Notebook structure

### Part 1: Data preparation (Steps 1–5)
- Load and merge the three source tables
- Build a combined `train_label` from all noisy label channels
- Encode categoricals, impute and standardise numeric features using train-split statistics only
- Construct Fin-JEPA sliding windows (`context_len=4`, `pred_len=1`, stride 1)
- Verify ID mapping, official train/test split, window counts, and ground-truth alignment

### Part 2: Fin-JEPA training and evaluation (Steps 6–11)
- `PriceEncoder`: per-transaction MLP encoder (features → 64-dim embedding)
- `TransformerPredictor`: 4-layer causal Transformer predicts the next transaction's latent
- `SIGReg`: Spectral Information Geometry regulariser prevents embedding collapse
- `FinJEPA`: combines encoder, predictor, frozen EMA target encoder, and SIGReg
- Training: 20 epochs, AdamW + cosine LR schedule, gradient clipping
- Evaluation: ROC-AUC, PR-AUC, Precision/Recall@K, precision at fixed recall, per-typology breakdown

### Part 3: T-JEPA training and evaluation (Steps 12–22)
- `FeatureTokenizer`: converts each scalar feature to a 128-dim token (value projection + learned position embedding)
- `ContextEncoder`: Transformer over visible (context) feature tokens
- `TargetEncoder`: EMA copy of context encoder; frozen, no gradients
- `TargetPredictor`: predicts target token representations from context summary + feature identity
- `FeatureMasker`: generates disjoint context (75–85%) / target (15–25%) feature masks per sample
- `TJEPA`: assembles all components; EMA decay increases 0.996 → 0.999 over 30 epochs
- Anomaly score: mean cosine error between predicted and EMA target tokens for masked features

Note: the original `TargetPredictor` (Step 13) contains a dimension bug. Step 14A replaces it with a corrected version before training begins.

### Part 4: Joint evaluation on the common test set (Steps 24–29)
- Both models are compared on exactly 9,632 transactions (the Fin-JEPA window targets)
- Metrics: ROC-AUC, PR-AUC, Precision/Recall@K, precision at fixed recall
- Per-typology evaluation against a clean normal-non-fraud reference set

### Part 5: Typology and failure analysis (Steps 28–29, emerge section)
- Emerge typology analysis: train vs. test distribution, timeline, entity overlap
- Failure grouping: each fraud case labelled as detected by both / Fin-JEPA only / T-JEPA only / both missed
- Feature distribution comparison between jointly-missed and detected fraud cases

### Part 6: Deterministic T-JEPA scoring and feature sensitivity (Steps 28A–28J)
- Fixed evaluation masks (seed 12345) enable reproducible scores across runs
- `score_tjepa_fixed`: scoring function that accepts pre-generated masks
- Permutation sensitivity: each feature is replaced with its training-set median baseline; mean absolute score change measures reliance on that feature
- Sensitivity is compared between jointly-missed and detected fraud cases to identify which features distinguish them

---

## Key design decisions

**No label leakage.** All normalisation statistics (mean, std, median) are computed on the train split and applied to the test split. Ground-truth labels are used only for final evaluation, never during training.

**Anomaly score semantics.** Higher score = harder to predict / reconstruct = more anomalous. Neither model is given any fraud signal during training.

**Common evaluation set.** Fin-JEPA operates on windows (one score per window, not per transaction). The final comparison uses only transactions that appear as Fin-JEPA window targets, giving both models an identical set of 9,632 scored transactions.

**EMA target encoder.** Both models use a momentum copy of the encoder as a stable prediction target. This is standard JEPA practice to prevent representation collapse without contrastive negatives.

---

## Requirements

```
pandas
numpy
torch
scikit-learn
```

The notebook was developed on the Kaggle Python Docker image (CUDA available). CPU execution will work but training will be significantly slower.

---

## Running the notebook

Run cells top to bottom. The notebook is self-contained; no manual checkpointing or intermediate saves are required. The diagnostic sections (Steps 4A, 14B, 14D, 27A, and the diagnostics block) can be skipped on re-runs once the models are trained — they exist to verify state at development time.

To use the notebook outside Kaggle, update `BASE` and `DATA_DIR` near the top of Part 1 to point to the local dataset directory.
