---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 3.3: Class Imbalance and Calibration

### Module 3: Supervised ML

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Identify and quantify class imbalance in datasets
- Apply resampling (SMOTE) and class weighting strategies
- Evaluate classifiers with precision, recall, F1, and AUC-ROC
- Calibrate model probabilities for reliable confidence scores

---

## Recap: Lesson 3.2

- Gradient boosting sequentially corrects residuals with weak learners
- `TrainingPipeline` supports XGBoost and LightGBM
- Early stopping prevents overfitting
- Feature importance reveals what the model learned

---

## The Imbalance Problem

```
Fraud detection dataset:
  Normal transactions: 99,700  (99.7%)
  Fraudulent:              300  (0.3%)

A model that predicts "not fraud" for EVERYTHING gets 99.7% accuracy.
But it catches zero fraud — completely useless.
```

Accuracy is misleading when classes are imbalanced.

---

## Measuring Imbalance

```python
import polars as pl
from shared.data_loader import ASCENTDataLoader

loader = ASCENTDataLoader()
df = loader.load("ascent03", "transactions.csv")

class_counts = df.group_by("is_fraud").agg(
    pl.col("is_fraud").count().alias("count")
)
print(class_counts)

# Imbalance ratio
majority = class_counts["count"].max()
minority = class_counts["count"].min()
ratio = majority / minority
print(f"Imbalance ratio: {ratio:.0f}:1")
```

---

## Better Metrics for Imbalanced Data

```
                    Predicted
                  Positive  Negative
Actual  Positive    TP        FN
        Negative    FP        TN

Precision = TP / (TP + FP)   "Of predicted positives, how many are correct?"
Recall    = TP / (TP + FN)   "Of actual positives, how many did we catch?"
F1        = 2 · (P · R)/(P + R)  "Harmonic mean of precision and recall"
```

---

## Precision vs Recall Tradeoff

```
High threshold (conservative):
  → High precision, low recall
  → "When we flag fraud, we are usually right — but we miss many"

Low threshold (aggressive):
  → Low precision, high recall
  → "We catch most fraud, but many false alarms"

               Precision
  1.0 │╲
      │  ╲
      │    ╲          Choose based on
      │      ╲        business cost:
      │        ╲      - False alarm cost?
      │          ╲    - Missed detection cost?
  0.0 └──────────→
      0.0       1.0
              Recall
```

---

## AUC-ROC: Threshold-Free Evaluation

```
ROC Curve: True Positive Rate vs False Positive Rate

  TPR │        ╱────
      │      ╱
      │    ╱          AUC = area under the curve
      │  ╱            AUC = 0.5: random guessing
      │╱              AUC = 1.0: perfect separation
  0.0 └──────────→
      0.0       1.0
              FPR

AUC-ROC measures ranking quality across ALL thresholds.
```

---

## Strategy 1: Class Weights

Tell the model that minority class errors cost more.

```python
from kailash_ml import TrainingPipeline

pipeline = TrainingPipeline()
pipeline.configure(
    dataset=df,
    target_column="is_fraud",
    task="classification",
    algorithm="lightgbm",
    params={
        "n_estimators": 500,
        "scale_pos_weight": 332,  # ratio of neg/pos samples
    },
)

result = pipeline.run()
print(f"F1: {result.metrics['f1']:.3f}")
print(f"AUC-ROC: {result.metrics['auc_roc']:.3f}")
```

---

## Strategy 2: SMOTE

Synthetic Minority Oversampling: create synthetic minority samples.

```python
from imblearn.over_sampling import SMOTE

X_train = df_train.drop("is_fraud").to_numpy()
y_train = df_train["is_fraud"].to_numpy()

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE: {len(X_train)} samples")
print(f"After SMOTE:  {len(X_resampled)} samples")
# Minority class now has same count as majority
```

SMOTE creates new points by interpolating between existing minority samples.

---

## Strategy 3: Threshold Tuning

```python
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score

# Get predicted probabilities
y_prob = result.model.predict_proba(X_test)[:, 1]

# Find optimal threshold for F1
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Default threshold: 0.5")
print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"F1 at optimal: {f1_scores[optimal_idx]:.3f}")
```

---

## Strategy Comparison

| Strategy             | Pros                            | Cons                                |
| -------------------- | ------------------------------- | ----------------------------------- |
| **Class weights**    | Simple, no data change          | May not be enough alone             |
| **SMOTE**            | Creates useful minority samples | Can create noisy samples            |
| **Undersampling**    | Fast, reduces data size         | Loses majority class information    |
| **Threshold tuning** | No retraining needed            | Requires good probability estimates |
| **Combine all**      | Most robust                     | More complex pipeline               |

---

## Probability Calibration: Why It Matters

```
Model says: "80% probability of fraud"
Reality:    Only 60% of "80% predictions" are actually fraud

Uncalibrated models → unreliable confidence scores
  → Poor decision-making downstream
  → Cannot compare probabilities across models
```

Calibration ensures predicted probabilities match observed frequencies.

---

## Calibration Curve

```
Observed frequency
  1.0 │              ╱  Perfect calibration
      │            ╱
      │          ╱
      │        ╱  ·  ·
      │      ╱ ·       Actual model
      │    ╱·          (overconfident)
      │  ╱·
      │╱·
  0.0 └──────────────→
      0.0            1.0
      Predicted probability
```

Points above the diagonal = underconfident; below = overconfident.

---

## Platt Scaling

Fit a logistic regression on the model's raw outputs.

```python
from sklearn.calibration import CalibratedClassifierCV

# Calibrate using Platt scaling (sigmoid method)
calibrated = CalibratedClassifierCV(
    result.model,
    method="sigmoid",
    cv=5,
)
calibrated.fit(X_train, y_train)

# Now probabilities are calibrated
y_prob_calibrated = calibrated.predict_proba(X_test)[:, 1]
```

---

## Isotonic Regression Calibration

Non-parametric calibration (no distributional assumption).

```python
calibrated_iso = CalibratedClassifierCV(
    result.model,
    method="isotonic",
    cv=5,
)
calibrated_iso.fit(X_train, y_train)
```

| Method              | Assumption        | Data Needed       | Best When              |
| ------------------- | ----------------- | ----------------- | ---------------------- |
| **Platt (sigmoid)** | S-shaped mapping  | Small datasets OK | Systematic bias        |
| **Isotonic**        | Monotonic mapping | Needs more data   | Complex miscalibration |

---

## Measuring Calibration: Brier Score and ECE

```python
from sklearn.metrics import brier_score_loss

# Brier score: mean squared error of probabilities (lower = better)
brier_before = brier_score_loss(y_test, y_prob)
brier_after = brier_score_loss(y_test, y_prob_calibrated)

print(f"Brier score before calibration: {brier_before:.4f}")
print(f"Brier score after calibration:  {brier_after:.4f}")
```

Expected Calibration Error (ECE) measures the average gap between predicted probability and observed frequency across bins.

---

## Complete Imbalanced Classification Pipeline

```python
from kailash_ml import TrainingPipeline

pipeline = TrainingPipeline()
pipeline.configure(
    dataset=df,
    target_column="is_fraud",
    task="classification",
    algorithm="lightgbm",
    params={
        "n_estimators": 500,
        "learning_rate": 0.05,
        "scale_pos_weight": 332,
    },
    resampling="smote",
    calibration="isotonic",
    metrics=["f1", "auc_roc", "precision", "recall", "brier_score"],
)

result = pipeline.run()
```

---

## Exercise Preview

**Exercise 3.3: Fraud Detection with Imbalanced Data**

You will:

1. Diagnose class imbalance and choose appropriate metrics
2. Apply class weights, SMOTE, and threshold tuning
3. Calibrate model probabilities and evaluate with Brier score
4. Build a complete imbalanced classification pipeline

Scaffolding level: **Moderate (~50% code provided)**

---

## Common Pitfalls

| Mistake                                | Fix                                      |
| -------------------------------------- | ---------------------------------------- |
| Using accuracy on imbalanced data      | Use F1, AUC-ROC, precision/recall        |
| Applying SMOTE before train/test split | SMOTE only on training data              |
| Ignoring calibration                   | Always calibrate if probabilities matter |
| Same threshold for all use cases       | Tune threshold based on business costs   |
| Oversampling test set                  | Never resample the test set              |

---

## Summary

- Class imbalance makes accuracy misleading -- use F1 and AUC-ROC
- Class weights, SMOTE, and threshold tuning address imbalance
- Probability calibration ensures predicted confidence is reliable
- Platt scaling (sigmoid) and isotonic regression are the two main methods
- Brier score measures calibration quality

---

## Next Lesson

**Lesson 3.4: SHAP, LIME, and Fairness**

We will learn:

- SHAP values for global and local feature explanations
- LIME for instance-level interpretability
- Fairness metrics and bias detection in ML models
