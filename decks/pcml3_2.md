---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 3.2: Gradient Boosting

### Module 3: Supervised ML

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain how gradient boosting builds ensembles of weak learners
- Train XGBoost and LightGBM models via `TrainingPipeline`
- Tune key hyperparameters for tree-based models
- Interpret training curves to diagnose over/underfitting

---

## Recap: Lesson 3.1

- Prediction error = bias squared + variance + noise
- L2 (Ridge) shrinks weights; L1 (Lasso) zeros out irrelevant ones
- Elastic Net combines both for robustness
- Regularisation = MAP estimation with shrinkage priors

---

## Ensemble Methods: The Big Picture

```
Single model:     One expert makes the decision
                  → Brittle, high variance

Bagging:          Many experts vote independently (Random Forest)
                  → Reduces variance

Boosting:         Each expert fixes the previous expert's mistakes
                  → Reduces bias AND variance
```

Gradient boosting is the most powerful general-purpose ML algorithm.

---

## How Gradient Boosting Works

```
Step 1: Fit a simple model → get residuals (errors)
Step 2: Fit a new model to the RESIDUALS
Step 3: Add the new model to the ensemble
Step 4: Repeat until residuals are small

Prediction = Model₁ + η·Model₂ + η·Model₃ + ... + η·Modelₙ

η (learning rate) = how much each new model contributes
```

Each tree corrects the mistakes of all previous trees combined.

---

## Gradient Boosting Visual

```
Iteration 1:    Iteration 2:    Iteration 3:
  ┌──┐            ┌──┐            ┌──┐
  │T₁│            │T₁│+η·        │T₁│+η·T₂+η·
  └──┘            └──┘  ┌──┐     └──┘     ┌──┐
Residuals:              │T₂│              │T₃│
  large                 └──┘              └──┘
                  Residuals:      Residuals:
                    smaller         smallest
```

Trees are typically shallow (depth 3-8), each capturing a small pattern.

---

## XGBoost vs LightGBM

| Aspect               | XGBoost               | LightGBM                       |
| -------------------- | --------------------- | ------------------------------ |
| Tree growth          | Level-wise (balanced) | Leaf-wise (greedy)             |
| Speed                | Fast                  | Faster (especially large data) |
| Accuracy             | Excellent             | Excellent                      |
| Memory               | Moderate              | Lower                          |
| Categorical handling | Requires encoding     | Native support                 |
| Default choice       | Established standard  | Modern preference              |

Both are available through Kailash `TrainingPipeline`.

---

## TrainingPipeline: Your First Model

```python
from kailash_ml import TrainingPipeline

pipeline = TrainingPipeline()
pipeline.configure(
    dataset=df,
    target_column="price",
    task="regression",
    algorithm="xgboost",
    test_size=0.2,
    random_state=42,
)

result = pipeline.run()
print(f"RMSE: ${result.metrics['rmse']:,.0f}")
print(f"R²:   {result.metrics['r2']:.3f}")
```

---

## TrainingPipeline with LightGBM

```python
pipeline = TrainingPipeline()
pipeline.configure(
    dataset=df,
    target_column="price",
    task="regression",
    algorithm="lightgbm",

    # Key hyperparameters
    params={
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },

    test_size=0.2,
    random_state=42,
)

result = pipeline.run()
```

---

## Key Hyperparameters

| Parameter           | What It Controls   | Too Low          | Too High                         |
| ------------------- | ------------------ | ---------------- | -------------------------------- |
| `n_estimators`      | Number of trees    | Underfitting     | Overfitting (use early stopping) |
| `learning_rate`     | Step size per tree | Underfitting     | Overfitting                      |
| `max_depth`         | Tree complexity    | Underfitting     | Overfitting                      |
| `min_child_samples` | Minimum leaf size  | Overfitting      | Underfitting                     |
| `subsample`         | Row sampling ratio | High variance    | No regularisation                |
| `colsample_bytree`  | Feature sampling   | Missing patterns | No regularisation                |

---

## The Learning Rate / N-Estimators Tradeoff

```
High learning rate (0.3) + few trees (100):
  → Fast training, but may miss subtle patterns

Low learning rate (0.01) + many trees (5000):
  → Better generalisation, but slow training

Sweet spot: learning_rate=0.05, n_estimators=500-2000
  + early stopping to find the right number of trees
```

Rule of thumb: lower the learning rate, increase the trees, use early stopping.

---

## Early Stopping

```python
pipeline = TrainingPipeline()
pipeline.configure(
    dataset=df,
    target_column="price",
    task="regression",
    algorithm="xgboost",
    params={
        "n_estimators": 5000,       # high ceiling
        "learning_rate": 0.05,
        "early_stopping_rounds": 50, # stop if no improvement for 50 rounds
    },
    validation_size=0.15,
)

result = pipeline.run()
print(f"Best iteration: {result.best_iteration}")
print(f"Stopped at: {result.n_iterations} / 5000")
```

---

## Training Curves

```
Error
  |
  |╲
  | ╲  Training error
  |  ╲───────────────────────  (keeps decreasing)
  |
  |    ╲
  |     ╲  Validation error
  |      ╲────╱──────────────  (U-shape → overfitting)
  |           ↑
  |     Early stopping point
  └────────────────────────→ Iterations
```

Early stopping catches the moment before overfitting begins.

---

## Feature Importance

```python
# TrainingPipeline provides feature importance
importance = result.feature_importance

# Display top features
import polars as pl
imp_df = pl.DataFrame({
    "feature": importance.keys(),
    "importance": importance.values(),
}).sort("importance", descending=True)

print(imp_df.head(10))
```

```
┌────────────────┬────────────┐
│ feature        ┆ importance │
│ floor_area     ┆ 0.285      │
│ lease_years    ┆ 0.192      │
│ town_encoded   ┆ 0.156      │
│ storey         ┆ 0.134      │
└────────────────┴────────────┘
```

---

## Classification with TrainingPipeline

```python
pipeline = TrainingPipeline()
pipeline.configure(
    dataset=df,
    target_column="is_premium",    # binary target
    task="classification",
    algorithm="lightgbm",
    params={
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
    },
)

result = pipeline.run()
print(f"Accuracy:  {result.metrics['accuracy']:.3f}")
print(f"F1 Score:  {result.metrics['f1']:.3f}")
print(f"AUC-ROC:   {result.metrics['auc_roc']:.3f}")
```

---

## Cross-Validated Training

```python
pipeline = TrainingPipeline()
pipeline.configure(
    dataset=df,
    target_column="price",
    task="regression",
    algorithm="xgboost",

    # 5-fold cross-validation
    cv_folds=5,

    params={
        "n_estimators": 500,
        "learning_rate": 0.05,
    },
)

result = pipeline.run()
print(f"CV RMSE: ${result.cv_metrics['rmse_mean']:,.0f} "
      f"(+/- ${result.cv_metrics['rmse_std']:,.0f})")
```

---

## Exercise Preview

**Exercise 3.2: HDB Price Prediction with Gradient Boosting**

You will:

1. Train XGBoost and LightGBM models via `TrainingPipeline`
2. Implement early stopping and interpret training curves
3. Tune learning rate and tree depth to find the sweet spot
4. Compare feature importance across model types

Scaffolding level: **Moderate (~50% code provided)**

---

## Common Pitfalls

| Mistake                            | Fix                                     |
| ---------------------------------- | --------------------------------------- |
| No early stopping with many trees  | Always set `early_stopping_rounds`      |
| Learning rate too high             | Start at 0.05, decrease if overfitting  |
| Ignoring feature importance        | Check if top features make domain sense |
| Not using cross-validation         | Single train/test split is noisy        |
| Encoding categoricals for LightGBM | LightGBM handles them natively          |

---

## Summary

- Gradient boosting sequentially corrects residuals with weak learners
- XGBoost and LightGBM are the two leading implementations
- `TrainingPipeline` provides a consistent API for both
- Key tradeoff: learning rate vs number of estimators
- Early stopping prevents overfitting automatically
- Feature importance reveals what the model learned

---

## Next Lesson

**Lesson 3.3: Class Imbalance and Calibration**

We will learn:

- Handling imbalanced datasets (SMOTE, class weights)
- Probability calibration for reliable confidence scores
- Evaluation metrics beyond accuracy
