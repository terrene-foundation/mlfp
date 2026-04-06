# Chapter 10: EnsembleEngine

## Overview

EnsembleEngine combines multiple trained models into stronger composite predictors. It provides four ensemble methods: `blend()` (voting), `stack()` (meta-learner), `bag()` (bootstrap aggregating), and `boost()` (AdaBoost). All methods accept polars DataFrames and auto-detect whether the task is classification or regression.

This chapter covers:

- Soft and hard voting with `blend()`
- Custom model weights for weighted averaging
- Stacking with a meta-learner via `stack()`
- Bootstrap aggregating with `bag()`
- Boosting with `boost()`
- Auto-detection of classification vs regression
- Component contributions and per-model metrics
- Serialization round-trips for all result types

## Prerequisites

| Requirement    | Details                  |
| -------------- | ------------------------ |
| Python         | 3.10+                    |
| kailash-ml     | `pip install kailash-ml` |
| sklearn        | For base model training  |
| Prior chapters | 1-9                      |
| Level          | Intermediate             |

## Concepts

### Ensemble Methods

| Method    | How It Works                                                  | When to Use                                     |
| --------- | ------------------------------------------------------------- | ----------------------------------------------- |
| `blend()` | Averages predictions (soft) or takes majority vote (hard)     | Quick improvement from diverse models           |
| `stack()` | Trains a meta-learner on cross-validated predictions          | When models make different types of errors      |
| `bag()`   | Trains copies of one model on bootstrap samples               | Reduce variance of high-variance models (trees) |
| `boost()` | Sequentially trains models, upweighting misclassified samples | Reduce bias, improve weak learners              |

### Soft vs Hard Voting

- **Soft voting**: Averages predicted probabilities, then picks the class with highest average probability. Works only with models that output probabilities.
- **Hard voting**: Each model votes for a class; the class with the most votes wins. Works with any classifier.

### Stacking

Stacking uses a two-level architecture:

1. **Base models** generate cross-validated predictions on the training set.
2. **Meta-model** (e.g., LogisticRegression) learns to combine base predictions optimally.

The meta-model class must be from the allowlist (sklearn prefix).

### Task Auto-Detection

EnsembleEngine examines the target column. If it has few unique integer values, the task is classification. If it has many unique continuous values, the task is regression. This determines which metrics are computed.

## Key API Reference

| Method                                                                                      | Purpose           |
| ------------------------------------------------------------------------------------------- | ----------------- |
| `engine.blend(models, data, target, method, weights, test_size, seed)`                      | Voting ensemble   |
| `engine.stack(models, data, target, meta_model_class, fold, test_size, seed)`               | Stacking ensemble |
| `engine.bag(model, data, target, n_estimators, max_samples, max_features, test_size, seed)` | Bagging ensemble  |
| `engine.boost(model, data, target, n_estimators, learning_rate, test_size, seed)`           | Boosting ensemble |

### Result Types

| Type          | Key Fields                                                                                          |
| ------------- | --------------------------------------------------------------------------------------------------- |
| `BlendResult` | `n_models`, `method`, `metrics`, `weights`, `component_contributions`, `ensemble_model`             |
| `StackResult` | `meta_model_class`, `n_base_models`, `fold`, `metrics`, `component_contributions`, `ensemble_model` |
| `BagResult`   | `n_estimators`, `max_samples`, `max_features`, `metrics`, `base_model_class`, `ensemble_model`      |
| `BoostResult` | `n_estimators`, `learning_rate`, `metrics`, `base_model_class`, `ensemble_model`                    |

## Code Walkthrough

### Step 1: Train Base Models

```python
import polars as pl
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

feature_cols = ["feat_a", "feat_b", "feat_c"]
X = df.select(feature_cols).to_numpy()
y = df["target"].to_numpy()

rf = RandomForestClassifier(n_estimators=20, random_state=42).fit(X, y)
gb = GradientBoostingClassifier(n_estimators=20, random_state=42).fit(X, y)
lr = LogisticRegression(max_iter=200, random_state=42).fit(X, y)
dt = DecisionTreeClassifier(random_state=42).fit(X, y)
```

### Step 2: Blend (Soft Voting)

```python
from kailash_ml.engines.ensemble import EnsembleEngine

engine = EnsembleEngine()
blend_result = engine.blend(
    models=[rf, gb, lr],
    data=df, target="target",
    method="soft", test_size=0.2, seed=42,
)

assert blend_result.n_models == 3
assert "accuracy" in blend_result.metrics
```

### Step 3: Blend with Custom Weights

```python
weighted_blend = engine.blend(
    models=[rf, gb, lr],
    data=df, target="target",
    weights=[2.0, 3.0, 1.0],  # Weight GradientBoosting highest
    method="soft", test_size=0.2, seed=42,
)
```

### Step 4: Stack with Meta-Learner

```python
stack_result = engine.stack(
    models=[rf, gb, lr],
    data=df, target="target",
    meta_model_class="sklearn.linear_model.LogisticRegression",
    fold=3, test_size=0.2, seed=42,
)

assert stack_result.n_base_models == 3
assert stack_result.fold == 3
```

### Step 5: Bag

```python
bag_result = engine.bag(
    model=dt, data=df, target="target",
    n_estimators=15, max_samples=0.8,
    max_features=0.9, test_size=0.2, seed=42,
)

assert bag_result.n_estimators == 15
```

### Step 6: Boost

```python
boost_result = engine.boost(
    model=dt, data=df, target="target",
    n_estimators=30, learning_rate=0.1,
    test_size=0.2, seed=42,
)

assert boost_result.learning_rate == 0.1
```

### Step 7: Regression Auto-Detection

```python
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=20, random_state=42)
rfr.fit(X_reg, y_reg)

reg_blend = engine.blend(
    models=[rfr], data=regression_df,
    target="price", method="soft",
    test_size=0.2, seed=42,
)
# Regression metrics: r2, rmse, mae
```

## Common Mistakes

| Mistake                          | What Happens     | Fix                                 |
| -------------------------------- | ---------------- | ----------------------------------- |
| Empty model list for blend/stack | `ValueError`     | Provide at least one model          |
| Invalid blend method             | `ValueError`     | Use `"soft"` or `"hard"`            |
| Weight length mismatch           | `ValueError`     | Match weights count to models count |
| Non-allowlisted meta_model_class | `ValueError`     | Use sklearn-prefixed classes        |
| Passing unfitted models          | `AttributeError` | Call `.fit()` before ensembling     |

## Exercises

1. **Blend vs stack**: Compare soft blending and stacking on the same base models. Which produces better accuracy? Why might stacking outperform blending?

2. **Weight optimization**: Try different weight combinations for `blend()`. Can you find weights that outperform equal weighting?

3. **Bag depth analysis**: Run `bag()` with n_estimators = 5, 10, 25, 50. Plot n_estimators vs accuracy. At what point do returns diminish?

## Key Takeaways

- Four ensemble methods (blend, stack, bag, boost) cover the major combination strategies.
- Soft voting averages probabilities; hard voting takes majority vote.
- Stacking learns optimal model combination via a meta-learner.
- Bagging reduces variance; boosting reduces bias.
- Task type (classification vs regression) is auto-detected from the target.
- Component contributions show per-model performance within the ensemble.

## Next Chapter

Chapter 11 covers **ModelRegistry** -- the engine that manages versioned model artifacts with lifecycle promotion (staging, shadow, production, archived).
