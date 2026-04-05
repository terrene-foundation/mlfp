---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 3.1: Bias-Variance and Regularisation

### Module 3: Supervised ML

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Decompose prediction error into bias, variance, and irreducible noise
- Explain why model complexity creates a tradeoff
- Apply L1 (Lasso) and L2 (Ridge) regularisation geometrically
- Connect regularisation to MAP estimation from Module 2

---

## Recap: Module 2

- Bayesian thinking, MLE/MAP estimation, credible intervals
- Hypothesis testing, bootstrap, CUPED variance reduction
- Causal inference with DiD and propensity matching
- Feature engineering and storage with FeatureEngineer and FeatureStore

Module 3 builds **predictive models** on this statistical foundation.

---

## The Prediction Error Decomposition

Every prediction error has three components:

```
Total Error = Bias² + Variance + Irreducible Noise

Bias:      How far off is the model ON AVERAGE?
           (systematic error — wrong assumptions)

Variance:  How much does the model CHANGE with different data?
           (sensitivity to training sample)

Noise:     Randomness in the data itself
           (cannot be reduced by any model)
```

---

## Bias-Variance Visual

```
         High Bias          Low Bias
         Low Variance        High Variance
        ┌───────────┐      ┌───────────┐
        │  · · · ·  │      │·    ·     │
        │   · · ·   │      │      ◎    │
        │  · ◎· ·   │      │  ·     ·  │
        │   · · ·   │      │     ·     │
        └───────────┘      └───────────┘
        Underfitting        Overfitting
        (too simple)        (too complex)

        ◎ = true target    · = predictions from different training sets
```

---

## The Complexity Tradeoff

```
Error
  |
  |╲                          ╱
  | ╲  Total Error           ╱
  |  ╲        ╱╲            ╱
  |   ╲      ╱  ╲          ╱  Variance
  |    ╲    ╱    ╲        ╱
  |     ╲──╱──────╲──────╱
  |      ╲╱        ╲    ╱
  |       ·  Sweet   ╲╱
  |      ╱   Spot
  |    ╱  Bias²
  |  ╱
  └──────────────────────→ Model Complexity
     Simple                Complex
```

The sweet spot: complex enough to capture patterns, simple enough to generalise.

---

## Underfitting vs Overfitting

|                    | Underfitting                 | Overfitting                        |
| ------------------ | ---------------------------- | ---------------------------------- |
| **Bias**           | High                         | Low                                |
| **Variance**       | Low                          | High                               |
| **Training error** | High                         | Low                                |
| **Test error**     | High                         | High                               |
| **Fix**            | More features, complex model | Regularisation, more data          |
| **Example**        | Linear model on curved data  | 100-degree polynomial on 10 points |

---

## Regularisation: Constraining Complexity

Add a **penalty** for large model weights to the loss function.

```
Standard loss:     L(w) = Σ(y_i - ŷ_i)²

Regularised loss:  L(w) = Σ(y_i - ŷ_i)² + λ · penalty(w)

λ = regularisation strength
  λ = 0:   no regularisation (standard model)
  λ → ∞:   all weights shrink to zero (extreme regularisation)
```

---

## L2 Regularisation (Ridge)

Penalty = sum of **squared** weights.

```
L_ridge(w) = Σ(y_i - ŷ_i)² + λ · Σ w_j²
```

Effect: shrinks all weights **toward zero** but rarely exactly to zero.

```
Geometric view:

  w₂ │     ╱  Loss contours (ellipses)
     │   ╱
     │ ╱  ○  ← L2 constraint (circle)
     │╱  ╱
  ───●──╱────── w₁
     │╱
     │
```

The optimal point is where the loss contour touches the circle.

---

## L1 Regularisation (Lasso)

Penalty = sum of **absolute** weights.

```
L_lasso(w) = Σ(y_i - ŷ_i)² + λ · Σ |w_j|
```

Effect: shrinks some weights **exactly to zero** (automatic feature selection).

```
Geometric view:

  w₂ │     ╱  Loss contours (ellipses)
     │   ╱
     │ ╱◇     ← L1 constraint (diamond)
     │╱╱
  ───●────────── w₁
     │
     │
```

The diamond has corners on the axes -- solutions often hit a corner (w_j = 0).

---

## L1 vs L2: When to Use Each

| Aspect                  | L1 (Lasso)                              | L2 (Ridge)                            |
| ----------------------- | --------------------------------------- | ------------------------------------- |
| **Effect on weights**   | Some become exactly 0                   | All shrink, none reach 0              |
| **Feature selection**   | Yes (built-in)                          | No                                    |
| **Correlated features** | Picks one, drops others                 | Keeps all, shares weight              |
| **Best for**            | Sparse models, many irrelevant features | Dense models, all features contribute |
| **Elastic Net**         | Combines L1 + L2 for best of both       |                                       |

---

## Elastic Net: The Best of Both

```
L_elastic(w) = Σ(y_i - ŷ_i)² + λ₁ · Σ|w_j| + λ₂ · Σw_j²
```

```python
# In practice, controlled by two parameters:
# alpha = overall regularisation strength
# l1_ratio = mix between L1 and L2

# l1_ratio = 1.0 → pure Lasso
# l1_ratio = 0.0 → pure Ridge
# l1_ratio = 0.5 → equal mix
```

Elastic Net handles correlated features better than pure Lasso.

---

## Connection to MAP Estimation

From Lesson 2.2, recall:

```
MAP = MLE + Prior

Ridge (L2) = MLE + Normal prior on weights
  → Prior: w ~ Normal(0, 1/λ)

Lasso (L1) = MLE + Laplace prior on weights
  → Prior: w ~ Laplace(0, 1/λ)
```

Regularisation is not a hack -- it is principled Bayesian inference.

---

## Cross-Validation for λ

How to choose the regularisation strength?

```python
from sklearn.linear_model import RidgeCV, LassoCV
import numpy as np

# RidgeCV: tests multiple λ values with cross-validation
alphas = np.logspace(-4, 4, 50)

ridge = RidgeCV(alphas=alphas, cv=5)
ridge.fit(X_train, y_train)

print(f"Best λ: {ridge.alpha_:.4f}")
print(f"Test R²: {ridge.score(X_test, y_test):.3f}")
```

Cross-validation finds the λ that minimises held-out error.

---

## Regularisation Path

```
Coefficient
  |  w₁ ────────────────────╲
  |                           ╲
  |  w₂ ──────────╲           ╲──────
  |                 ╲──────────
  |  w₃ ──────╲
  |             ╲──── 0 ───────────────
  |  w₄ ─ 0 ──────────────────────────
  └────────────────────────────────→ λ
    0 (no reg)                   ∞ (max reg)
```

As λ increases, weights shrink. Lasso drives them to zero; Ridge does not.

---

## Full Example with Polars

```python
import polars as pl
from shared.data_loader import ASCENTDataLoader
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

loader = ASCENTDataLoader()
df = loader.load("ascent03", "hdbprices_engineered.csv")

feature_cols = [c for c in df.columns if c != "price"]
X = df.select(feature_cols).to_numpy()
y = df["price"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

for model_cls, name in [(Ridge, "Ridge"), (Lasso, "Lasso")]:
    model = model_cls(alpha=1.0)
    model.fit(X_train_s, y_train)
    score = model.score(X_test_s, y_test)
    n_zero = (model.coef_ == 0).sum()
    print(f"{name}: R²={score:.3f}, zero weights={n_zero}")
```

---

## Exercise Preview

**Exercise 3.1: Regularisation for HDB Price Prediction**

You will:

1. Train unregularised, Ridge, Lasso, and Elastic Net models
2. Visualise the regularisation path (coefficients vs lambda)
3. Use cross-validation to find optimal regularisation strength
4. Compare feature selection by Lasso to Module 2 results

Scaffolding level: **Moderate (~50% code provided)**

---

## Common Pitfalls

| Mistake                                        | Fix                                                                  |
| ---------------------------------------------- | -------------------------------------------------------------------- |
| Not scaling features before regularisation     | Regularisation penalises large weights -- unscaled features dominate |
| Choosing lambda without cross-validation       | Always use CV; never set lambda by intuition                         |
| Using Lasso with highly correlated features    | Use Elastic Net instead                                              |
| Forgetting to scale test data with train stats | `scaler.transform(X_test)`, not `fit_transform`                      |
| Interpreting Ridge coefficients as importance  | They are shrunken, not zero -- use Lasso for selection               |

---

## Summary

- Prediction error = bias squared + variance + irreducible noise
- Simple models underfit (high bias); complex models overfit (high variance)
- L2 (Ridge) shrinks all weights; L1 (Lasso) zeros out irrelevant ones
- Elastic Net combines L1 and L2 for robust regularisation
- Regularisation = MAP estimation with shrinkage priors
- Cross-validation selects the optimal regularisation strength

---

## Next Lesson

**Lesson 3.2: Gradient Boosting**

We will learn:

- How gradient boosting builds ensembles of weak learners
- XGBoost and LightGBM via `TrainingPipeline`
- Hyperparameter tuning for tree-based models
