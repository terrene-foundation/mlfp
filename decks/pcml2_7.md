---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 2.7: Feature Engineering

### Module 2: Statistical Mastery

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Generate features automatically with `FeatureEngineer`
- Apply filter, wrapper, and embedded feature selection methods
- Understand feature interactions and polynomial features
- Build a feature engineering pipeline using Kailash ML

---

## Recap: Lesson 2.6

- Sequential testing allows valid early stopping of experiments
- DiD estimates causal effects by comparing group changes over time
- Propensity score matching creates comparable groups from observational data
- Each causal method has assumptions that must be verified

---

## Why Feature Engineering?

Raw data rarely predicts well on its own.

```
Raw features:               Engineered features:
  floor_area: 92             price_per_sqm: 5,217
  price: 480,000             floor_area_squared: 8,464
  lease_years: 72            lease_remaining_pct: 0.76
  transaction_date: 2024-03  quarter: Q1
                             years_since_2020: 4
                             is_large_flat: True
```

Good features encode **domain knowledge** into numbers a model can use.

---

## Feature Engineering Categories

| Category        | Examples                    | When                                  |
| --------------- | --------------------------- | ------------------------------------- |
| **Arithmetic**  | price/area, price\*rooms    | Ratios, products reveal relationships |
| **Temporal**    | day_of_week, month, quarter | Seasonal patterns                     |
| **Binning**     | price_band, age_group       | Non-linear effects from continuous    |
| **Encoding**    | one_hot, ordinal, target    | Converting categories to numbers      |
| **Interaction** | town x flat_type            | Combined effects                      |
| **Polynomial**  | area^2, area^3              | Non-linear continuous relationships   |
| **Aggregation** | town_avg_price, type_count  | Group-level statistics                |

---

## FeatureEngineer: Automated Generation

```python
from kailash_ml import FeatureEngineer

engineer = FeatureEngineer()
engineer.configure(
    dataset=df,
    target_column="price",
    strategies=[
        "arithmetic",       # ratios, products, differences
        "temporal",         # date decomposition
        "polynomial",       # squared, cubed terms
        "interaction",      # pairwise combinations
    ],
    max_features=50,
)

df_engineered = engineer.generate()
print(f"Original: {df.shape[1]} cols → Engineered: {df_engineered.shape[1]} cols")
```

---

## Arithmetic Features

```python
engineer = FeatureEngineer()
engineer.configure(
    dataset=df,
    target_column="price",
    strategies=["arithmetic"],
    arithmetic_ops=["divide", "multiply", "subtract"],
    numeric_columns=["floor_area", "lease_years", "storey"],
)

df_arith = engineer.generate()
# Creates: floor_area_div_lease_years, storey_mul_floor_area, etc.
```

```python
# Manual equivalent in Polars
df_manual = df.with_columns(
    (pl.col("price") / pl.col("floor_area")).alias("price_per_sqm"),
    (pl.col("lease_years") / 99 * 100).alias("lease_pct_remaining"),
)
```

---

## Temporal Features

```python
engineer = FeatureEngineer()
engineer.configure(
    dataset=df,
    target_column="price",
    strategies=["temporal"],
    date_columns=["transaction_date"],
)

df_temporal = engineer.generate()
# Creates: transaction_year, transaction_month, transaction_quarter,
#          transaction_day_of_week, transaction_is_weekend
```

Temporal features capture **seasonality** and **trends**.

---

## Interaction Features

```python
# Interaction: the combined effect of two features
# "4-ROOM in TAMPINES" might behave differently from
# "4-ROOM in BEDOK" -- the interaction captures this

engineer = FeatureEngineer()
engineer.configure(
    dataset=df,
    target_column="price",
    strategies=["interaction"],
    interaction_columns=["town", "flat_type", "storey"],
)

df_interact = engineer.generate()
# Creates: town_x_flat_type, town_x_storey, flat_type_x_storey
```

---

## The Feature Selection Problem

More features is not always better.

```
100 raw features
  ↓ FeatureEngineer
500 engineered features
  ↓ Problem!
  - Many are redundant (correlated)
  - Some add noise (irrelevant)
  - Model overfits to spurious patterns
  - Training becomes slow

Solution: Feature selection — keep only what helps
```

---

## Three Selection Approaches

```
Filter Methods (fast, pre-model):
  Score each feature independently → keep top K
  Examples: correlation, mutual information, chi-squared

Wrapper Methods (slow, model-based):
  Train models with different feature subsets → keep best
  Examples: forward selection, backward elimination, RFE

Embedded Methods (built into model):
  Model learns which features matter during training
  Examples: Lasso (L1), tree feature importance
```

---

## FeatureEngineer: Selection

```python
engineer = FeatureEngineer()
engineer.configure(
    dataset=df_engineered,
    target_column="price",

    # Selection after generation
    selection_method="mutual_information",
    n_select=20,                    # keep top 20 features
    remove_correlated=True,
    correlation_threshold=0.95,     # drop near-duplicates
)

df_selected = engineer.select()

print(f"Selected {df_selected.shape[1]} features from {df_engineered.shape[1]}")
print(engineer.get_feature_rankings())
```

---

## Filter: Correlation with Target

```python
# Manual filter selection with Polars
numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Float64, pl.Int64]]

correlations = []
for col in numeric_cols:
    if col != "price":
        r = df.select(pl.corr("price", col)).item()
        correlations.append({"feature": col, "correlation": abs(r)})

corr_df = pl.DataFrame(correlations).sort("correlation", descending=True)
print(corr_df.head(10))
```

Simple but effective: keep features most correlated with the target.

---

## Filter: Mutual Information

```python
from sklearn.feature_selection import mutual_info_regression

X = df_engineered.drop("price").to_numpy()
y = df_engineered["price"].to_numpy()

mi_scores = mutual_info_regression(X, y)

mi_df = pl.DataFrame({
    "feature": df_engineered.drop("price").columns,
    "mi_score": mi_scores,
}).sort("mi_score", descending=True)

print(mi_df.head(10))
```

Mutual information captures **non-linear** relationships (unlike correlation).

---

## Embedded: L1 Feature Selection

```python
# Lasso regression automatically zeros out unimportant features
from sklearn.linear_model import LassoCV
import numpy as np

X = df_engineered.drop("price").to_numpy()
y = df_engineered["price"].to_numpy()

lasso = LassoCV(cv=5)
lasso.fit(X, y)

# Non-zero coefficients = selected features
selected = np.where(lasso.coef_ != 0)[0]
feature_names = df_engineered.drop("price").columns

print(f"Lasso selected {len(selected)} features:")
for idx in selected:
    print(f"  {feature_names[idx]}: {lasso.coef_[idx]:.2f}")
```

---

## Feature Engineering Workflow

```
1. Start with domain knowledge
   └→ Create features YOU think matter

2. Automated generation
   └→ FeatureEngineer explores combinations

3. Remove redundancy
   └→ Drop highly correlated pairs (r > 0.95)

4. Select top features
   └→ Mutual information + embedded (Lasso/tree importance)

5. Validate on holdout
   └→ Does performance improve with engineered features?
```

---

## Feature Engineering Best Practices

| Practice                                | Why                                |
| --------------------------------------- | ---------------------------------- |
| Engineer before splitting train/test    | Consistent features across sets    |
| Fit scalers on train only               | Prevent data leakage               |
| Document each feature's meaning         | Interpretability matters           |
| Test feature importance stability       | Bootstrap feature selection        |
| Domain features first, automated second | Domain knowledge beats brute force |

---

## Exercise Preview

**Exercise 2.7: HDB Feature Engineering Pipeline**

You will:

1. Generate arithmetic, temporal, and interaction features with `FeatureEngineer`
2. Apply filter selection (correlation, mutual information)
3. Apply embedded selection (Lasso)
4. Compare model performance: raw features vs engineered + selected

Scaffolding level: **Moderate+ (~60% code provided)**

---

## Common Pitfalls

| Mistake                                     | Fix                                      |
| ------------------------------------------- | ---------------------------------------- |
| Engineering features after train/test split | Engineer first, then split               |
| Fitting scalers on full data                | Fit on train, transform test             |
| Too many interaction features               | Limit to domain-relevant pairs           |
| Ignoring feature correlation                | Remove near-duplicates (r > 0.95)        |
| Using only one selection method             | Combine filter + embedded for robustness |

---

## Summary

- Feature engineering encodes domain knowledge into model-ready numbers
- `FeatureEngineer` automates generation across multiple strategies
- Selection (filter, wrapper, embedded) keeps only useful features
- Mutual information captures non-linear relationships
- Domain knowledge features outperform brute-force generation

---

## Next Lesson

**Lesson 2.8: Feature Store and Project**

We will learn:

- Managing features with `FeatureStore` for reuse and versioning
- The feature lifecycle: create, register, serve, retire
- Module 2 capstone project
