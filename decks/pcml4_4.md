---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 4.4: Anomaly Detection and Ensembles

### Module 4: Advanced ML

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Detect anomalies with Isolation Forest and Local Outlier Factor
- Combine multiple models with `EnsembleEngine`
- Apply stacking, blending, and voting strategies
- Choose the right ensemble method for the problem

---

## Recap: Lesson 4.3

- PCA compresses features along maximum-variance axes
- UMAP and t-SNE visualise high-dimensional data in 2D
- PCA for compression before modelling; UMAP/t-SNE for exploration
- Always scale before dimensionality reduction

---

## Anomaly Detection: Finding the Unusual

```
Normal transactions:  consistent patterns
  $420k, 4-room, Tampines, 92sqm

Anomalies:  something unusual
  $1.2M, 3-room, Woodlands, 65sqm  ← price way too high for type/area
  $50k, 5-room, Central, 120sqm    ← price suspiciously low
```

Anomalies may be errors, fraud, or genuinely rare events.

---

## Isolation Forest

**Idea**: Anomalies are easier to isolate than normal points.

```
Normal point: needs many splits to isolate
  ┌─────────────┐
  │ · · · · · · │ → split → split → split → split → isolated!
  │ · · ● · · · │
  │ · · · · · · │
  └─────────────┘

Anomaly: isolated quickly (few splits)
  ┌─────────────┐
  │ · · · · · · │
  │ · · · · · · │ → split → isolated!
  │             ●│
  └─────────────┘

Shorter path = more anomalous
```

---

## Isolation Forest in Code

```python
from sklearn.ensemble import IsolationForest
import polars as pl

features = ["price", "floor_area", "lease_years", "storey"]
X = df.select(features).to_numpy()

iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.05,   # expect ~5% anomalies
    random_state=42,
)

# -1 = anomaly, 1 = normal
labels = iso_forest.fit_predict(X)
scores = iso_forest.decision_function(X)  # lower = more anomalous

df_anomalies = df.with_columns(
    pl.Series("anomaly", labels == -1),
    pl.Series("anomaly_score", scores),
)
print(f"Anomalies found: {(labels == -1).sum()}")
```

---

## Local Outlier Factor (LOF)

**Idea**: Compare each point's local density to its neighbours' density.

```
Normal point: similar density to neighbours
  · · · ·
  · ● · ·    density(●) ≈ density(neighbours)
  · · · ·    LOF ≈ 1

Anomaly: much lower density than neighbours
  · · · ·
  · · · ·
              ●           density(●) << density(neighbours)
                          LOF >> 1
```

---

## LOF in Code

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05,
)

labels = lof.fit_predict(X)
scores = lof.negative_outlier_factor_

df_lof = df.with_columns(
    pl.Series("anomaly_lof", labels == -1),
    pl.Series("lof_score", -scores),  # higher = more anomalous
)
```

---

## Isolation Forest vs LOF

| Aspect                    | Isolation Forest               | LOF                     |
| ------------------------- | ------------------------------ | ----------------------- |
| Speed                     | Fast (tree-based)              | Slower (distance-based) |
| Scalability               | Excellent                      | Moderate                |
| Handles high dimensions   | Yes                            | Struggles               |
| Local vs global anomalies | Global                         | Local                   |
| Best for                  | Large datasets, clear outliers | Density-varying data    |

---

## Why Ensembles?

```
Single model:   One perspective, one set of errors
                "All eggs in one basket"

Ensemble:       Multiple perspectives, errors cancel out
                "Wisdom of crowds"

If models make INDEPENDENT errors,
combining them reduces overall error.
```

---

## EnsembleEngine Overview

```python
from kailash_ml import EnsembleEngine

ensemble = EnsembleEngine()
ensemble.configure(
    dataset=df,
    target_column="price",
    task="regression",
    strategy="stacking",
    base_models=[
        {"algorithm": "lightgbm", "params": {"n_estimators": 300}},
        {"algorithm": "xgboost", "params": {"n_estimators": 300}},
        {"algorithm": "ridge", "params": {"alpha": 1.0}},
    ],
    meta_model={"algorithm": "ridge"},
)

result = ensemble.run()
print(f"Ensemble RMSE: ${result.metrics['rmse']:,.0f}")
```

---

## Voting: Simple Combination

```python
ensemble = EnsembleEngine()
ensemble.configure(
    dataset=df,
    target_column="price",
    task="regression",
    strategy="voting",
    voting_method="mean",    # or "median" for robustness
    base_models=[
        {"algorithm": "lightgbm"},
        {"algorithm": "xgboost"},
        {"algorithm": "ridge"},
    ],
)
```

```
Model 1: $480k
Model 2: $510k    → Average: $490k
Model 3: $480k
```

---

## Stacking: Learned Combination

```
Level 0 (base models):
  LightGBM  → prediction₁
  XGBoost   → prediction₂    → Meta-model learns optimal weights
  Ridge     → prediction₃

Level 1 (meta-model):
  Ridge(prediction₁, prediction₂, prediction₃) → final prediction
```

Stacking learns **how much to trust** each base model.

---

## Blending vs Stacking

```
Stacking:  Uses cross-validation to generate base model predictions
           → No data leakage, but slower
           → More reliable

Blending:  Uses a held-out validation set for base predictions
           → Faster, simpler
           → Wastes some training data

ensemble.configure(
    strategy="blending",
    blend_size=0.2,   # 20% validation set for blending
)
```

---

## Ensemble Strategy Selection

```
How many models?
│
├─ 2-3 models → Voting (simple, effective)
│
├─ 3-5 diverse models → Stacking (learns weights)
│
└─ Need speed → Blending (no cross-validation)

Diversity matters most:
  ✅ LightGBM + Ridge + KNN (different algorithms)
  ❌ LightGBM + LightGBM + LightGBM (same algorithm, same errors)
```

---

## Exercise Preview

**Exercise 4.4: Anomaly Detection and Model Ensembles**

You will:

1. Detect anomalous HDB transactions with Isolation Forest and LOF
2. Compare anomaly methods and investigate flagged transactions
3. Build stacking and voting ensembles with `EnsembleEngine`
4. Measure ensemble improvement over individual models

Scaffolding level: **Light+ (~40% code provided)**

---

## Common Pitfalls

| Mistake                           | Fix                                          |
| --------------------------------- | -------------------------------------------- |
| Setting contamination too high    | Start at 1-5%; domain knowledge guides this  |
| Treating all anomalies as errors  | Some are genuine rare events worth studying  |
| Ensemble of identical models      | Diversity is key -- use different algorithms |
| Stacking without cross-validation | Use stacking, not blending, to avoid leakage |
| Too many base models              | 3-5 diverse models is usually optimal        |

---

## Summary

- Isolation Forest isolates anomalies with few random splits
- LOF detects local density anomalies relative to neighbours
- `EnsembleEngine` combines models via voting, stacking, or blending
- Diversity among base models is more important than individual accuracy
- Stacking with cross-validation is the most reliable ensemble method

---

## Next Lesson

**Lesson 4.5: NLP — Text to Topics**

We will learn:

- Text preprocessing: tokenisation, TF-IDF
- Topic modelling with BERTopic
- Processing text data with Polars
