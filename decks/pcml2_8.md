---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 2.8: Feature Store and Project

### Module 2: Statistical Mastery

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Manage features with `FeatureStore` for reuse and versioning
- Understand the feature lifecycle: create, register, serve, retire
- Apply the `ExperimentTracker` to log and compare experiments
- Combine all Module 2 skills into a capstone project

---

## Recap: Lesson 2.7

- `FeatureEngineer` automates feature generation (arithmetic, temporal, interaction)
- Filter, wrapper, and embedded methods select the most useful features
- Mutual information captures non-linear relationships
- Domain knowledge features outperform brute-force generation

---

## The Feature Reuse Problem

```
Data Scientist A: engineers price_per_sqm, lease_pct_remaining
Data Scientist B: engineers price_psm, remaining_lease_ratio

Same features, different names, different implementations.

Problems:
  - Duplicated work
  - Inconsistent definitions
  - Training/serving skew (different code paths)
  - No versioning or lineage
```

The `FeatureStore` solves this with a central registry.

---

## FeatureStore Overview

```python
from kailash_ml import FeatureStore

store = FeatureStore()
store.configure(
    storage_path="./feature_store",
    versioning=True,
)
```

```
FeatureStore
├── Feature Groups (logical collections)
│   ├── hdb_price_features (v1, v2, v3)
│   ├── hdb_location_features (v1)
│   └── hdb_temporal_features (v1, v2)
├── Registry (metadata, lineage, statistics)
└── Serving Layer (consistent retrieval)
```

---

## Feature Lifecycle

```
CREATE          REGISTER        SERVE           RETIRE
  │                │              │                │
  ▼                ▼              ▼                ▼
Engineer        Add to store    Retrieve for     Mark as
features        with metadata   training or      deprecated
                + version       inference

  FeatureEngineer → FeatureStore.register() → FeatureStore.get() → FeatureStore.retire()
```

---

## Registering Features

```python
from kailash_ml import FeatureStore, FeatureEngineer

# Step 1: Engineer features
engineer = FeatureEngineer()
engineer.configure(dataset=df, target_column="price")
df_features = engineer.generate()

# Step 2: Register in the store
store = FeatureStore()
store.configure(storage_path="./feature_store")

store.register(
    name="hdb_price_features",
    features=df_features,
    version="1.0",
    description="HDB price-derived features for Module 2",
    metadata={
        "source": "hdbprices.csv",
        "engineer": "FeatureEngineer",
        "n_features": df_features.shape[1],
    },
)
```

---

## Feature Metadata

Every registered feature group includes:

```python
info = store.get_info("hdb_price_features")

print(info)
# {
#     "name": "hdb_price_features",
#     "version": "1.0",
#     "created_at": "2026-03-15T10:30:00",
#     "n_features": 25,
#     "n_rows": 12345,
#     "schema": {"price_per_sqm": Float64, ...},
#     "statistics": {"price_per_sqm": {"mean": 5200, "std": 1800, ...}},
#     "description": "HDB price-derived features for Module 2",
# }
```

Metadata enables discovery, validation, and debugging.

---

## Retrieving Features

```python
# Get latest version
features = store.get("hdb_price_features")

# Get specific version
features_v1 = store.get("hdb_price_features", version="1.0")

# Get specific columns only
subset = store.get(
    "hdb_price_features",
    columns=["price_per_sqm", "lease_pct_remaining"],
)

# Get with time filter
recent = store.get(
    "hdb_price_features",
    filter=pl.col("transaction_date") > "2024-01-01",
)
```

---

## Feature Versioning

```python
# Version 1: basic features
store.register(name="hdb_price_features", features=v1_features, version="1.0")

# Version 2: added interaction features
store.register(name="hdb_price_features", features=v2_features, version="2.0")

# List all versions
versions = store.list_versions("hdb_price_features")
print(versions)
# ["1.0", "2.0"]

# Compare versions
diff = store.compare_versions("hdb_price_features", "1.0", "2.0")
print(f"Added columns: {diff.added}")
print(f"Removed columns: {diff.removed}")
print(f"Modified columns: {diff.modified}")
```

---

## Feature Statistics and Drift

```python
# Store tracks statistics at registration time
stats = store.get_statistics("hdb_price_features", version="2.0")

# Compare current data against registered statistics
new_df = loader.load("ascent02", "hdbprices_2025.csv")
drift_report = store.check_drift(
    name="hdb_price_features",
    new_data=new_df,
    threshold=0.1,    # PSI threshold
)

for feature, result in drift_report.items():
    if result["drifted"]:
        print(f"DRIFT: {feature} (PSI={result['psi']:.3f})")
```

This connects to `DriftMonitor` in Module 4.

---

## ExperimentTracker: Logging Results

```python
from kailash_ml import ExperimentTracker

tracker = ExperimentTracker()
tracker.configure(storage_path="./experiments")

# Start an experiment
with tracker.experiment("hdb_price_prediction") as exp:
    exp.log_params({
        "feature_version": "2.0",
        "model": "linear_regression",
        "n_features": 25,
    })

    # ... train model ...

    exp.log_metrics({
        "rmse": 45_000,
        "r2": 0.82,
        "mae": 32_000,
    })

    exp.log_artifact("feature_importance.csv")
```

---

## Comparing Experiments

```python
# List all runs
runs = tracker.list_runs("hdb_price_prediction")

# Compare runs
comparison = tracker.compare(
    "hdb_price_prediction",
    run_ids=[runs[0].id, runs[1].id],
    metrics=["rmse", "r2"],
)
print(comparison)

# Find best run
best = tracker.best_run(
    "hdb_price_prediction",
    metric="rmse",
    direction="minimize",
)
print(f"Best RMSE: {best.metrics['rmse']:,.0f} (run {best.id})")
```

---

## ExperimentTracker + FeatureStore Together

```python
# Track which features produced which results
with tracker.experiment("hdb_v2_features") as exp:
    # Log feature store version
    exp.log_params({
        "feature_store": "hdb_price_features",
        "feature_version": "2.0",
    })

    # Get features from store
    features = store.get("hdb_price_features", version="2.0")

    # Train and evaluate
    # ... model code ...

    exp.log_metrics({"rmse": 42_000, "r2": 0.85})
```

Full lineage: data -> features -> experiment -> results.

---

## Module 2 Capstone Overview

**Project: Statistical Analysis and Feature Engineering Pipeline**

```
1. Bayesian Analysis (2.1-2.2)
   └→ Prior beliefs → posterior estimates for HDB towns

2. Hypothesis Testing (2.3-2.4)
   └→ Which town differences are statistically significant?
   └→ Bootstrap confidence intervals for key metrics

3. Experiment Design (2.5-2.6)
   └→ CUPED-adjusted A/B test analysis
   └→ DiD estimate of MRT station impact

4. Feature Pipeline (2.7-2.8)
   └→ FeatureEngineer → FeatureStore → ExperimentTracker
```

---

## Capstone Deliverables

| Deliverable                   | Skills Used                            |
| ----------------------------- | -------------------------------------- |
| Bayesian town price estimates | Prior/posterior, conjugate updates     |
| Hypothesis test report        | P-values, effect sizes, FDR correction |
| Bootstrap analysis            | BCa intervals, permutation tests       |
| CUPED variance reduction      | Pre-experiment adjustment              |
| Engineered feature set        | FeatureEngineer, selection             |
| Feature store registration    | FeatureStore with versioning           |
| Experiment comparison         | ExperimentTracker logging              |

---

## Exercise Preview

**Exercise 2.8: Module 2 Capstone**

You will:

1. Engineer and select features using `FeatureEngineer`
2. Register features in `FeatureStore` with metadata and versioning
3. Track experiment results with `ExperimentTracker`
4. Produce a statistical report combining Bayesian, frequentist, and causal methods

Scaffolding level: **Moderate+ (~60% code provided)**

---

## Common Pitfalls

| Mistake                                       | Fix                                             |
| --------------------------------------------- | ----------------------------------------------- |
| Skipping feature registration                 | Always register in FeatureStore for reuse       |
| No versioning                                 | Bump versions when features change              |
| Training/serving feature mismatch             | Use FeatureStore.get() in both paths            |
| Not logging experiment parameters             | Log everything -- you will forget later         |
| Comparing experiments with different features | Use ExperimentTracker to track feature versions |

---

## Module 2 Summary

| Lesson | Key Skills                                      |
| ------ | ----------------------------------------------- |
| 2.1    | Bayesian thinking, priors, posteriors           |
| 2.2    | MLE, MAP, credible intervals                    |
| 2.3    | Hypothesis testing, power, multiple corrections |
| 2.4    | Bootstrap, BCa intervals, permutation tests     |
| 2.5    | CUPED, stratification, variance reduction       |
| 2.6    | Sequential testing, DiD, propensity matching    |
| 2.7    | FeatureEngineer, feature selection              |
| 2.8    | FeatureStore, ExperimentTracker, capstone       |

---

## What Comes Next

**Module 3: Supervised ML**

- Bias-variance tradeoff and regularisation (L1/L2)
- Gradient boosting with TrainingPipeline
- Class imbalance, calibration, and fairness
- Workflow orchestration and production deployment

You now have the statistical foundation. Next, we build models.
