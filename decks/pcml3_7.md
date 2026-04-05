---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 3.7: Model Registry and HPO

### Module 3: Supervised ML

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Use `HyperparameterSearch` for automated model tuning
- Register and version models with `ModelRegistry`
- Compare model candidates and promote the best to production
- Design efficient search strategies (grid, random, Bayesian)

---

## Recap: Lesson 3.6

- DataFlow provides zero-config database operations from model definitions
- `@db.model` creates tables with auto-generated CRUD
- `db.express` handles ad-hoc queries
- Persisted predictions and model metadata for audit trails

---

## The Hyperparameter Problem

```
LightGBM has dozens of hyperparameters:
  n_estimators:     [100, 200, 500, 1000, 2000]
  learning_rate:    [0.001, 0.01, 0.05, 0.1, 0.3]
  max_depth:        [3, 4, 5, 6, 8, 10]
  num_leaves:       [15, 31, 63, 127]
  min_child_samples:[5, 10, 20, 50]

Total combinations: 5 × 5 × 6 × 4 × 4 = 2,400
Each takes ~30 seconds to train
Exhaustive search: 2,400 × 30s = 20 hours
```

We need smarter search strategies.

---

## Search Strategies

| Strategy               | How It Works                      | When to Use              |
| ---------------------- | --------------------------------- | ------------------------ |
| **Grid Search**        | Try all combinations              | Few params, small space  |
| **Random Search**      | Sample randomly                   | Large space, many params |
| **Bayesian (TPE)**     | Learn which regions are promising | Best default choice      |
| **Successive Halving** | Eliminate bad configs early       | Very large search spaces |

Random search finds good configs faster than grid search for most problems.

---

## HyperparameterSearch: Basic Usage

```python
from kailash_ml import HyperparameterSearch

search = HyperparameterSearch()
search.configure(
    dataset=df,
    target_column="price",
    task="regression",
    algorithm="lightgbm",

    search_space={
        "n_estimators": [100, 300, 500, 1000],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [4, 6, 8],
        "num_leaves": [15, 31, 63],
    },

    strategy="bayesian",
    n_trials=50,
    cv_folds=5,
    metric="rmse",
    direction="minimize",
)

result = search.run()
```

---

## Search Space Types

```python
search.configure(
    search_space={
        # Categorical: choose from a list
        "algorithm": ["xgboost", "lightgbm"],

        # Integer range
        "n_estimators": {"type": "int", "low": 100, "high": 2000},

        # Float range (linear)
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},

        # Float range (log scale — for learning rate)
        "learning_rate": {"type": "log_float", "low": 0.001, "high": 0.3},

        # Integer range (log scale)
        "num_leaves": {"type": "log_int", "low": 8, "high": 256},
    },
)
```

Use log scale for parameters that span orders of magnitude.

---

## Bayesian Optimisation Visual

```
Trial 1-5:   Explore randomly
             · · · · ·

Trial 6-15:  Focus on promising regions
             · ·● ● ·
                ↑ good region

Trial 16-50: Exploit best region, occasional exploration
               ●●●● ·
               ↑↑↑↑
             Converging on optimum

Bayesian search learns from past trials to pick better next candidates.
```

---

## Interpreting Search Results

```python
result = search.run()

# Best configuration
print(f"Best params: {result.best_params}")
print(f"Best RMSE: ${result.best_score:,.0f}")

# All trials
trials_df = result.trials_dataframe()
print(trials_df.sort("rmse").head(10))

# Hyperparameter importance
print(result.param_importances)
```

```
┌───────────────┬────────────┐
│ parameter     ┆ importance │
│ learning_rate ┆ 0.42       │
│ n_estimators  ┆ 0.28       │
│ max_depth     ┆ 0.18       │
│ num_leaves    ┆ 0.12       │
└───────────────┴────────────┘
```

---

## ModelRegistry: Version Management

```python
from kailash_ml import ModelRegistry

registry = ModelRegistry()
registry.configure(storage_path="./model_registry")

# Register a model
registry.register(
    name="hdb_price_predictor",
    model=result.best_model,
    version="1.0.0",
    metrics=result.best_metrics,
    params=result.best_params,
    tags={"dataset": "hdb_2024", "author": "ascent03"},
)
```

---

## Model Lifecycle

```
DEVELOPMENT → STAGING → PRODUCTION → ARCHIVED

  register()     promote()     promote()      archive()
  ┌──────┐      ┌───────┐    ┌──────────┐   ┌──────────┐
  │ Dev  │─────→│Staging│───→│Production│──→│ Archived │
  └──────┘      └───────┘    └──────────┘   └──────────┘
                 Validate     Serve           Retire
                 on staging   predictions     old version
```

---

## Promoting Models

```python
# List all versions
versions = registry.list_versions("hdb_price_predictor")
for v in versions:
    print(f"v{v.version}: {v.stage} | RMSE={v.metrics['rmse']:,.0f}")

# Promote to staging
registry.promote(
    name="hdb_price_predictor",
    version="1.0.0",
    stage="staging",
)

# After validation, promote to production
registry.promote(
    name="hdb_price_predictor",
    version="1.0.0",
    stage="production",
)
```

---

## Comparing Model Candidates

```python
# Compare two versions
comparison = registry.compare(
    name="hdb_price_predictor",
    versions=["1.0.0", "1.1.0"],
    metrics=["rmse", "r2", "mae"],
)

print(comparison)
```

```
┌─────────┬──────────┬───────┬──────────┐
│ version ┆ rmse     ┆ r2    ┆ mae      │
│ 1.0.0   ┆ 45,000   ┆ 0.82  ┆ 32,000   │
│ 1.1.0   ┆ 41,000   ┆ 0.85  ┆ 29,000   │  ← better
└─────────┴──────────┴───────┴──────────┘
```

---

## Loading Production Models

```python
# Load the current production model
model = registry.load(
    name="hdb_price_predictor",
    stage="production",
)

# Make predictions
predictions = model.predict(new_data)

# Load a specific version
model_v1 = registry.load(
    name="hdb_price_predictor",
    version="1.0.0",
)
```

---

## HPO + Registry Workflow

```python
from kailash_ml import HyperparameterSearch, ModelRegistry

# Step 1: Search for best hyperparameters
search = HyperparameterSearch()
search.configure(
    dataset=df, target_column="price",
    algorithm="lightgbm", strategy="bayesian",
    n_trials=50, metric="rmse",
    search_space={...},
)
result = search.run()

# Step 2: Register best model
registry = ModelRegistry()
registry.configure(storage_path="./model_registry")
registry.register(
    name="hdb_price_predictor",
    model=result.best_model,
    version="1.1.0",
    metrics=result.best_metrics,
    params=result.best_params,
)

# Step 3: Compare with current production
comparison = registry.compare("hdb_price_predictor", ["1.0.0", "1.1.0"])
```

---

## Exercise Preview

**Exercise 3.7: HPO and Model Registry**

You will:

1. Run Bayesian hyperparameter search with `HyperparameterSearch`
2. Analyse search results and parameter importances
3. Register multiple model versions in `ModelRegistry`
4. Compare candidates and promote the best to production

Scaffolding level: **Moderate (~50% code provided)**

---

## Common Pitfalls

| Mistake                               | Fix                                                    |
| ------------------------------------- | ------------------------------------------------------ |
| Grid search with many parameters      | Use random or Bayesian search                          |
| Not using log scale for learning rate | `"type": "log_float"` for spanning orders of magnitude |
| Too few trials for Bayesian search    | Minimum 30-50 for Bayesian to learn                    |
| Promoting without validation          | Always validate on staging data first                  |
| Not recording search history          | `HyperparameterSearch` logs all trials automatically   |

---

## Summary

- `HyperparameterSearch` automates model tuning with grid, random, or Bayesian strategies
- Bayesian optimisation learns from past trials for efficient search
- `ModelRegistry` manages versioned models through dev, staging, and production
- Compare candidates on metrics before promoting
- The HPO-to-registry pipeline ensures reproducible model selection

---

## Next Lesson

**Lesson 3.8: Production Pipeline Project**

We will learn:

- Combining all Module 3 components into a production pipeline
- End-to-end: data to model to predictions to database
- Module 3 capstone project
