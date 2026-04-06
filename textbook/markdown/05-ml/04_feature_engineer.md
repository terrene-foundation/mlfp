# Chapter 4: FeatureEngineer

## Overview

FeatureEngineer automates the two-step feature engineering workflow: **generate** candidate features, then **select** the best subset. The `generate()` method produces interaction terms, polynomial features, and binned columns from your existing numeric columns. The `select()` method ranks all features (original and generated) using importance, correlation, or mutual information, then picks the top-k.

This chapter covers:

- Defining a `FeatureSchema` to declare your feature columns
- Generating candidates with interaction, polynomial, and binning strategies
- Selecting the best features using importance, correlation, or mutual information
- Understanding `GeneratedFeatures`, `GeneratedColumn`, `SelectedFeatures`, and `FeatureRank`
- Serialization round-trips for all result types
- Controlling output size with `max_features`

## Prerequisites

| Requirement    | Details                                       |
| -------------- | --------------------------------------------- |
| Python         | 3.10+                                         |
| kailash-ml     | `pip install kailash-ml`                      |
| polars         | Installed with kailash-ml                     |
| Prior chapters | 1-3 (DataExplorer, Preprocessing, Visualizer) |
| Level          | Intermediate                                  |

## Concepts

### The Two-Step Pattern

Feature engineering in kailash-ml follows a deliberate two-step process:

1. **Generate** -- Cast a wide net. Create many candidate features (interactions, polynomials, bins) without worrying about usefulness.
2. **Select** -- Narrow down. Rank all features (original + generated) by predictive value and keep only the best.

This separation keeps the generation logic reusable across different targets and lets you compare selection methods on the same candidate pool.

### Generation Strategies

| Strategy       | What It Produces                                     | Naming Convention |
| -------------- | ---------------------------------------------------- | ----------------- |
| `interactions` | Products of every numeric pair (C(n,2) combinations) | `col_a_x_col_b`   |
| `polynomial`   | Squared value of each numeric column                 | `col_squared`     |
| `binning`      | Quantile-binned version of each numeric column       | `col_binned`      |

### Selection Methods

| Method        | How It Ranks                                                   |
| ------------- | -------------------------------------------------------------- |
| `importance`  | Trains a tree-based model, extracts `feature_importances_`     |
| `correlation` | Computes absolute Pearson correlation with the target          |
| `mutual_info` | Computes mutual information (non-linear relationship strength) |

### FeatureSchema

`FeatureSchema` declares which columns are features and their expected dtypes. The `entity_id_column` identifies the row key (excluded from feature generation). The target column is not part of the schema -- it is passed separately to `select()`.

## Key API Reference

| Class / Method                                             | Purpose                                                                                                        |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `FeatureEngineer(max_features=10)`                         | Instantiate with optional feature count limit                                                                  |
| `engineer.generate(df, schema, strategies=[...])`          | Produce candidate features. Returns `GeneratedFeatures`                                                        |
| `engineer.select(data, candidates, target, method, top_k)` | Rank and select. Returns `SelectedFeatures`                                                                    |
| `FeatureSchema(name, features, entity_id_column)`          | Declare feature columns                                                                                        |
| `FeatureField(name, dtype)`                                | Single feature declaration                                                                                     |
| `GeneratedFeatures`                                        | Fields: `data` (augmented DataFrame), `generated_columns`, `total_candidates`                                  |
| `GeneratedColumn`                                          | Fields: `name`, `strategy`, `source_columns`, `dtype`                                                          |
| `SelectedFeatures`                                         | Fields: `selected_columns`, `dropped_columns`, `rankings`, `method`, `n_original`, `n_generated`, `n_selected` |
| `FeatureRank`                                              | Fields: `column_name`, `rank`, `score`, `source` ("original" or "generated")                                   |

## Code Walkthrough

### Step 1: Prepare Data and Schema

```python
import polars as pl
from kailash_ml import FeatureField, FeatureSchema

df = pl.DataFrame({
    "entity_id": [f"u{i}" for i in range(100)],
    "age": [20 + (i % 50) for i in range(100)],
    "income": [30000.0 + i * 1000.0 for i in range(100)],
    "tenure_months": [1 + (i * 3 % 60) for i in range(100)],
    "num_products": [1 + (i % 5) for i in range(100)],
    "churned": [(i % 3 == 0) for i in range(100)],
})
df = df.with_columns(pl.col("churned").cast(pl.Int64))

schema = FeatureSchema(
    name="churn_features",
    features=[
        FeatureField(name="age", dtype="int64"),
        FeatureField(name="income", dtype="float64"),
        FeatureField(name="tenure_months", dtype="int64"),
        FeatureField(name="num_products", dtype="int64"),
    ],
    entity_id_column="entity_id",
)
```

### Step 2: Generate Candidate Features

```python
from kailash_ml.engines.feature_engineer import FeatureEngineer

engineer = FeatureEngineer(max_features=10)
candidates = engineer.generate(
    df, schema,
    strategies=["interactions", "polynomial", "binning"],
)

# Original columns preserved, new ones added
assert "age" in candidates.data.columns
assert len(candidates.generated_columns) > 0
```

### Step 3: Inspect Generated Columns

```python
interactions = [g for g in candidates.generated_columns if g.strategy == "interaction"]
polynomials = [g for g in candidates.generated_columns if g.strategy == "polynomial"]
binned = [g for g in candidates.generated_columns if g.strategy == "binning"]

# With 4 numeric features: C(4,2) = 6 interactions, 4 polynomials, 4 binned
assert len(interactions) == 6
assert len(polynomials) == 4
assert len(binned) == 4

# Naming conventions
for g in interactions:
    assert "_x_" in g.name
for g in polynomials:
    assert g.name.endswith("_squared")
for g in binned:
    assert g.name.endswith("_binned")
```

### Step 4: Select Best Features by Importance

```python
selected = engineer.select(
    candidates.data, candidates,
    target="churned",
    method="importance",
    top_k=8,
)

assert len(selected.selected_columns) <= 8
assert selected.method == "importance"

# Rankings are sorted by score descending
for i in range(len(selected.rankings) - 1):
    assert selected.rankings[i].score >= selected.rankings[i + 1].score
```

### Step 5: Try Other Selection Methods

```python
# Correlation-based
selected_corr = engineer.select(
    candidates.data, candidates,
    target="churned", method="correlation", top_k=5,
)

# Mutual information
selected_mi = engineer.select(
    candidates.data, candidates,
    target="churned", method="mutual_info", top_k=5,
)
```

### Step 6: Serialization

```python
# All result types support to_dict() / from_dict()
gc_dict = candidates.generated_columns[0].to_dict()
gc_restored = GeneratedColumn.from_dict(gc_dict)

sf_dict = selected.to_dict()
sf_restored = SelectedFeatures.from_dict(sf_dict)
assert sf_restored.n_selected == selected.n_selected
```

## Common Mistakes

| Mistake                                              | What Happens                              | Fix                                                     |
| ---------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------- |
| Requesting `interactions` with only 1 numeric column | Zero interactions generated (no error)    | Need at least 2 numeric columns for interactions        |
| Using an invalid selection method                    | `ValueError`                              | Use `importance`, `correlation`, or `mutual_info`       |
| Forgetting `entity_id_column` in schema              | Entity ID treated as a feature            | Always declare it in the schema                         |
| Not casting boolean target to int                    | May cause issues with selection methods   | Cast with `pl.col("target").cast(pl.Int64)`             |
| Expecting `candidates.data` to serialize             | `data` field is `None` in serialized form | DataFrames are not serializable; only metadata persists |

## Exercises

1. **Strategy comparison**: Generate features with only `interactions` vs only `polynomial` vs only `binning`. Which strategy produces features that rank highest for your target?

2. **Selection method agreement**: Run `importance`, `correlation`, and `mutual_info` selection on the same candidate pool. Do the top-5 features agree? When might they disagree?

3. **max_features control**: Create a `FeatureEngineer(max_features=3)` and run `select()` without specifying `top_k`. Verify that at most 3 features are selected.

## Key Takeaways

- FeatureEngineer follows a two-step pattern: `generate()` then `select()`.
- Three generation strategies (interactions, polynomial, binning) create diverse candidates.
- Three selection methods (importance, correlation, mutual_info) rank features by predictive value.
- Each generated column has metadata: strategy, source columns, naming convention.
- Rankings distinguish between `original` and `generated` features so you know what the model relies on.
- `max_features` provides a global cap when `top_k` is not specified.

## Next Chapter

Chapter 5 covers **FeatureStore** -- the engine that registers feature schemas, stores computed features in a database, and retrieves them with point-in-time correctness.
