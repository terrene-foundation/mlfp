# Chapter 5: FeatureStore

## Overview

FeatureStore bridges the gap between feature engineering and model training by providing a centralized, versioned repository for computed features. You register a `FeatureSchema`, compute features from raw data, store them in a database, and retrieve them by entity ID or time window. Point-in-time correctness ensures that training data never leaks future information.

This chapter covers:

- Setting up FeatureStore with a `ConnectionManager`
- Registering feature schemas with `register_features()`
- Computing and validating features with `compute()`
- Storing features with `store()`
- Retrieving features by entity ID with `get_features()`
- Building training sets with time windows via `get_training_set()`
- Lazy retrieval with `get_features_lazy()`
- Schema conflict handling and nullable validation

## Prerequisites

| Requirement        | Details                                    |
| ------------------ | ------------------------------------------ |
| Python             | 3.10+                                      |
| kailash-ml         | `pip install kailash-ml`                   |
| kailash (core SDK) | For `ConnectionManager`                    |
| polars             | Installed with kailash-ml                  |
| Prior chapters     | 1-4 (DataExplorer through FeatureEngineer) |
| Level              | Intermediate                               |

## Concepts

### Why a Feature Store?

Without a feature store, features are computed ad-hoc in notebooks, leading to:

- **Training/serving skew** -- Different code computes features for training vs inference.
- **Feature duplication** -- Multiple teams recompute the same features independently.
- **Point-in-time leakage** -- Training data accidentally includes future feature values.

FeatureStore solves these by centralizing feature definitions (schemas), storing computed values with timestamps, and providing retrieval APIs that respect time boundaries.

### Architecture

FeatureStore uses `ConnectionManager` from the Kailash core SDK for database access. It creates internal metadata tables on `initialize()` and one backing table per registered schema. Features are stored with `entity_id`, `created_at` timestamps, and the declared feature columns.

### Point-in-Time Retrieval

`get_training_set(schema, start, end)` returns only rows where `created_at` falls within the window. This prevents lookahead bias when building training datasets from historical features.

### Lazy Retrieval

`get_features_lazy()` returns a polars `LazyFrame` instead of a `DataFrame`, enabling streaming operations on large feature sets without loading everything into memory.

## Key API Reference

| Class / Method                                            | Purpose                                                 |
| --------------------------------------------------------- | ------------------------------------------------------- |
| `FeatureStore(conn, table_prefix="kml_feat_")`            | Create store with a database connection                 |
| `fs.initialize()`                                         | Async. Create metadata tables (idempotent)              |
| `fs.register_features(schema)`                            | Async. Register a schema and create its backing table   |
| `fs.compute(data, schema)`                                | Validate data against schema, project to schema columns |
| `fs.store(data, schema)`                                  | Async. Persist computed features to the database        |
| `fs.get_features(entity_ids, feature_names, schema)`      | Async. Retrieve by entity IDs                           |
| `fs.get_training_set(schema, start, end)`                 | Async. Retrieve by time window                          |
| `fs.get_features_lazy(entity_ids, feature_names, schema)` | Async. Returns a polars LazyFrame                       |
| `fs.list_schemas()`                                       | Async. List all registered schemas                      |

## Code Walkthrough

### Step 1: Set Up Infrastructure

```python
from kailash.db.connection import ConnectionManager
from kailash_ml import FeatureField, FeatureSchema
from kailash_ml.engines.feature_store import FeatureStore

conn = ConnectionManager("sqlite:///:memory:")
await conn.initialize()

fs = FeatureStore(conn, table_prefix="kml_feat_")
await fs.initialize()
```

### Step 2: Define and Register a Schema

```python
schema = FeatureSchema(
    name="user_churn",
    features=[
        FeatureField(name="age", dtype="float64"),
        FeatureField(name="tenure_months", dtype="float64"),
        FeatureField(name="monthly_charges", dtype="float64"),
    ],
    entity_id_column="user_id",
)

await fs.register_features(schema)

# Re-registering the same schema is safe (idempotent)
await fs.register_features(schema)
```

### Step 3: Compute and Store Features

```python
import polars as pl

raw_data = pl.DataFrame({
    "user_id": [f"u{i}" for i in range(20)],
    "age": [25.0 + i for i in range(20)],
    "tenure_months": [1.0 + i * 2 for i in range(20)],
    "monthly_charges": [50.0 + i * 5.0 for i in range(20)],
})

computed = fs.compute(raw_data, schema)  # Validates and projects
stored_count = await fs.store(computed, schema)
assert stored_count == 20
```

### Step 4: Retrieve by Entity IDs

```python
features = await fs.get_features(
    entity_ids=["u0", "u1", "u2"],
    feature_names=["age", "tenure_months"],
    schema=schema,
)
assert features.height == 3

# Non-existent entities return empty DataFrame
missing = await fs.get_features(
    entity_ids=["nonexistent"],
    feature_names=["age"],
    schema=schema,
)
assert missing.height == 0
```

### Step 5: Build a Training Set by Time Window

```python
from datetime import datetime, timedelta, timezone

now = datetime.now(timezone.utc)
training_set = await fs.get_training_set(
    schema,
    start=now - timedelta(hours=1),
    end=now + timedelta(hours=1),
)
assert training_set.height == 20
```

### Step 6: Lazy Retrieval

```python
lazy_features = await fs.get_features_lazy(
    entity_ids=["u0", "u5"],
    feature_names=["age", "monthly_charges"],
    schema=schema,
)
assert isinstance(lazy_features, pl.LazyFrame)

collected = lazy_features.collect()
assert collected.height == 2
```

### Step 7: Validation Edge Cases

```python
# Missing columns raises ValueError
try:
    fs.compute(pl.DataFrame({"user_id": ["u0"], "age": [30.0]}), schema)
except ValueError as e:
    assert "missing" in str(e).lower()

# Nulls in non-nullable column raises ValueError
strict_schema = FeatureSchema(
    name="strict_features",
    features=[FeatureField(name="value", dtype="float64", nullable=False)],
    entity_id_column="entity_id",
)
try:
    fs.compute(pl.DataFrame({"entity_id": ["a"], "value": [None]}), strict_schema)
except ValueError as e:
    assert "null" in str(e).lower()
```

## Common Mistakes

| Mistake                                                                   | What Happens                   | Fix                                              |
| ------------------------------------------------------------------------- | ------------------------------ | ------------------------------------------------ |
| Forgetting `await fs.initialize()`                                        | Metadata tables do not exist   | Always initialize before first use               |
| Re-registering with a different schema under the same name                | `ValueError` (schema conflict) | Use a unique name or keep the schema identical   |
| Using an invalid table prefix (starts with digit, contains special chars) | `ValueError`                   | Use alphanumeric + underscore prefixes           |
| Calling `compute()` with missing feature columns                          | `ValueError`                   | Ensure all schema fields are present in the data |
| Not closing `ConnectionManager`                                           | Resource leak                  | Call `await conn.close()` when done              |

## Exercises

1. **Time-window training**: Store features at two different times (separated by `time.sleep(1)` or explicit timestamps). Use `get_training_set()` with a narrow window to retrieve only the first batch.

2. **Schema evolution**: Try registering a schema with the same name but different features. What error do you get? How would you handle schema evolution in practice?

3. **Lazy vs eager**: Compare memory usage of `get_features()` vs `get_features_lazy()` on a large dataset. When does lazy retrieval provide a meaningful advantage?

## Key Takeaways

- FeatureStore centralizes feature definitions and storage, preventing training/serving skew.
- `compute()` validates data against the schema before storage, catching missing columns and null violations early.
- `get_features()` retrieves by entity ID; `get_training_set()` retrieves by time window for point-in-time correctness.
- `get_features_lazy()` returns a `LazyFrame` for memory-efficient streaming.
- Schema registration is idempotent, but conflicting schemas under the same name are rejected.
- Always pair `ConnectionManager` with `initialize()` and `close()`.

## Next Chapter

Chapter 6 covers **ExperimentTracker** -- the engine that tracks experiment runs, parameters, step-based metrics, tags, and provides run comparison.
