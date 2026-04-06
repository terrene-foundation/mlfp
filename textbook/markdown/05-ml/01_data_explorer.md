# Chapter 1: DataExplorer & AlertConfig

## Overview

DataExplorer is the first engine you reach for when working with a new dataset. It profiles every column in a polars DataFrame -- computing statistical summaries, inferring semantic types, building correlation matrices, detecting missing-value patterns, and raising data-quality alerts. AlertConfig lets you tune the thresholds that trigger those alerts so you can adapt sensitivity to your domain.

This chapter covers:

- Profiling a dataset with `DataExplorer.profile()`
- Understanding `DataProfile` and `ColumnProfile` objects
- Semantic type inference (numeric, categorical, boolean, constant, id, text)
- Pearson and Spearman correlation matrices
- Missing-value pattern detection
- Data-quality alerts via `AlertConfig`
- Comparing two datasets with `compare()`
- Serialization round-trips for reproducible reporting

## Prerequisites

| Requirement | Details                   |
| ----------- | ------------------------- |
| Python      | 3.10+                     |
| kailash-ml  | `pip install kailash-ml`  |
| polars      | Installed with kailash-ml |
| Level       | Basic                     |

You should be comfortable creating polars DataFrames and using `async`/`await`.

## Concepts

### Dataset Profiling

Before training any model, you need to understand your data. DataExplorer automates the tedious process of running descriptive statistics on every column. A single `await explorer.profile(df)` call produces a `DataProfile` containing:

- **Row and column counts** -- the shape of your data.
- **Per-column statistics** -- mean, standard deviation, min, max, quartiles (Q25, Q50, Q75), skewness, kurtosis, IQR, outlier count, and zero count for numeric columns.
- **Cardinality ratio** -- `unique_count / count`, which helps distinguish ID columns (ratio near 1.0) from categorical columns (low ratio).
- **Type inference** -- DataExplorer classifies each column as `numeric`, `categorical`, `boolean`, `constant`, `id`, or `text` based on dtype and cardinality.
- **Top values** -- For categorical/string columns, the most frequent values and their counts.
- **Correlation matrices** -- Both Pearson (linear) and Spearman (rank) correlations among numeric columns.
- **Missing patterns** -- Which columns have nulls and how they co-occur.
- **Metadata** -- Duplicate row count, memory usage in bytes, head/tail samples, and a type summary dictionary.

### AlertConfig

Alerts are data-quality warnings that fire when a column exceeds a threshold. The eight alert types are:

| Alert Type         | Trigger                                                         |
| ------------------ | --------------------------------------------------------------- |
| `high_nulls`       | Null percentage exceeds `high_null_pct_threshold`               |
| `constant`         | Column has only one unique value                                |
| `high_skewness`    | Absolute skewness exceeds `skewness_threshold`                  |
| `high_zeros`       | Zero percentage exceeds `zero_pct_threshold`                    |
| `high_cardinality` | Cardinality ratio exceeds `high_cardinality_ratio`              |
| `high_correlation` | Correlation between a pair exceeds `high_correlation_threshold` |
| `duplicates`       | Duplicate percentage exceeds `duplicate_pct_threshold`          |
| `imbalanced`       | Class imbalance ratio exceeds `imbalance_ratio_threshold`       |

You create an `AlertConfig` with custom thresholds and pass it to `DataExplorer(alert_config=...)`.

### Dataset Comparison

`compare()` profiles two DataFrames in parallel and computes column-level deltas -- useful for detecting distribution shift between training and production data before you even get to DriftMonitor.

## Key API Reference

| Class / Method                                      | Purpose                                                                                                                                                                                                                                                                  |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `DataExplorer()`                                    | Instantiate the explorer, optionally with `AlertConfig`                                                                                                                                                                                                                  |
| `explorer.profile(df, columns=None)`                | Async. Profile a DataFrame, optionally a subset of columns                                                                                                                                                                                                               |
| `explorer.compare(df_a, df_b)`                      | Async. Profile both DataFrames and compute deltas                                                                                                                                                                                                                        |
| `DataProfile`                                       | Result of `profile()`. Fields: `n_rows`, `n_columns`, `columns`, `correlation_matrix`, `spearman_matrix`, `alerts`, `duplicate_count`, `memory_bytes`, `sample_head`, `sample_tail`, `type_summary`                                                                      |
| `ColumnProfile`                                     | Per-column stats. Fields: `name`, `dtype`, `count`, `null_count`, `null_pct`, `unique_count`, `cardinality_ratio`, `mean`, `std`, `min_val`, `max_val`, `q25`, `q50`, `q75`, `skewness`, `kurtosis`, `iqr`, `outlier_count`, `zero_count`, `inferred_type`, `top_values` |
| `AlertConfig(...)`                                  | Configure alert thresholds                                                                                                                                                                                                                                               |
| `DataProfile.to_dict()` / `DataProfile.from_dict()` | Serialization round-trip                                                                                                                                                                                                                                                 |
| `ColumnProfile.from_dict()`                         | Deserialize a column profile (validates required fields)                                                                                                                                                                                                                 |

## Code Walkthrough

### Step 1: Create a DataFrame

DataExplorer works exclusively with polars DataFrames. Here we create a small dataset with numeric, categorical, boolean, and nullable columns:

```python
import polars as pl

df = pl.DataFrame({
    "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    "income": [30000.0, 45000.0, 55000.0, 60000.0, 70000.0,
               80000.0, 90000.0, 100000.0, 110000.0, 120000.0],
    "department": ["eng", "eng", "sales", "sales", "hr",
                   "hr", "eng", "sales", "hr", "eng"],
    "is_manager": [False, False, True, False, True,
                   False, True, True, False, True],
    "score": [88.0, None, 72.0, 95.0, None, 63.0, 77.0, 81.0, None, 90.0],
})
```

### Step 2: Instantiate and Profile

```python
from kailash_ml.engines.data_explorer import DataExplorer, DataProfile

explorer = DataExplorer()
profile = await explorer.profile(df)

assert profile.n_rows == 10
assert profile.n_columns == 5
```

### Step 3: Inspect Column Statistics

Each column in `profile.columns` is a `ColumnProfile` with rich statistics:

```python
age_col = next(cp for cp in profile.columns if cp.name == "age")

# Basic stats
assert age_col.count == 10
assert age_col.null_count == 0
assert age_col.mean is not None

# Extended stats
assert age_col.skewness is not None
assert age_col.kurtosis is not None
assert age_col.iqr is not None
assert age_col.outlier_count is not None

# Cardinality
assert age_col.cardinality_ratio == age_col.unique_count / age_col.count
```

### Step 4: Check Type Inference

```python
assert age_col.inferred_type == "numeric"

dept_col = next(cp for cp in profile.columns if cp.name == "department")
assert dept_col.inferred_type == "categorical"
assert dept_col.top_values is not None  # value counts for string columns
```

### Step 5: Examine Correlation Matrices

```python
assert profile.correlation_matrix is not None
assert abs(profile.correlation_matrix["age"]["age"] - 1.0) < 0.01  # self-correlation
assert profile.spearman_matrix is not None
```

### Step 6: Configure Strict Alerts

```python
from kailash_ml.engines.data_explorer import AlertConfig

strict_config = AlertConfig(
    high_null_pct_threshold=0.01,    # Alert at >1% nulls
    high_correlation_threshold=0.8,
    skewness_threshold=1.5,
    zero_pct_threshold=0.3,
    high_cardinality_ratio=0.8,
    imbalance_ratio_threshold=0.2,
    duplicate_pct_threshold=0.0,
)

explorer_strict = DataExplorer(alert_config=strict_config)
strict_profile = await explorer_strict.profile(df)

alert_types = {a["type"] for a in strict_profile.alerts}
assert "high_nulls" in alert_types  # "score" has 30% nulls
```

### Step 7: Profile a Subset of Columns

```python
partial = await explorer.profile(df, columns=["age", "income"])
assert partial.n_columns == 2
```

### Step 8: Compare Two Datasets

```python
comparison = await explorer.compare(df.head(7), df.tail(5))

assert comparison["shape_comparison"]["rows_a"] == 7
assert comparison["shape_comparison"]["rows_b"] == 5
assert "column_deltas" in comparison
```

### Step 9: Serialization

```python
profile_dict = profile.to_dict()
restored = DataProfile.from_dict(profile_dict)
assert restored.n_rows == profile.n_rows
```

## Common Mistakes

| Mistake                                                 | What Happens                                       | Fix                                                                  |
| ------------------------------------------------------- | -------------------------------------------------- | -------------------------------------------------------------------- |
| Passing a pandas DataFrame                              | `TypeError`                                        | Use `polars.DataFrame` only                                          |
| Forgetting `await` on `profile()`                       | Returns a coroutine, not a `DataProfile`           | Always `await explorer.profile(df)`                                  |
| Ignoring alerts                                         | Data quality issues silently propagate to training | Check `profile.alerts` and address them                              |
| Using default thresholds for all domains                | Alerts may be too lenient or too strict            | Customize `AlertConfig` for your domain                              |
| Calling `ColumnProfile.from_dict()` with missing fields | `ValueError`                                       | Ensure `count`, `null_count`, `null_pct`, `unique_count` are present |

## Exercises

1. **Profile a real dataset**: Load any CSV with `polars.read_csv()`, profile it with DataExplorer, and list every column that has more than 5% nulls. How many columns are `numeric` vs `categorical`?

2. **Tune AlertConfig**: Create an AlertConfig where the skewness threshold is 1.0 and the null threshold is 0.05. Profile a dataset and explain each alert that fires.

3. **Compare train/test splits**: Split a DataFrame 80/20, run `explorer.compare()`, and examine the `column_deltas`. Are the distributions similar enough for reliable model evaluation?

## Key Takeaways

- DataExplorer automates the entire descriptive-statistics phase of data analysis.
- `profile()` returns a `DataProfile` with per-column stats, correlations, missing patterns, and alerts.
- Semantic type inference (`numeric`, `categorical`, `constant`, etc.) helps you understand each column without manual inspection.
- AlertConfig lets you set domain-specific thresholds for eight data-quality alert types.
- `compare()` gives you a quick distribution-shift check between any two DataFrames.
- All result objects support `to_dict()` / `from_dict()` for serialization and reporting.

## Next Chapter

Chapter 2 covers **PreprocessingPipeline** -- the engine that takes the raw data you just profiled and prepares it for training by imputing missing values, encoding categoricals, scaling numerics, and splitting into train/test sets.
