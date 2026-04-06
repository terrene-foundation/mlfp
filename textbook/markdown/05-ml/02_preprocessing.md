# Chapter 2: PreprocessingPipeline

## Overview

PreprocessingPipeline transforms raw data into ML-ready features with a single `setup()` call. It auto-detects whether your task is classification or regression, identifies column types, imputes missing values, encodes categoricals, scales numerics, removes outliers, and splits into train/test sets. At inference time, `transform()` applies the exact same transformations learned during training.

This chapter covers:

- Auto-detection of task type (classification vs regression)
- Column type identification (numeric, categorical)
- Imputation strategies: mean, median, drop
- Categorical encoding: one-hot, ordinal, target encoding
- Normalization (StandardScaler)
- Outlier removal
- Train/test splitting
- Inference-time `transform()` and `inverse_transform()`
- Pipeline configuration inspection with `get_config()`

## Prerequisites

| Requirement   | Details                   |
| ------------- | ------------------------- |
| Python        | 3.10+                     |
| kailash-ml    | `pip install kailash-ml`  |
| polars        | Installed with kailash-ml |
| Prior chapter | Chapter 1 (DataExplorer)  |
| Level         | Basic                     |

## Concepts

### Task Type Detection

PreprocessingPipeline examines the target column to determine task type. If the target has few unique values relative to the dataset size, it is classified as **classification**. If the target has many unique continuous values, it is **regression**. This detection drives downstream decisions like which encoding strategies and evaluation metrics are appropriate.

### Imputation Strategies

Missing values must be handled before training. Three strategies are available:

| Strategy | Behavior                                                             |
| -------- | -------------------------------------------------------------------- |
| `mean`   | Replace nulls with the column mean (numeric) or mode (categorical)   |
| `median` | Replace nulls with the column median (numeric) or mode (categorical) |
| `drop`   | Remove rows containing any null value                                |

The fitted imputation statistics (mean/median values) are stored in the pipeline so `transform()` uses the same values at inference time.

### Categorical Encoding

Three encoding strategies convert string columns into numeric values:

| Encoding  | Behavior                                   | Output                                        |
| --------- | ------------------------------------------ | --------------------------------------------- |
| `onehot`  | Creates binary columns for each category   | Expands column count; original column removed |
| `ordinal` | Maps categories to integers                | Same column, Float64 dtype                    |
| `target`  | Replaces categories with mean target value | Same column, Float64 dtype                    |

### Normalization

When `normalize=True`, StandardScaler is applied to numeric columns, centering them to mean=0 and scaling to std=1. The scaler parameters are saved for `inverse_transform()`.

### Outlier Removal

When `remove_outliers=True`, rows with extreme values (beyond the `outlier_threshold` percentile) are removed. This can reduce the dataset size.

## Key API Reference

| Class / Method                     | Purpose                                                                                                                                                                                              |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PreprocessingPipeline()`          | Instantiate the pipeline                                                                                                                                                                             |
| `pipeline.setup(df, target, ...)`  | Fit and transform: detect types, impute, encode, scale, split                                                                                                                                        |
| `pipeline.transform(new_data)`     | Apply fitted transformations to new data                                                                                                                                                             |
| `pipeline.inverse_transform(data)` | Reverse scaling for interpretability                                                                                                                                                                 |
| `pipeline.get_config()`            | Inspect fitted configuration                                                                                                                                                                         |
| `SetupResult`                      | Result of `setup()` with fields: `task_type`, `target_column`, `numeric_columns`, `categorical_columns`, `original_shape`, `transformed_shape`, `train_data`, `test_data`, `transformers`, `summary` |

### setup() Parameters

| Parameter              | Type           | Default    | Description                           |
| ---------------------- | -------------- | ---------- | ------------------------------------- |
| `df`                   | `pl.DataFrame` | required   | Input data                            |
| `target`               | `str`          | required   | Target column name                    |
| `train_size`           | `float`        | `0.8`      | Fraction for training (0 to 1)        |
| `seed`                 | `int`          | `None`     | Random seed for reproducibility       |
| `normalize`            | `bool`         | `False`    | Apply StandardScaler                  |
| `categorical_encoding` | `str`          | `"onehot"` | One of: `onehot`, `ordinal`, `target` |
| `imputation_strategy`  | `str`          | `"mean"`   | One of: `mean`, `median`, `drop`      |
| `remove_outliers`      | `bool`         | `False`    | Remove outlier rows                   |
| `outlier_threshold`    | `float`        | `0.05`     | Percentile threshold for outliers     |

## Code Walkthrough

### Step 1: Create Data with Missing Values

```python
import polars as pl

df = pl.DataFrame({
    "age": [25.0, 30.0, None, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0],
    "income": [30000.0, 45000.0, 55000.0, None, 70000.0, 80000.0,
               90000.0, 100000.0, 110000.0, 120000.0],
    "department": ["eng", "eng", "sales", "sales", "hr", None,
                   "eng", "sales", "hr", "eng"],
    "churned": [0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
})
```

### Step 2: Setup with One-Hot Encoding

```python
from kailash_ml.engines.preprocessing import PreprocessingPipeline, SetupResult

pipeline = PreprocessingPipeline()
result = pipeline.setup(
    df,
    target="churned",
    train_size=0.8,
    seed=42,
    normalize=True,
    categorical_encoding="onehot",
    imputation_strategy="mean",
)

assert result.task_type == "classification"
assert result.train_data.height == 8   # 80% of 10
assert result.test_data.height == 2    # 20% of 10
```

### Step 3: Verify One-Hot Encoding

```python
train_cols = result.train_data.columns
assert "department" not in train_cols  # Original column removed
onehot_cols = [c for c in train_cols if c.startswith("department_")]
assert len(onehot_cols) >= 2  # department_eng, department_hr, etc.
```

### Step 4: Verify Missing Value Imputation

```python
for col in result.numeric_columns:
    if col in result.train_data.columns:
        assert result.train_data[col].null_count() == 0
```

### Step 5: Inspect Pipeline Configuration

```python
config = pipeline.get_config()
assert config["target"] == "churned"
assert config["task_type"] == "classification"
assert config["normalize"] is True
assert config["categorical_encoding"] == "onehot"
```

### Step 6: Transform New Data at Inference Time

```python
new_data = pl.DataFrame({
    "age": [33.0, None],
    "income": [52000.0, 68000.0],
    "department": ["eng", "hr"],
    "churned": [0, 1],
})

transformed = pipeline.transform(new_data)
assert "department" not in transformed.columns  # Same encoding applied
```

### Step 7: Inverse Transform for Interpretability

```python
inversed = pipeline.inverse_transform(transformed)
# Reverses StandardScaler so values are in original units
```

### Step 8: Try Different Encoding Modes

```python
# Ordinal encoding
pipeline_ord = PreprocessingPipeline()
result_ord = pipeline_ord.setup(df, target="churned", categorical_encoding="ordinal")
assert result_ord.train_data["department"].dtype == pl.Float64

# Target encoding
pipeline_te = PreprocessingPipeline()
result_te = pipeline_te.setup(df, target="churned", categorical_encoding="target")
assert result_te.train_data["department"].dtype == pl.Float64
```

### Step 9: Regression Task Detection

```python
regression_df = pl.DataFrame({
    "sqft": [800.0, 1200.0, 1500.0, 1800.0, 2200.0, 2600.0, 3000.0, 3500.0],
    "bedrooms": [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0],
    "price": [200000.0, 350000.0, 420000.0, 510000.0,
              620000.0, 730000.0, 850000.0, 980000.0],
})

pipeline_reg = PreprocessingPipeline()
result_reg = pipeline_reg.setup(regression_df, target="price", normalize=True)
assert result_reg.task_type == "regression"
```

## Common Mistakes

| Mistake                                 | What Happens   | Fix                                        |
| --------------------------------------- | -------------- | ------------------------------------------ |
| Calling `transform()` before `setup()`  | `RuntimeError` | Always call `setup()` first                |
| Calling `get_config()` before `setup()` | `RuntimeError` | Fit the pipeline before inspecting config  |
| Setting `train_size` outside (0, 1)     | `ValueError`   | Keep between 0 and 1 exclusive             |
| Using an invalid encoding name          | `ValueError`   | Use `onehot`, `ordinal`, or `target`       |
| Specifying a non-existent target column | `ValueError`   | Verify column name exists in the DataFrame |
| Passing an empty DataFrame              | `ValueError`   | Ensure data has at least one row           |

## Exercises

1. **Compare imputation strategies**: Run `setup()` three times on the same dataset with `mean`, `median`, and `drop` imputation. Compare the resulting `train_data.height` and the values filled in for formerly-null cells.

2. **Encoding impact on column count**: Start with a categorical column that has 8 unique values. Run `setup()` with `onehot` vs `ordinal` encoding. How does `transformed_shape[1]` differ? When would you prefer ordinal?

3. **Round-trip verification**: Apply `setup()` with `normalize=True`, then call `inverse_transform()` on the training data. Verify that the values are close to the originals.

## Key Takeaways

- PreprocessingPipeline automates the entire data preparation workflow: imputation, encoding, scaling, splitting.
- Task type (classification vs regression) is auto-detected from the target column.
- Three imputation strategies and three encoding modes cover common data preparation needs.
- The pipeline stores fitted transformers so `transform()` applies identical processing at inference time.
- `inverse_transform()` reverses scaling for human-readable model interpretation.
- Edge cases (missing target, empty data, invalid parameters) are caught with clear error messages.

## Next Chapter

Chapter 3 covers **ModelVisualizer** -- the engine that creates interactive Plotly visualizations for confusion matrices, ROC curves, feature importance, residuals, and more.
