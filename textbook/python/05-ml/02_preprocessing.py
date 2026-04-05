# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — ML / PreprocessingPipeline
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build preprocessing pipelines that auto-detect task type,
#            encode categoricals, scale numerics, impute missing values,
#            split train/test, and transform new data at inference time.
# LEVEL: Basic
# PARITY: Python-only (kailash-ml has no Rust counterpart)
# VALIDATES: PreprocessingPipeline, SetupResult — setup(), transform(),
#            inverse_transform(), get_config()
#
# Run: uv run python textbook/python/05-ml/02_preprocessing.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl

from kailash_ml.engines.preprocessing import PreprocessingPipeline, SetupResult

# ── 1. Create synthetic data ────────────────────────────────────────
# A classification dataset with numeric and categorical features,
# plus some missing values.

df = pl.DataFrame(
    {
        "age": [25.0, 30.0, None, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0],
        "income": [
            30000.0,
            45000.0,
            55000.0,
            None,
            70000.0,
            80000.0,
            90000.0,
            100000.0,
            110000.0,
            120000.0,
        ],
        "department": [
            "eng",
            "eng",
            "sales",
            "sales",
            "hr",
            None,
            "eng",
            "sales",
            "hr",
            "eng",
        ],
        "churned": [0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
    }
)

assert isinstance(df, pl.DataFrame)

# ── 2. Instantiate pipeline ─────────────────────────────────────────

pipeline = PreprocessingPipeline()
assert isinstance(pipeline, PreprocessingPipeline)

# ── 3. setup() — auto-detect and preprocess ─────────────────────────
# setup() detects task type, identifies column types, imputes, encodes,
# scales, and splits into train/test.

result = pipeline.setup(
    df,
    target="churned",
    train_size=0.8,
    seed=42,
    normalize=True,
    categorical_encoding="onehot",
    imputation_strategy="mean",
)

assert isinstance(result, SetupResult)

# Task type detection: "churned" has few unique values -> classification
assert result.task_type == "classification"
assert result.target_column == "churned"

# Column detection
assert "age" in result.numeric_columns
assert "income" in result.numeric_columns
assert "department" in result.categorical_columns

# Shape information
assert result.original_shape == (10, 4)
assert result.transformed_shape[0] <= 10  # May shrink from outlier removal
assert result.transformed_shape[1] >= 4  # One-hot expands columns

# Train/test split
assert isinstance(result.train_data, pl.DataFrame)
assert isinstance(result.test_data, pl.DataFrame)
assert result.train_data.height + result.test_data.height == df.height
assert result.train_data.height == 8  # 80% of 10
assert result.test_data.height == 2  # 20% of 10

# Transformers are stored for inference
assert isinstance(result.transformers, dict)

# Human-readable summary
assert isinstance(result.summary, str)
assert "classification" in result.summary

# ── 4. One-hot encoding verification ────────────────────────────────
# "department" should be replaced by department_eng, department_hr,
# department_sales columns.

train_cols = result.train_data.columns
assert "department" not in train_cols, "Original column removed after one-hot"
onehot_cols = [c for c in train_cols if c.startswith("department_")]
assert len(onehot_cols) >= 2, "One-hot columns created"

# ── 5. Missing value imputation ─────────────────────────────────────
# After imputation, no nulls in numeric columns.

for col in result.numeric_columns:
    if col in result.train_data.columns:
        assert (
            result.train_data[col].null_count() == 0
        ), f"Column '{col}' should have no nulls after imputation"

# ── 6. get_config() — inspect fitted configuration ──────────────────

config = pipeline.get_config()
assert isinstance(config, dict)
assert config["target"] == "churned"
assert config["task_type"] == "classification"
assert config["normalize"] is True
assert config["categorical_encoding"] == "onehot"
assert config["imputation_strategy"] == "mean"

# ── 7. transform() — apply to new data at inference time ────────────
# Uses the same imputation stats, encoding, and scaling fitted during
# setup().

new_data = pl.DataFrame(
    {
        "age": [33.0, None],
        "income": [52000.0, 68000.0],
        "department": ["eng", "hr"],
        "churned": [0, 1],
    }
)

transformed = pipeline.transform(new_data)
assert isinstance(transformed, pl.DataFrame)
# One-hot encoding applied to new data too
assert "department" not in transformed.columns
new_onehot = [c for c in transformed.columns if c.startswith("department_")]
assert len(new_onehot) >= 2

# ── 8. inverse_transform() — reverse scaling ────────────────────────
# Reverses StandardScaler for interpretability.

inversed = pipeline.inverse_transform(transformed)
assert isinstance(inversed, pl.DataFrame)

# ── 9. Ordinal encoding mode ────────────────────────────────────────

pipeline_ord = PreprocessingPipeline()
result_ord = pipeline_ord.setup(
    df,
    target="churned",
    categorical_encoding="ordinal",
    normalize=False,
)

# Ordinal encoding keeps the column name but with integer values
assert "department" in result_ord.train_data.columns
dept_dtype = result_ord.train_data["department"].dtype
assert dept_dtype == pl.Float64, "Ordinal encoding produces Float64 values"

# ── 10. Target encoding mode ────────────────────────────────────────

pipeline_te = PreprocessingPipeline()
result_te = pipeline_te.setup(
    df,
    target="churned",
    categorical_encoding="target",
    normalize=False,
)

# Target encoding replaces categories with mean target value
assert "department" in result_te.train_data.columns
assert result_te.train_data["department"].dtype == pl.Float64

# ── 11. Median imputation ───────────────────────────────────────────

pipeline_med = PreprocessingPipeline()
result_med = pipeline_med.setup(
    df,
    target="churned",
    imputation_strategy="median",
    normalize=False,
    categorical_encoding="ordinal",
)
assert result_med.task_type == "classification"

# ── 12. Drop imputation strategy ────────────────────────────────────

pipeline_drop = PreprocessingPipeline()
result_drop = pipeline_drop.setup(
    df,
    target="churned",
    imputation_strategy="drop",
    normalize=False,
    categorical_encoding="ordinal",
)
# Rows with nulls are dropped
total_rows = result_drop.train_data.height + result_drop.test_data.height
assert total_rows <= df.height, "Drop strategy removes rows with nulls"

# ── 13. Outlier removal ─────────────────────────────────────────────

pipeline_outlier = PreprocessingPipeline()
result_outlier = pipeline_outlier.setup(
    df,
    target="churned",
    remove_outliers=True,
    outlier_threshold=0.05,
    normalize=False,
    categorical_encoding="ordinal",
)
total_after_outlier = result_outlier.train_data.height + result_outlier.test_data.height
assert total_after_outlier <= df.height

# ── 14. Regression task detection ───────────────────────────────────

regression_df = pl.DataFrame(
    {
        "sqft": [800.0, 1200.0, 1500.0, 1800.0, 2200.0, 2600.0, 3000.0, 3500.0],
        "bedrooms": [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0],
        "price": [
            200000.0,
            350000.0,
            420000.0,
            510000.0,
            620000.0,
            730000.0,
            850000.0,
            980000.0,
        ],
    }
)

pipeline_reg = PreprocessingPipeline()
result_reg = pipeline_reg.setup(
    regression_df,
    target="price",
    normalize=True,
)
assert result_reg.task_type == "regression", "Many unique values -> regression"

# ── 15. Edge case: missing target column ─────────────────────────────

try:
    pipeline.setup(df, target="nonexistent_column")
    assert False, "Should raise ValueError for missing target"
except ValueError:
    pass  # Expected: target column not found

# ── 16. Edge case: empty DataFrame ───────────────────────────────────

try:
    empty_df = pl.DataFrame(
        {"x": pl.Series([], dtype=pl.Float64), "y": pl.Series([], dtype=pl.Int64)}
    )
    pipeline.setup(empty_df, target="y")
    assert False, "Should raise ValueError for empty data"
except ValueError:
    pass  # Expected: empty data

# ── 17. Edge case: invalid train_size ────────────────────────────────

try:
    pipeline.setup(df, target="churned", train_size=1.5)
    assert False, "Should raise ValueError for invalid train_size"
except ValueError:
    pass  # Expected: train_size must be between 0 and 1

# ── 18. Edge case: invalid encoding ─────────────────────────────────

try:
    pipeline.setup(df, target="churned", categorical_encoding="invalid")
    assert False, "Should raise ValueError for invalid encoding"
except ValueError:
    pass  # Expected: unknown encoding

# ── 19. Edge case: get_config before setup ───────────────────────────

fresh_pipeline = PreprocessingPipeline()
try:
    fresh_pipeline.get_config()
    assert False, "Should raise RuntimeError before setup()"
except RuntimeError:
    pass  # Expected: pipeline not fitted

# ── 20. Edge case: transform before setup ────────────────────────────

try:
    fresh_pipeline.transform(df)
    assert False, "Should raise RuntimeError before setup()"
except RuntimeError:
    pass  # Expected: pipeline not fitted

print("PASS: 05-ml/02_preprocessing")
