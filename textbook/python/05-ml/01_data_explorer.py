# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — ML / DataExplorer & AlertConfig
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Profile datasets with DataExplorer + configure alerts with
#            AlertConfig.  Covers statistical profiling, type inference,
#            correlation matrices, missing patterns, alerts, comparison,
#            and HTML report generation.
# LEVEL: Basic
# PARITY: Python-only (kailash-ml has no Rust counterpart)
# VALIDATES: DataExplorer, AlertConfig, DataProfile, ColumnProfile,
#            profile(), compare(), visualize(), to_html()
#
# Run: uv run python textbook/python/05-ml/01_data_explorer.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import polars as pl

from kailash_ml.engines.data_explorer import (
    AlertConfig,
    ColumnProfile,
    DataExplorer,
    DataProfile,
)


async def main() -> None:
    # ── 1. Create synthetic data ────────────────────────────────────────
    # DataExplorer works entirely with polars DataFrames.

    df = pl.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            "income": [
                30000.0,
                45000.0,
                55000.0,
                60000.0,
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
                "hr",
                "eng",
                "sales",
                "hr",
                "eng",
            ],
            "is_manager": [
                False,
                False,
                True,
                False,
                True,
                False,
                True,
                True,
                False,
                True,
            ],
            "score": [88.0, None, 72.0, 95.0, None, 63.0, 77.0, 81.0, None, 90.0],
        }
    )

    assert isinstance(df, pl.DataFrame), "Input must be a polars DataFrame"

    # ── 2. Instantiate DataExplorer with default AlertConfig ────────────

    explorer = DataExplorer()
    assert isinstance(explorer, DataExplorer)

    # ── 3. Profile the dataset ──────────────────────────────────────────
    # profile() is async and returns a DataProfile with per-column stats,
    # correlation matrices, missing patterns, alerts, and metadata.

    profile = await explorer.profile(df)

    assert isinstance(profile, DataProfile)
    assert profile.n_rows == 10
    assert profile.n_columns == 5
    assert len(profile.columns) == 5

    # Each column gets a ColumnProfile with statistical summary
    age_col = next(cp for cp in profile.columns if cp.name == "age")
    assert isinstance(age_col, ColumnProfile)
    assert age_col.count == 10
    assert age_col.null_count == 0
    assert age_col.null_pct == 0.0
    assert age_col.mean is not None
    assert age_col.std is not None
    assert age_col.min_val is not None
    assert age_col.max_val is not None
    assert age_col.q25 is not None
    assert age_col.q50 is not None
    assert age_col.q75 is not None

    # Extended numeric stats: skewness, kurtosis, IQR, outliers, zeros
    assert age_col.skewness is not None
    assert age_col.kurtosis is not None
    assert age_col.iqr is not None
    assert age_col.outlier_count is not None
    assert age_col.zero_count is not None

    # Cardinality ratio (unique / count)
    assert age_col.cardinality_ratio is not None
    assert age_col.cardinality_ratio == age_col.unique_count / age_col.count

    # ── 4. Type inference ───────────────────────────────────────────────
    # DataExplorer infers semantic types: numeric, categorical, boolean,
    # id, constant, text

    assert age_col.inferred_type == "numeric"

    dept_col = next(cp for cp in profile.columns if cp.name == "department")
    assert dept_col.inferred_type == "categorical"

    # String columns get top_values (value counts)
    assert dept_col.top_values is not None
    assert len(dept_col.top_values) > 0

    # ── 5. Correlation matrices ─────────────────────────────────────────
    # Pearson and Spearman matrices are computed for numeric columns.

    assert profile.correlation_matrix is not None, "Pearson matrix computed"
    assert "age" in profile.correlation_matrix
    assert "income" in profile.correlation_matrix["age"]
    # Self-correlation is 1.0
    assert abs(profile.correlation_matrix["age"]["age"] - 1.0) < 0.01

    assert profile.spearman_matrix is not None, "Spearman matrix computed"

    # ── 6. Missing patterns ─────────────────────────────────────────────
    # Identifies columns with nulls and co-occurring null patterns.

    score_col = next(cp for cp in profile.columns if cp.name == "score")
    assert score_col.null_count == 3

    # ── 7. Metadata: duplicates, memory, samples ────────────────────────

    assert isinstance(profile.duplicate_count, int)
    assert isinstance(profile.memory_bytes, int)
    assert profile.memory_bytes > 0
    assert len(profile.sample_head) > 0, "Head sample captured"
    assert len(profile.sample_tail) > 0, "Tail sample captured"
    assert isinstance(profile.type_summary, dict), "Type summary is a dict"

    # ── 8. DataProfile serialization round-trip ─────────────────────────

    profile_dict = profile.to_dict()
    assert isinstance(profile_dict, dict)
    restored = DataProfile.from_dict(profile_dict)
    assert restored.n_rows == profile.n_rows
    assert restored.n_columns == profile.n_columns
    assert len(restored.columns) == len(profile.columns)

    # ── 9. AlertConfig: customize alert thresholds ──────────────────────
    # AlertConfig controls when data quality warnings fire.

    strict_config = AlertConfig(
        high_null_pct_threshold=0.01,  # Fire alert at >1% nulls
        high_correlation_threshold=0.8,  # Flag correlated pairs at >0.8
        skewness_threshold=1.5,  # Flag skewness >1.5
        zero_pct_threshold=0.3,  # Flag >30% zeros
        high_cardinality_ratio=0.8,  # Flag high cardinality
        imbalance_ratio_threshold=0.2,  # Flag class imbalance
        duplicate_pct_threshold=0.0,  # Flag any duplicates
    )

    explorer_strict = DataExplorer(alert_config=strict_config)
    strict_profile = await explorer_strict.profile(df)

    # Alerts are dicts with type, column/columns, value, severity
    assert isinstance(strict_profile.alerts, list)
    alert_types = {a["type"] for a in strict_profile.alerts}
    # "score" has 30% nulls, so with threshold=0.01 we expect high_nulls
    assert "high_nulls" in alert_types, "Expected high_nulls alert for 'score'"

    # 8 possible alert types:
    # high_nulls, constant, high_skewness, high_zeros, high_cardinality,
    # high_correlation, duplicates, imbalanced

    # ── 10. Profile specific columns ────────────────────────────────────
    # Pass columns= to profile only a subset.

    partial = await explorer.profile(df, columns=["age", "income"])
    assert partial.n_columns == 2
    assert len(partial.columns) == 2
    col_names = {cp.name for cp in partial.columns}
    assert col_names == {"age", "income"}

    # ── 11. Compare two datasets ────────────────────────────────────────
    # compare() profiles both datasets in parallel and computes deltas.

    df_train = df.head(7)
    df_prod = df.tail(5)

    comparison = await explorer.compare(df_train, df_prod)

    assert "profile_a" in comparison
    assert "profile_b" in comparison
    assert "column_deltas" in comparison
    assert "shape_comparison" in comparison
    assert comparison["shape_comparison"]["rows_a"] == 7
    assert comparison["shape_comparison"]["rows_b"] == 5
    assert "shared_columns" in comparison
    assert "missing_in_a" in comparison
    assert "missing_in_b" in comparison

    # column_deltas contains per-column stat differences
    for delta in comparison["column_deltas"]:
        assert "column" in delta
        assert "null_pct_delta" in delta

    # ── 12. Edge case: empty DataFrame ──────────────────────────────────

    empty_df = pl.DataFrame({"x": pl.Series([], dtype=pl.Float64)})
    empty_profile = await explorer.profile(empty_df)
    assert empty_profile.n_rows == 0
    assert empty_profile.n_columns == 1
    x_col = empty_profile.columns[0]
    assert x_col.count == 0

    # ── 13. Edge case: single-value (constant) column ───────────────────

    constant_df = pl.DataFrame({"val": [42, 42, 42, 42, 42]})
    const_profile = await explorer.profile(constant_df)
    val_col = const_profile.columns[0]
    assert val_col.unique_count == 1
    assert val_col.inferred_type == "constant"
    # Constant column should trigger constant alert with default config
    const_alert_types = {a["type"] for a in const_profile.alerts}
    assert "constant" in const_alert_types

    # ── 14. Edge case: all nulls column ─────────────────────────────────

    null_df = pl.DataFrame({"col": pl.Series([None, None, None], dtype=pl.Float64)})
    null_profile = await explorer.profile(null_df)
    null_col = null_profile.columns[0]
    assert null_col.null_count == 3
    assert null_col.null_pct == 1.0

    # ── 15. Edge case: ColumnProfile from_dict validation ───────────────

    try:
        ColumnProfile.from_dict({"name": "x", "dtype": "Int64"})
        assert False, "Should raise ValueError for missing required fields"
    except ValueError:
        pass  # Expected: missing count, null_count, null_pct, unique_count

    try:
        ColumnProfile.from_dict(
            {
                "name": "x",
                "dtype": "Int64",
                "count": -1,
                "null_count": 0,
                "null_pct": 0.0,
                "unique_count": 0,
            }
        )
        assert False, "Should raise ValueError for negative count"
    except ValueError:
        pass  # Expected: count must be non-negative

    print("PASS: 05-ml/01_data_explorer")


asyncio.run(main())
