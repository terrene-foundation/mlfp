# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — ML / FeatureEngineer
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Generate and select features using the two-step pattern:
#            generate() produces candidates, select() ranks and picks
#            the best subset.  Covers interaction, polynomial, binning,
#            and temporal strategies, plus importance, correlation, and
#            mutual information selection methods.
# LEVEL: Intermediate
# PARITY: Python-only (kailash-ml has no Rust counterpart)
# VALIDATES: FeatureEngineer, GeneratedFeatures, GeneratedColumn,
#            SelectedFeatures, FeatureRank — generate() then select()
#
# Run: uv run python textbook/python/05-ml/04_feature_engineer.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl

from kailash_ml import FeatureField, FeatureSchema
from kailash_ml.engines.feature_engineer import (
    FeatureEngineer,
    FeatureRank,
    GeneratedColumn,
    GeneratedFeatures,
    SelectedFeatures,
)

# ── 1. Create synthetic data ────────────────────────────────────────

df = pl.DataFrame(
    {
        "entity_id": [f"u{i}" for i in range(100)],
        "age": [20 + (i % 50) for i in range(100)],
        "income": [30000.0 + i * 1000.0 for i in range(100)],
        "tenure_months": [1 + (i * 3 % 60) for i in range(100)],
        "num_products": [1 + (i % 5) for i in range(100)],
        "churned": [(i % 3 == 0) for i in range(100)],
    }
)
# Cast boolean target to int for ML
df = df.with_columns(pl.col("churned").cast(pl.Int64))

# ── 2. Define the FeatureSchema ─────────────────────────────────────
# Schema declares which columns are features and their dtypes.

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

# ── 3. Instantiate FeatureEngineer ──────────────────────────────────

engineer = FeatureEngineer(max_features=10)
assert isinstance(engineer, FeatureEngineer)

# ── 4. Step 1: generate() — produce candidate features ──────────────
# generate(data, schema, strategies=[...]) returns GeneratedFeatures
# containing the augmented DataFrame and metadata about each new column.

candidates = engineer.generate(
    df,
    schema,
    strategies=["interactions", "polynomial", "binning"],
)

assert isinstance(candidates, GeneratedFeatures)
assert isinstance(candidates.data, pl.DataFrame)

# Original columns preserved
assert "age" in candidates.data.columns
assert "income" in candidates.data.columns

# Generated columns added
assert len(candidates.generated_columns) > 0
assert candidates.total_candidates > len(schema.features)

# Each generated column has metadata
for gc in candidates.generated_columns:
    assert isinstance(gc, GeneratedColumn)
    assert gc.name in candidates.data.columns
    assert gc.strategy in ("interaction", "polynomial", "binning")
    assert len(gc.source_columns) >= 1
    assert gc.dtype in ("float64", "categorical")

# ── 5. Inspect generated features by strategy ───────────────────────

interactions = [g for g in candidates.generated_columns if g.strategy == "interaction"]
polynomials = [g for g in candidates.generated_columns if g.strategy == "polynomial"]
binned = [g for g in candidates.generated_columns if g.strategy == "binning"]

# With 4 numeric features: C(4,2) = 6 interaction pairs
assert len(interactions) == 6, f"Expected 6 interactions, got {len(interactions)}"

# 4 polynomial (squared) columns
assert len(polynomials) == 4, f"Expected 4 polynomials, got {len(polynomials)}"

# 4 binned columns (one per numeric feature)
assert len(binned) == 4, f"Expected 4 binned, got {len(binned)}"

# Interaction columns are products of two source columns
for g in interactions:
    assert "_x_" in g.name, "Interaction columns use _x_ naming"
    assert len(g.source_columns) == 2

# Polynomial columns are squared
for g in polynomials:
    assert g.name.endswith("_squared"), "Polynomial columns use _squared suffix"
    assert len(g.source_columns) == 1

# Binned columns use qcut
for g in binned:
    assert g.name.endswith("_binned"), "Binned columns use _binned suffix"

# ── 6. Step 2: select() — rank and choose best features ─────────────
# select(data, candidates, target, method=, top_k=) returns
# SelectedFeatures with rankings and selected/dropped column lists.

selected = engineer.select(
    candidates.data,
    candidates,
    target="churned",
    method="importance",
    top_k=8,
)

assert isinstance(selected, SelectedFeatures)
assert len(selected.selected_columns) <= 8
assert len(selected.rankings) > 0
assert isinstance(selected.method, str)
assert selected.method == "importance"
assert selected.n_original == len(schema.features)
assert selected.n_generated == len(candidates.generated_columns)
assert selected.n_selected == len(selected.selected_columns)

# Rankings are ordered by score descending
for i in range(len(selected.rankings) - 1):
    assert selected.rankings[i].score >= selected.rankings[i + 1].score

# Each ranking has metadata
for rank in selected.rankings:
    assert isinstance(rank, FeatureRank)
    assert rank.rank >= 1
    assert rank.score >= 0.0
    assert rank.source in ("original", "generated")

# Selected + dropped = all ranked features
assert len(selected.selected_columns) + len(selected.dropped_columns) == len(
    selected.rankings
)

# ── 7. Correlation-based selection ──────────────────────────────────

selected_corr = engineer.select(
    candidates.data,
    candidates,
    target="churned",
    method="correlation",
    top_k=5,
)

assert selected_corr.method == "correlation"
assert len(selected_corr.selected_columns) <= 5

# ── 8. Mutual information selection ─────────────────────────────────

selected_mi = engineer.select(
    candidates.data,
    candidates,
    target="churned",
    method="mutual_info",
    top_k=5,
)

assert selected_mi.method == "mutual_info"
assert len(selected_mi.selected_columns) <= 5

# ── 9. Selective strategies ─────────────────────────────────────────
# Generate with only polynomial features

poly_only = engineer.generate(
    df,
    schema,
    strategies=["polynomial"],
)

poly_cols = [g for g in poly_only.generated_columns if g.strategy == "polynomial"]
inter_cols = [g for g in poly_only.generated_columns if g.strategy == "interaction"]
assert len(poly_cols) == 4
assert len(inter_cols) == 0, "No interactions when not requested"

# ── 10. Serialization round-trip ─────────────────────────────────────

# GeneratedColumn
gc_dict = candidates.generated_columns[0].to_dict()
gc_restored = GeneratedColumn.from_dict(gc_dict)
assert gc_restored.name == candidates.generated_columns[0].name
assert gc_restored.strategy == candidates.generated_columns[0].strategy

# GeneratedFeatures (note: data is not serializable)
gf_dict = candidates.to_dict()
assert gf_dict["data"] is None  # polars DataFrame not serializable
gf_restored = GeneratedFeatures.from_dict(gf_dict)
assert gf_restored.total_candidates == candidates.total_candidates

# FeatureRank
rank_dict = selected.rankings[0].to_dict()
rank_restored = FeatureRank.from_dict(rank_dict)
assert rank_restored.column_name == selected.rankings[0].column_name

# SelectedFeatures
sf_dict = selected.to_dict()
sf_restored = SelectedFeatures.from_dict(sf_dict)
assert sf_restored.n_selected == selected.n_selected
assert sf_restored.method == selected.method

# ── 11. Edge case: data with only one numeric column ────────────────
# Interactions require at least 2 numeric columns.

single_schema = FeatureSchema(
    name="single",
    features=[FeatureField(name="value", dtype="float64")],
    entity_id_column="entity_id",
)

single_df = pl.DataFrame(
    {
        "entity_id": ["a", "b", "c"],
        "value": [1.0, 2.0, 3.0],
        "target": [0, 1, 0],
    }
)

single_candidates = engineer.generate(
    single_df,
    single_schema,
    strategies=["interactions", "polynomial"],
)

# No interactions with single column, but polynomial still works
single_interactions = [
    g for g in single_candidates.generated_columns if g.strategy == "interaction"
]
single_poly = [
    g for g in single_candidates.generated_columns if g.strategy == "polynomial"
]
assert len(single_interactions) == 0
assert len(single_poly) == 1

# ── 12. Edge case: invalid selection method ──────────────────────────

try:
    engineer.select(
        candidates.data,
        candidates,
        target="churned",
        method="invalid_method",
    )
    assert False, "Should raise ValueError for unknown method"
except ValueError:
    pass  # Expected: Unknown selection method

# ── 13. Edge case: max_features limits output ───────────────────────

small_engineer = FeatureEngineer(max_features=3)
small_selected = small_engineer.select(
    candidates.data,
    candidates,
    target="churned",
    method="importance",
)
# When top_k is None, max_features is used
assert len(small_selected.selected_columns) <= 3

print("PASS: 05-ml/04_feature_engineer")
