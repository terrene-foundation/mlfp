# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 8.1: FeatureSchema — Typed Feature Contracts
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Define a FeatureSchema with typed FeatureField entries
#   - Compute base property features from raw HDB resale data
#   - Validate feature VALUES with rules the dtype schema cannot express
#   - Materialise v1 features into a FeatureStore and read them back
#   - Apply schema-driven feature engineering to Singapore HDB valuation
#
# PREREQUISITES: MLFP02 Exercises 1-7 (Bayesian inference, hypothesis
#   testing, regression, causal inference)
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — why typed feature schemas prevent silent failures
#   2. Build — define FeatureSchema v1 and compute features
#   3. Train — materialise features into the FeatureStore and read back
#   4. Visualise — feature distributions and correlation structure
#   5. Apply — HDB flat valuation with schema-validated features
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import polars as pl
import plotly.graph_objects as go
from scipy import stats

from shared.mlfp02.ex_8 import (
    OUTPUT_DIR,
    build_schema_v1,
    compute_v1_features,
    create_feature_store,
    load_hdb_resale,
    materialize_features,
    to_store_frame,
    validate_v1_features,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Typed Feature Schemas Prevent Silent Failures
# ════════════════════════════════════════════════════════════════════════
# A feature schema is a contract between the producer (feature engineer)
# and the consumer (model trainer). Without it, the following failures
# happen silently:
#
#   1. Dtype drift — a feature that was float64 in training becomes
#      string after a data source migration. The model loads, runs, and
#      produces garbage predictions without any error.
#
#   2. Null smuggling — a feature declared "never null" starts getting
#      nulls from a new data partition. The model NaN-propagates through
#      every prediction silently.
#
#   3. Phantom columns — an upstream process renames "price_per_sqm" to
#      "price_sqm". The old column vanishes, the model falls back to
#      defaults, and no one notices until the accuracy report.
#
# A typed schema catches phantom columns and dtype drift when features
# are written to the store, not at prediction time. Value-level rules
# (a lease cannot exceed 99 years, a price cannot be $10) need explicit
# validation on top — Task 2 adds them. Think of both as unit tests for
# your feature contract.
#
# Singapore HDB analogy: HDB publishes standard flat categories
# (3-room, 4-room, 5-room). If a listing arrives as "four-room" instead
# of "4 ROOM", every downstream system that filters by flat_type fails.
# The schema catches this at ingestion, not at report generation.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Define FeatureSchema v1 and compute property features
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Exercise 8.1 — FeatureSchema: Typed Feature Contracts")
print("=" * 70)

# --- 2a. Load HDB resale data ---
hdb = load_hdb_resale()
print(f"\n  Data loaded: {hdb.shape[0]:,} HDB resale transactions")

# --- 2b. Exploratory statistics (Ex 1-2 recap) ---
prices = hdb["resale_price"].to_numpy().astype(np.float64)
skew = stats.skew(prices)
kurt = stats.kurtosis(prices)
sw_stat, sw_p = stats.shapiro(
    np.random.default_rng(42).choice(prices, size=5000, replace=False)
)
mle_mu = prices.mean()
mle_sigma = prices.std(ddof=0)

print(f"\n  Price distribution:")
print(f"    n = {len(prices):,}")
print(f"    Mean: ${mle_mu:,.0f}, Median: ${np.median(prices):,.0f}")
print(f"    Std: ${mle_sigma:,.0f}")
print(
    f"    Skewness: {skew:.3f} "
    f"({'right-skewed' if skew > 0.5 else 'approximately symmetric'})"
)
print(
    f"    Excess kurtosis: {kurt:.3f} "
    f"({'heavy-tailed' if kurt > 1 else 'normal-tailed'})"
)
print(f"    Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.6f}")
print(f"\n  MLE (Normal): mu={mle_mu:,.0f}, sigma={mle_sigma:,.0f}")

# --- 2c. Define FeatureSchema v1 ---
property_schema_v1 = build_schema_v1()

print(f"\n  === FeatureSchema v1 ===")
print(f"  Name: {property_schema_v1.name}, Version: {property_schema_v1.version}")
for f in property_schema_v1.fields:
    print(f"    {f.name}: {f.dtype} (nullable={f.nullable}) — {f.description}")

# --- 2d. Compute v1 features ---
features_v1 = compute_v1_features(hdb)
print(f"\n  Computed v1 features: {features_v1.shape}")

for feat_name in ["price_per_sqm", "storey_midpoint", "remaining_lease_years"]:
    vals = features_v1[feat_name].drop_nulls()
    print(
        f"    {feat_name}: mean={vals.mean():.1f}, "
        f"min={vals.min():.1f}, max={vals.max():.1f}"
    )

# --- 2e. Value-level validation (rules a dtype cannot express) ---
features_v1_valid, violations = validate_v1_features(features_v1)
print(f"\n  Validation rules:")
for rule, n_bad in violations.items():
    print(f"    {rule:<32} {n_bad:>6,} rows")
n_dropped = features_v1.height - features_v1_valid.height
print(f"  Valid rows: {features_v1_valid.height:,} ({n_dropped:,} rejected)")
# INTERPRETATION: Every rejected row passed the dtype contract — a lease of
# 107 years is a perfectly good float64. Value rules catch what types
# cannot: impossible leases (a flat sold before its lease began) and
# sentinel prices such as $10 or $9,000,000.

# --- 2f. Correlation analysis (on validated rows) ---
corr_cols = [
    "resale_price",
    "floor_area_sqm",
    "storey_midpoint",
    "remaining_lease_years",
]
corr_data = (
    features_v1_valid.drop_nulls(subset=corr_cols)
    .select(corr_cols)
    .to_numpy()
    .astype(np.float64)
)
corr_matrix = np.corrcoef(corr_data.T)

print(f"\n  Correlation matrix:")
print(f"  {'':>20}", end="")
for c in corr_cols:
    print(f"  {c[:12]:>12}", end="")
print()
for i, name in enumerate(corr_cols):
    print(f"  {name[:20]:<20}", end="")
    for j in range(len(corr_cols)):
        print(f"  {corr_matrix[i,j]:>12.3f}", end="")
    print()


# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert len(prices) > 0, "Task 2: must have price data"
assert "transaction_id" in features_v1.columns, "Task 2: transaction_id missing"
assert "price_per_sqm" in features_v1.columns, "Task 2: price_per_sqm missing"
assert features_v1["price_per_sqm"].min() > 0, "Task 2: price_per_sqm must be positive"
assert corr_matrix.shape == (4, 4), "Task 2: correlation matrix must be 4x4"
assert features_v1_valid["remaining_lease_years"].max() <= 99, "Task 2: lease must be <= 99 years"
assert 0 < features_v1_valid.height < features_v1.height, "Task 2: validation should reject some rows"
print("\n[ok] Checkpoint 1 passed — v1 features computed and validated\n")

# INTERPRETATION: The schema declares 4 features — all non-nullable,
# all float64. Together with the value rules, this contract means any
# downstream model can trust that these columns exist, have no nulls,
# and hold physically possible values. The storey_midpoint extraction
# from "01 TO 03" → 2.0 is a classic example of feature engineering that
# should be captured in the schema, not rediscovered per model.


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Register schema and store features in FeatureStore
# ════════════════════════════════════════════════════════════════════════
# kailash-ml's FeatureStore persists feature tables through DataFlow.
# materialize() writes the schema's columns (keyed by entity id + event
# time, idempotent upsert); get_features() reads them back. Writing ~47K
# rows to a local SQLite file takes a minute or two.

print("\n--- FeatureStore Materialisation ---")

fs = create_feature_store()
materialized = asyncio.run(
    materialize_features(fs, property_schema_v1, features_v1_valid)
)
stored_v1 = asyncio.run(fs.get_features(property_schema_v1))

print(f"  Materialised {materialized['row_count']:,} rows "
      f"(schema {materialized['group']} v{materialized['version']})")
print(f"  Lineage hash: {materialized['lineage_hash'][:23]}...")
print(f"  Read back: {stored_v1.shape[0]:,} rows x {stored_v1.shape[1]} columns "
      f"{stored_v1.columns}")

# Price by flat type (the ANOVA question from Exercise 6.4)
flat_types = hdb["flat_type"].unique().sort().to_list()
print(f"\n  --- Price by Flat Type ---")
for ft in flat_types:
    subset = features_v1_valid.filter(pl.col("flat_type") == ft)["resale_price"]
    if subset.len() > 10:
        print(
            f"    {ft:<12}: n={subset.len():>7,}, "
            f"mean=${subset.mean():>10,.0f}, "
            f"median=${subset.median():>10,.0f}"
        )


# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert property_schema_v1.version == 1, "Task 3: schema must be version 1"
assert len(property_schema_v1.fields) == 4, "Task 3: v1 must have 4 features"
assert materialized["row_count"] == features_v1_valid.height, "Task 3: every valid row must be written"
assert stored_v1.height == features_v1_valid.height, "Task 3: read-back must return every row"
print("\n[ok] Checkpoint 2 passed — features materialised and read back\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Feature distributions and structure
# ════════════════════════════════════════════════════════════════════════
# Visual proof: the validated features should show plausible
# distributions for Singapore HDB flats — storey_midpoint in discrete
# steps, remaining_lease at most 99 years. Read the printed quartiles
# rather than assuming a range.

print("\n--- Feature Distribution Summary (validated rows) ---")
for feat_name in ["price_per_sqm", "storey_midpoint", "remaining_lease_years"]:
    vals = features_v1_valid[feat_name].drop_nulls()
    q25 = vals.quantile(0.25)
    q75 = vals.quantile(0.75)
    print(
        f"  {feat_name:<25} " f"Q1={q25:>8,.1f}  Q3={q75:>8,.1f}  IQR={q75-q25:>8,.1f}"
    )

# Plot: feature distributions
fig = go.Figure()
for feat_name in ["price_per_sqm", "remaining_lease_years"]:
    vals = features_v1_valid[feat_name].drop_nulls().to_numpy()
    fig.add_trace(
        go.Histogram(
            x=vals,
            name=feat_name,
            opacity=0.6,
            nbinsx=50,
        )
    )
fig.update_layout(
    title="v1 Feature Distributions — HDB Property Features",
    xaxis_title="Value",
    yaxis_title="Count",
    barmode="overlay",
)
fig.write_html(str(OUTPUT_DIR / "01_feature_distributions.html"))
print(f"\n  Saved: {OUTPUT_DIR / '01_feature_distributions.html'}")

# Plot: correlation heatmap
fig2 = go.Figure(
    data=go.Heatmap(
        z=corr_matrix,
        x=corr_cols,
        y=corr_cols,
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(corr_matrix, 3),
        texttemplate="%{text}",
    )
)
fig2.update_layout(title="Feature Correlation Matrix — HDB v1")
fig2.write_html(str(OUTPUT_DIR / "01_correlation_heatmap.html"))
print(f"  Saved: {OUTPUT_DIR / '01_correlation_heatmap.html'}")


# ── Checkpoint 3 ─────────────────────────────────────────────────────
print("\n[ok] Checkpoint 3 passed — feature distributions visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: HDB Flat Valuation with Schema-Validated Features
# ════════════════════════════════════════════════════════════════════════
# Scenario (illustrative figures): a Singapore property portal builds an
# automated valuation model (AVM) for HDB resale flats. Before any model
# training, it needs a reliable feature pipeline with typed schemas.
#
# Without a schema: an upstream change renames "remaining_lease_years"
# to "lease_remaining". A model that fills missing inputs with a default
# silently produces wrong valuations until someone notices.
#
# With a schema: the write to the store projects onto the schema's
# columns, so the renamed column fails loudly before anything is stored.
# Below we trigger exactly that failure.
#
# S$ impact (assumed figures): 15,000 listings/month, 3 weeks of silent
# failure, 5% of affected deals lost, S$8,000 commission per deal.

print("=== APPLY: HDB Flat Valuation — Schema-Driven Pipeline ===")
print()
print("  Scenario: a Singapore property portal's automated valuation model")

# Simulate the upstream rename and try to write it through the schema
renamed = features_v1_valid.rename({"remaining_lease_years": "lease_remaining"})
try:
    to_store_frame(renamed, property_schema_v1)
    rename_caught = False
except pl.exceptions.ColumnNotFoundError as err:
    rename_caught = True
    print(f"\n  Upstream rename rejected at the schema boundary:")
    print(f"    {type(err).__name__}: {str(err).splitlines()[0]}")

listings_per_month = 15_000  # assumed
weeks_silent = 3  # assumed
deal_loss_rate = 0.05  # assumed
commission = 8_000  # S$, assumed
affected = listings_per_month * weeks_silent / 4
print()
print("  WITHOUT a schema (illustrative):")
print(f"    - {affected:,.0f} listings valued with a default lease for {weeks_silent} weeks")
print(f"    - Lost commission: S${affected * deal_loss_rate * commission:,.0f}")
print()
print("  WITH FeatureSchema v1:")
print("    - The write fails loudly at the schema boundary (shown above)")
print("    - No wrong valuation is produced from the renamed feed")
print()
print("  Your v1 schema enforces:")
for f in property_schema_v1.fields:
    nullable_str = "optional" if f.nullable else "required"
    print(f"    {f.name}: {f.dtype} ({nullable_str})")
print()
print(
    f"  Total features validated: {features_v1_valid.shape[0]:,} rows "
    f"x {len(property_schema_v1.fields)} columns"
)

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert rename_caught, "Task 5: a renamed feature column must be rejected"
print("\n[ok] Checkpoint 4 passed — schema-driven valuation pipeline demonstrated\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [ok] FeatureSchema: typed fields with dtype, nullable, description
  [ok] FeatureField: individual feature contracts within a schema
  [ok] Feature computation: storey_midpoint, price_per_sqm, remaining_lease
  [ok] Value validation: impossible leases and sentinel prices rejected
  [ok] FeatureStore: materialise features and read them back
  [ok] Correlation analysis: identifying multicollinearity early

  KEY INSIGHT: A schema is a unit test for your feature pipeline.
  Types catch phantom columns and dtype drift; value rules catch
  impossible values. Both fire at INGESTION time — not at prediction
  time when it's too late.

  Next: In 02_point_in_time.py, you'll learn how point-in-time
  retrieval prevents data leakage — the #1 cause of models that
  look great in development and fail catastrophically in production.
"""
)
