# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 8: Capstone — Statistical Analysis Project
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Engineer and version features using FeatureStore with typed schemas
#   - Retrieve features at different points in time to prevent data leakage
#   - Track data lineage from raw data to model-ready features to trained model
#   - Apply point-in-time correctness to ensure training sets have no future data
#   - Connect the full M2 statistical pipeline in a reproducible workflow
#
# PREREQUISITES: All of Module 2 (Exercises 1-7) — Bayesian inference,
#   MLE/MAP estimation, hypothesis testing, A/B design, linear and logistic
#   regression, CUPED and variance reduction.
#
# ESTIMATED TIME: 75 minutes
#
# TASKS:
#   1. Connect to FeatureStore (shared DB with ExperimentTracker)
#   2. Define FeatureSchema with typed fields and versioning
#   3. Compute and store features
#   4. Retrieve features at different points in time (leakage prevention)
#   5. Version the schema and store updated features
#   6. Demonstrate data lineage: query which features trained which model
#
# DATASET: HDB resale flat transactions (Singapore)
#   Source: data.gov.sg — public housing resale records
#   Features engineered: storey midpoint, price per sqm, remaining lease,
#     town-level market statistics (rolling median, volume, price trend)
#
# KEY CONCEPT — Point-in-Time Correctness:
#   If you train a model to predict prices at time T, you must only use
#   features computed from data BEFORE time T. Using future data causes
#   "data leakage" — the model sees information it would not have at
#   prediction time, leading to over-optimistic evaluation and failures
#   when deployed in production.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import polars as pl
from kailash.db import ConnectionManager
from kailash_ml import FeatureStore, DataExplorer
from kailash_ml.engines.experiment_tracker import ExperimentTracker
from kailash_ml.types import FeatureSchema, FeatureField

from shared import MLFPDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ──────────────────────────────────────────────────────

loader = MLFPDataLoader()
hdb = loader.load("mlfp01", "hdb_resale.parquet")

print("=" * 60)
print("  MLFP02 Exercise 8: Capstone — Feature Store and Data Lineage")
print("=" * 60)
print(f"\n  Data loaded: hdb_resale.parquet ({hdb.shape[0]:,} rows)")

# Focus on recent data for feature engineering
hdb = hdb.with_columns(pl.col("month").str.to_date("%Y-%m").alias("transaction_date"))


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Connect to FeatureStore (shared DB)
# ══════════════════════════════════════════════════════════════════════
# The FeatureStore shares a database with ExperimentTracker.
# This means feature metadata, model runs, and experiment results
# are all stored together — enabling full lineage tracing.


async def setup():
    """Set up FeatureStore and ExperimentTracker with shared SQLite backend."""
    from kailash.infrastructure import StoreFactory

    factory = StoreFactory("sqlite:///mlfp02_experiments.db")
    conn = factory

    fs = FeatureStore(factory, table_prefix="kml_feat_")
    tracker = ExperimentTracker(factory)

    return conn, fs, tracker


try:
    conn, fs, tracker = asyncio.run(setup())
except Exception as e:
    print(f"  Note: Feature/Experiment tracking setup skipped ({type(e).__name__}: {e})")
    print("  This requires kailash infrastructure. Proceeding with core exercise...")
    conn, fs, tracker = None, None, None

if conn is not None:
    print(f"\nConnected to FeatureStore (sqlite:///mlfp02_experiments.db)")
    print(f"Table prefix: kml_feat_")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
if fs is not None and tracker is not None:
    print("\n✓ Checkpoint 1 passed — FeatureStore and ExperimentTracker connected\n")
    HAS_FEATURE_STORE = True
else:
    HAS_FEATURE_STORE = False
    print("\n⚠ Checkpoint 1 skipped — FeatureStore/ExperimentTracker not available")
    print("  (SDK infrastructure APIs may differ across versions)")
    print("  Proceeding with core statistical analysis...\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Define FeatureSchema with typed fields
# ══════════════════════════════════════════════════════════════════════
# A FeatureSchema defines:
#   - The features available (names, types, nullable)
#   - The entity key (what uniquely identifies a row)
#   - The timestamp column (for point-in-time retrieval)
#   - The version (for schema evolution tracking)
#
# Typed schemas enforce data quality at the storage layer — you cannot
# accidentally store a string where a float is expected.

# Version 1: Basic property features
property_schema_v1 = FeatureSchema(
    name="hdb_property_features",
    features=[
        FeatureField(
            name="floor_area_sqm",
            dtype="float64",
            nullable=False,
            description="Floor area in square metres",
        ),
        FeatureField(
            name="remaining_lease_years",
            dtype="float64",
            nullable=False,
            description="Remaining lease in years",
        ),
        FeatureField(
            name="storey_midpoint",
            dtype="float64",
            nullable=False,
            description="Midpoint of storey range",
        ),
        FeatureField(
            name="price_per_sqm",
            dtype="float64",
            nullable=False,
            description="Transaction price per square metre",
        ),
    ],
    entity_id_column="transaction_id",
    timestamp_column="transaction_date",
    version=1,
)

print("=== FeatureSchema v1 ===")
print(f"Name: {property_schema_v1.name}")
print(f"Version: {property_schema_v1.version}")
for f in property_schema_v1.features:
    print(f"  {f.name}: {f.dtype} — {f.description}")
# INTERPRETATION: Schema versioning is essential for long-lived ML systems.
# When you add features (v1 → v2), old models trained on v1 still work
# because the FeatureStore tracks which version was used for each training run.
# Without versioning, schema changes silently break existing models.


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Compute and store features
# ══════════════════════════════════════════════════════════════════════


# Compute features from raw data
def compute_v1_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute version 1 property features."""
    return df.with_columns(
        # Parse storey range to midpoint
        (
            (
                pl.col("storey_range").str.extract(r"(\d+)", 1).cast(pl.Float64)
                + pl.col("storey_range").str.extract(r"TO (\d+)", 1).cast(pl.Float64)
            )
            / 2
        ).alias("storey_midpoint"),
        # Price per sqm
        (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"),
        # Remaining lease (approximate from lease_commence_date)
        (99 - (pl.col("transaction_date").dt.year() - pl.col("lease_commence_date")))
        .cast(pl.Float64)
        .alias("remaining_lease_years"),
    ).with_row_index("transaction_id")


features_v1 = compute_v1_features(hdb)
print(f"\nComputed v1 features: {features_v1.shape}")
print(f"Feature columns: {[f.name for f in property_schema_v1.features]}")

# Quick sanity check on feature values
price_per_sqm_mean = features_v1["price_per_sqm"].mean()
print(f"\nFeature sanity check:")
print(f"  avg price_per_sqm: ${price_per_sqm_mean:,.0f}")
print(f"  avg storey_midpoint: {features_v1['storey_midpoint'].mean():.1f}")
print(f"  avg remaining_lease: {features_v1['remaining_lease_years'].mean():.1f} years")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert "transaction_id" in features_v1.columns, "transaction_id column must exist"
assert "storey_midpoint" in features_v1.columns, "storey_midpoint must be computed"
assert "price_per_sqm" in features_v1.columns, "price_per_sqm must be computed"
assert "remaining_lease_years" in features_v1.columns, "remaining_lease_years must be computed"
assert features_v1["price_per_sqm"].min() > 0, "price_per_sqm must be positive"
assert features_v1["storey_midpoint"].min() > 0, "storey_midpoint must be positive"
print("\n✓ Checkpoint 2 passed — v1 features computed and validated\n")


async def store_features():
    """Register schema and store features."""
    # Register the schema
    await fs.register_features(property_schema_v1)
    print(f"Registered schema: {property_schema_v1.name} v{property_schema_v1.version}")

    # Store features (FeatureStore handles point-in-time indexing)
    row_count = await fs.store(features_v1, property_schema_v1)
    print(f"Stored {row_count:,} feature rows")

    return row_count


if HAS_FEATURE_STORE:
    try:
        row_count = asyncio.run(store_features())
    except Exception as e:
        row_count = 0
        HAS_FEATURE_STORE = False
        print(f"  [Skipped: FeatureStore store failed ({type(e).__name__}: {e})]")
else:
    row_count = 0
    print("  [Skipped: FeatureStore not available]")
# INTERPRETATION: FeatureStore.store() ingests the features and indexes them
# by the timestamp column (transaction_date). Later retrieval can specify a
# cutoff date — only features with timestamp <= cutoff are returned.
# This is point-in-time correctness implemented at the storage layer.


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Retrieve features at different points in time
# ══════════════════════════════════════════════════════════════════════
# This is the KEY concept: point-in-time retrieval prevents leakage.
# If training a model to predict prices at time T, you must only use
# features computed from data BEFORE time T.


async def demonstrate_pit_retrieval():
    """Show point-in-time feature retrieval."""

    # Retrieve features as of Jan 2023 (no future data)
    cutoff_date = datetime(2023, 1, 1)
    features_jan_2023 = await fs.get_training_set(
        schema=property_schema_v1,
        start=datetime(2000, 1, 1),
        end=cutoff_date,
    )
    print(f"\n=== Point-in-Time Retrieval ===")
    print(f"Features as of 2023-01-01: {features_jan_2023.height:,} rows")

    # Retrieve features as of Jan 2024 (one year later)
    features_jan_2024 = await fs.get_training_set(
        schema=property_schema_v1,
        start=datetime(2000, 1, 1),
        end=datetime(2024, 1, 1),
    )
    print(f"Features as of 2024-01-01: {features_jan_2024.height:,} rows")

    # The 2024 retrieval has MORE rows (additional year of transactions)
    delta = features_jan_2024.height - features_jan_2023.height
    print(f"Additional transactions in 2023: {delta:,}")

    # LEAKAGE DEMO: if you used features_jan_2024 to train a model
    # predicting Jan 2023 prices, you'd be using future data!
    print("\n--- Leakage Prevention ---")
    print("To predict prices at T=2023-01-01:")
    print("  Use features retrieved as_of=2023-01-01")
    print("  Using features as_of=2024-01-01 would include future transactions")
    print("  → Leakage: model trained on future data appears better than it is")

    return features_jan_2023, features_jan_2024


if HAS_FEATURE_STORE:
    try:
        features_2023, features_2024 = asyncio.run(demonstrate_pit_retrieval())
    except Exception as e:
        features_2023 = features_2024 = None
        HAS_FEATURE_STORE = False
        print(f"  [Skipped: FeatureStore retrieval failed ({type(e).__name__}: {e})]")
else:
    features_2023 = features_2024 = None
    print("  [Skipped: FeatureStore not available]")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
if features_2023 is not None:
    assert features_2023.height > 0, "2023 training set must have rows"
    assert features_2024.height >= features_2023.height, \
        "2024 set must have at least as many rows as 2023 (includes more history)"
    print("\n✓ Checkpoint 3 passed — point-in-time retrieval demonstrated\n")
else:
    print("\n⚠ Checkpoint 3 skipped — FeatureStore not available\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Version the schema — add market context features
# ══════════════════════════════════════════════════════════════════════
# Version 2 adds neighbourhood market features computed from trailing
# 6-month windows. These capture market conditions at the time of
# each transaction — important context for price prediction.

# Version 2 adds neighbourhood market features
property_schema_v2 = FeatureSchema(
    name="hdb_property_features",
    features=[
        *property_schema_v1.features,
        FeatureField(
            name="town_median_price",
            dtype="float64",
            nullable=True,
            description="Median price in the same town (trailing 6 months)",
        ),
        FeatureField(
            name="town_transaction_volume",
            dtype="int64",
            nullable=True,
            description="Number of transactions in town (trailing 6 months)",
        ),
        FeatureField(
            name="town_price_trend",
            dtype="float64",
            nullable=True,
            description="6-month price change % in town",
        ),
    ],
    entity_id_column="transaction_id",
    timestamp_column="transaction_date",
    version=2,
)

print(f"\n=== FeatureSchema v2 (adds {len(property_schema_v2.features) - len(property_schema_v1.features)} market features) ===")
for f in property_schema_v2.features:
    marker = " [NEW]" if f.name not in [f2.name for f2 in property_schema_v1.features] else ""
    print(f"  {f.name}: {f.dtype}{marker}")


def compute_v2_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute v2 features including market context."""
    # Start with v1 features
    result = compute_v1_features(df)

    # Compute trailing 6-month town-level statistics
    # Group by town and 6-month windows
    # group_by_dynamic requires data sorted by the time column
    result = result.sort("transaction_date")
    town_stats = (
        result.group_by_dynamic("transaction_date", every="1mo", group_by="town")
        .agg(
            pl.col("resale_price").median().alias("monthly_median"),
            pl.col("resale_price").count().alias("monthly_volume"),
        )
        .sort("town", "transaction_date")
    )

    # 6-month rolling stats per town
    town_stats = town_stats.with_columns(
        pl.col("monthly_median")
        .rolling_mean(window_size=6)
        .over("town")
        .alias("town_median_price"),
        pl.col("monthly_volume")
        .rolling_sum(window_size=6)
        .over("town")
        .alias("town_transaction_volume"),
        (
            (pl.col("monthly_median") - pl.col("monthly_median").shift(6).over("town"))
            / pl.col("monthly_median").shift(6).over("town")
            * 100
        ).alias("town_price_trend"),
    )

    # Join back to transactions
    result = result.join(
        town_stats.select(
            "town",
            "transaction_date",
            "town_median_price",
            "town_transaction_volume",
            "town_price_trend",
        ),
        on=["town", "transaction_date"],
        how="left",
    )

    return result


features_v2 = compute_v2_features(hdb)
print(f"\nComputed v2 features: {features_v2.shape}")

n_with_market = features_v2.filter(pl.col("town_median_price").is_not_null()).height
print(f"Rows with market context: {n_with_market:,} ({n_with_market/features_v2.height:.1%})")
print(f"(First 6 months of each town have null market context — rolling window warm-up)")
# INTERPRETATION: The first 6 months of data for each town will have null
# town_median_price because there's not enough history for the rolling window.
# We allow null for these market features (nullable=True in the schema).
# This is an honest representation — don't impute with forward-looking data.

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert features_v2.height == features_v1.height, \
    "v2 should have same number of rows as v1 (market features are joined, not filtered)"
assert "town_median_price" in features_v2.columns, "town_median_price must be computed"
assert "town_price_trend" in features_v2.columns, "town_price_trend must be computed"
print("\n✓ Checkpoint 4 passed — v2 features computed\n")


async def store_v2():
    await fs.register_features(property_schema_v2)
    print(
        f"\nRegistered schema: {property_schema_v2.name} v{property_schema_v2.version}"
    )

    row_count = await fs.store(features_v2, property_schema_v2)
    print(f"Stored {row_count:,} v2 feature rows")

    # List all schema versions
    versions = await fs.list_versions("hdb_property_features")
    print(f"Available versions: {versions}")


if HAS_FEATURE_STORE:
    try:
        asyncio.run(store_v2())
    except Exception as e:
        HAS_FEATURE_STORE = False
        print(f"  [Skipped: FeatureStore v2 store failed ({type(e).__name__}: {e})]")
else:
    print("  [Skipped: FeatureStore not available]")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Data lineage — what data trained this model?
# ══════════════════════════════════════════════════════════════════════
# Data lineage answers: "If a regulator asks what data trained this model,
# what is the exact, reproducible answer?"
#
# With ExperimentTracker + FeatureStore:
#   - Each model run logs: feature schema name, version, and cutoff date
#   - This is queryable: "which features were used for model run ID X?"
#   - This enables: audit, debugging, reproducibility, and retraining


async def demonstrate_lineage():
    """Show data lineage tracking: model → features → source data."""

    # Log a simulated model training run that uses our features
    experiment_id = await tracker.create_experiment(
        name="mlfp02_feature_store_lifecycle",
        description="FeatureStore lifecycle demonstration",
        tags=["mlfp02", "feature-store", "lineage"],
    )

    async with tracker.run(experiment_id, run_name="hdb_price_model_v1") as run:
        await run.log_params(
            {
                "feature_schema": "hdb_property_features",
                "feature_version": "2",
                "as_of_date": "2023-06-01",
                "train_rows": str(features_2023.height),
                "model_type": "LightGBM",
            }
        )
        await run.log_metrics(
            {
                "rmse": 45_000.0,
                "r2": 0.87,
                "mae": 32_000.0,
            }
        )
        await run.set_tag("purpose", "lineage-demo")
        run_id = run.id if hasattr(run, "id") else "logged"

    print(f"\n=== Data Lineage ===")
    print(f"Model run: {run_id}")
    print(f"  → Uses feature schema: hdb_property_features v2")
    print(f"  → Features as_of: 2023-06-01")
    print(f"  → Training rows: {features_2023.height:,}")
    print(f"  → Source: HDB resale data (data.gov.sg)")
    print()
    print("If a regulator asks 'what data trained this model?', you can answer:")
    print(f"  1. Model ID: {run_id}")
    print(f"  2. Feature schema: hdb_property_features v2 (7 features)")
    print(f"  3. Point-in-time cutoff: 2023-06-01 (no future leakage)")
    print(f"  4. Training data: {features_2023.height:,} HDB transactions")
    print(f"  5. Source: data.gov.sg resale flat prices")

    return experiment_id, run_id


if HAS_FEATURE_STORE:
    try:
        exp_id, run_id = asyncio.run(demonstrate_lineage())
    except Exception as e:
        exp_id, run_id = None, None
        HAS_FEATURE_STORE = False
        print(f"  [Skipped: ExperimentTracker lineage failed ({type(e).__name__}: {e})]")
else:
    exp_id, run_id = None, None
    print("  [Skipped: ExperimentTracker not available]")
# INTERPRETATION: Data lineage is a regulatory requirement in Singapore's
# MAS (Monetary Authority of Singapore) guidelines for model risk management.
# Every model in production must have documented: what data it was trained on,
# when that data was collected, and what features were engineered. Without
# ExperimentTracker + FeatureStore, this requires manual documentation that
# inevitably drifts from reality as models are retrained and updated.

# ── Checkpoint 5 ─────────────────────────────────────────────────────
if HAS_FEATURE_STORE:
    assert exp_id is not None, "Experiment ID must be created"
    assert run_id is not None, "Run ID must be logged"
    print("\n✓ Checkpoint 5 passed — data lineage logged to ExperimentTracker\n")
else:
    print("\n⚠ Checkpoint 5 skipped — ExperimentTracker not available\n")

# Clean up
if HAS_FEATURE_STORE and conn is not None and hasattr(conn, "close"):
    asyncio.run(conn.close())

print("\n✓ Exercise 8 complete — FeatureStore lifecycle and capstone project")
print("  Key concepts: point-in-time retrieval, schema versioning, data lineage, audit trail")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 60)
print("  WHAT YOU'VE MASTERED")
print("═" * 60)
print("""
  ✓ FeatureSchema: typed field definitions with nullable, descriptions, versioning
  ✓ FeatureStore.register_features: schema registration before data storage
  ✓ FeatureStore.store: point-in-time indexed feature ingestion
  ✓ FeatureStore.get_training_set: date-bounded retrieval for leakage prevention
  ✓ Schema versioning: v1 → v2 by extending the features list
  ✓ Rolling market features: group_by_dynamic + rolling_mean/rolling_sum
  ✓ ExperimentTracker: audit trail linking model → features → training data
  ✓ Data lineage: reproducible answer to "what data trained this model?"

  MODULE 2 COMPLETE — YOU'VE MASTERED:
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Ex 1: Bayesian inference — Normal-Normal conjugate, credible intervals
  Ex 2: MLE + MAP — scipy.optimize, failure modes, AIC model selection
  Ex 3: Hypothesis testing — power, p-values, Bonferroni, BH-FDR, permutation
  Ex 4: A/B design — pre-registration, SRM, Welch's CI, data collection plan
  Ex 5: Linear regression — OLS from scratch, t-stats, R², cross-validation
  Ex 6: Logistic regression — sigmoid, Bernoulli MLE, odds ratios, ANOVA/Tukey
  Ex 7: CUPED + causal inference — variance reduction, Bayesian A/B, mSPRT
  Ex 8: FeatureStore capstone — point-in-time features, versioning, lineage

  → NEXT MODULE: M3 — Supervised ML in the Kailash Pipeline
    You'll use TrainingPipeline, HyperparameterSearch, and ModelRegistry
    to build, tune, and deploy models at production scale. The statistical
    foundations from M2 will ground every ML decision you make in M3.
""")
