# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT2 — Exercise 2: FeatureStore Lifecycle & Data Lineage
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Use FeatureStore to persist, version, and retrieve features
#   with point-in-time correctness. Demonstrate data lineage — "if a
#   regulator asks what data trained this model, you can answer."
#
# TASKS:
#   1. Connect to FeatureStore (shared DB with ExperimentTracker)
#   2. Define FeatureSchema with typed fields and versioning
#   3. Compute and store features
#   4. Retrieve features at different points in time (leakage prevention)
#   5. Version the schema and store updated features
#   6. Demonstrate data lineage: query which features trained which model
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import polars as pl
from kailash.db.connection import ConnectionManager
from kailash_ml import FeatureStore, DataExplorer
from kailash_ml.engines.experiment_tracker import ExperimentTracker
from kailash_ml.types import FeatureSchema, FeatureField

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
hdb = loader.load("ascent01", "hdb_resale.parquet")

# Focus on recent data for feature engineering
hdb = hdb.with_columns(pl.col("month").str.to_date("%Y-%m").alias("transaction_date"))


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Connect to FeatureStore (shared DB)
# ══════════════════════════════════════════════════════════════════════


async def setup():
    conn = ConnectionManager("sqlite:///ascent02_experiments.db")
    await conn.initialize()

    # TODO: initialise FeatureStore with the connection and a table prefix
    fs = FeatureStore(
        ____, table_prefix=____
    )  # Hint: pass conn, then table_prefix="kml_feat_"
    await fs.initialize()

    tracker = ExperimentTracker(conn)
    await tracker.initialize()

    return conn, fs, tracker


conn, fs, tracker = asyncio.run(setup())


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Define FeatureSchema with typed fields
# ══════════════════════════════════════════════════════════════════════

# Version 1: Basic property features
# TODO: build the v1 FeatureSchema with four fields and version=1
property_schema_v1 = FeatureSchema(
    name="hdb_property_features",
    features=[
        # TODO: define floor_area_sqm field
        FeatureField(
            name=____,  # Hint: "floor_area_sqm"
            dtype="float64",
            nullable=False,
            description="Floor area in square metres",
        ),
        # TODO: define remaining_lease_years field
        FeatureField(
            name=____,  # Hint: "remaining_lease_years"
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


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Compute and store features
# ══════════════════════════════════════════════════════════════════════


# Compute features from raw data
def compute_v1_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute version 1 property features."""
    return df.with_columns(
        # TODO: parse storey range to midpoint — extract lower and upper bounds, average them
        (
            (
                ____  # Hint: pl.col("storey_range").str.extract(r"(\d+)", 1).cast(pl.Float64)
                + ____  # Hint: pl.col("storey_range").str.extract(r"TO (\d+)", 1).cast(pl.Float64)
            )
            / 2
        ).alias("storey_midpoint"),
        # TODO: compute price per square metre
        (____).alias(
            "price_per_sqm"
        ),  # Hint: pl.col("resale_price") / pl.col("floor_area_sqm")
        # TODO: compute remaining lease years from lease_commence_date
        (
            ____.cast(  # Hint: (99 - (pl.col("transaction_date").dt.year() - pl.col("lease_commence_date")))
                pl.Float64
            ).alias(
                "remaining_lease_years"
            )
        ),
    ).with_row_index("transaction_id")


features_v1 = compute_v1_features(hdb)
print(f"\nComputed v1 features: {features_v1.shape}")


async def store_features():
    """Register schema and store features."""
    # TODO: register the schema with FeatureStore
    await fs.register_features(____)  # Hint: pass property_schema_v1
    print(f"Registered schema: {property_schema_v1.name} v{property_schema_v1.version}")

    # TODO: store the computed features — FeatureStore handles point-in-time indexing
    row_count = await fs.store(
        ____, ____
    )  # Hint: pass features_v1, then property_schema_v1
    print(f"Stored {row_count:,} feature rows")

    return row_count


row_count = asyncio.run(store_features())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Retrieve features at different points in time
# ══════════════════════════════════════════════════════════════════════
# This is the KEY concept: point-in-time retrieval prevents leakage.
# If training a model to predict prices at time T, you must only use
# features computed from data BEFORE time T.


async def demonstrate_pit_retrieval():
    """Show point-in-time feature retrieval."""

    # TODO: retrieve features as of Jan 2023 using get_training_set()
    cutoff_date = datetime(2023, 1, 1)
    features_jan_2023 = await fs.get_training_set(
        schema=____,  # Hint: property_schema_v1
        start=datetime(2000, 1, 1),
        end=____,  # Hint: cutoff_date
    )
    print(f"\n=== Point-in-Time Retrieval ===")
    print(f"Features as of 2023-01-01: {features_jan_2023.height:,} rows")

    # TODO: retrieve features as of Jan 2024 to compare
    features_jan_2024 = await fs.get_training_set(
        schema=property_schema_v1,
        start=datetime(2000, 1, 1),
        end=____,  # Hint: datetime(2024, 1, 1)
    )
    print(f"Features as of 2024-01-01: {features_jan_2024.height:,} rows")

    # The 2024 retrieval has MORE rows (additional year of transactions)
    delta = features_jan_2024.height - features_jan_2023.height
    print(f"Additional transactions in 2023: {delta:,}")

    # LEAKAGE DEMO: if you used features_jan_2024 to train a model
    # predicting Jan 2023 prices, you'd be using future data!
    print("\n--- Leakage Prevention ---")
    print("To predict prices at T=2023-01-01:")
    print("  ✓ Use features retrieved as_of=2023-01-01")
    print("  ✗ Using features as_of=2024-01-01 would include future transactions")

    return features_jan_2023, features_jan_2024


features_2023, features_2024 = asyncio.run(demonstrate_pit_retrieval())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Version the schema — add market context features
# ══════════════════════════════════════════════════════════════════════

# Version 2 adds neighbourhood market features
# TODO: define v2 schema — extend v1 features with three new market-context fields
property_schema_v2 = FeatureSchema(
    name="hdb_property_features",
    features=[
        *property_schema_v1.features,  # inherit all v1 fields
        # TODO: add town_median_price field (trailing 6-month median, nullable)
        FeatureField(
            name=____,  # Hint: "town_median_price"
            dtype="float64",
            nullable=____,  # Hint: True — not all towns have trailing data
            description="Median price in the same town (trailing 6 months)",
        ),
        FeatureField(
            name="town_transaction_volume",
            dtype="int64",
            nullable=True,
            description="Number of transactions in town (trailing 6 months)",
        ),
        # TODO: add town_price_trend field (6-month % change, nullable)
        FeatureField(
            name=____,  # Hint: "town_price_trend"
            dtype="float64",
            nullable=True,
            description="6-month price change % in town",
        ),
    ],
    entity_id_column="transaction_id",
    timestamp_column="transaction_date",
    version=____,  # Hint: 2
)


def compute_v2_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute v2 features including market context."""
    # Start with v1 features
    result = compute_v1_features(df)

    # Compute trailing 6-month town-level statistics
    # Group by town and 6-month windows
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


async def store_v2():
    # TODO: register the v2 schema and store the v2 features
    await fs.register_features(____)  # Hint: property_schema_v2
    print(
        f"\nRegistered schema: {property_schema_v2.name} v{property_schema_v2.version}"
    )

    row_count = await fs.store(____, ____)  # Hint: features_v2, then property_schema_v2
    print(f"Stored {row_count:,} v2 feature rows")

    # List all schema versions
    versions = await fs.list_versions(____)  # Hint: "hdb_property_features"
    print(f"Available versions: {versions}")


asyncio.run(store_v2())


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Data lineage — what data trained this model?
# ══════════════════════════════════════════════════════════════════════


async def demonstrate_lineage():
    """Show data lineage tracking: model → features → source data."""

    # Log a simulated model training run that uses our features
    experiment_id = await tracker.create_experiment(
        name="ascent02_feature_store_lifecycle",
        description="FeatureStore lifecycle demonstration",
        tags=["ascent02", "feature-store", "lineage"],
    )

    # TODO: log a model training run using the ExperimentTracker context manager
    # Record feature_schema, feature_version, as_of_date, train_rows, model_type as params
    # Record rmse, r2, mae as metrics
    async with tracker.run(
        ____, run_name=____
    ) as run:  # Hint: experiment_id, "hdb_price_model_v1"
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


exp_id, run_id = asyncio.run(demonstrate_lineage())

# Clean up
asyncio.run(conn.close())

print("\n✓ Exercise 2 complete — FeatureStore lifecycle with data lineage")
print("  Key concepts: point-in-time retrieval, schema versioning, audit trail")
