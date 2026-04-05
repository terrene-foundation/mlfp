# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — ML / FeatureStore
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Register feature schemas, compute and store features, then
#            retrieve them with point-in-time correctness via
#            get_features() and get_training_set().
# LEVEL: Intermediate
# PARITY: Python-only (kailash-ml has no Rust counterpart)
# VALIDATES: FeatureStore, FeatureSchema, FeatureField — register_features(),
#            compute(), store(), get_features(), get_training_set(),
#            get_features_lazy(), list_schemas()
#
# Run: uv run python textbook/python/05-ml/05_feature_store.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import polars as pl

from kailash.db.connection import ConnectionManager
from kailash_ml import FeatureField, FeatureSchema
from kailash_ml.engines.feature_store import FeatureStore


async def main() -> None:
    # ── 1. Set up ConnectionManager (in-memory SQLite) ──────────────────
    # FeatureStore uses ConnectionManager directly (not Express).
    # Point-in-time queries need window functions that Express cannot
    # express.

    conn = ConnectionManager("sqlite:///:memory:")
    await conn.initialize()

    # ── 2. Create FeatureStore with default table prefix ────────────────

    fs = FeatureStore(conn, table_prefix="kml_feat_")
    assert isinstance(fs, FeatureStore)

    # initialize() creates internal metadata tables (idempotent)
    await fs.initialize()

    # ── 3. Define a FeatureSchema ───────────────────────────────────────
    # Schema declares the entity_id column, optional timestamp column,
    # and feature columns with their dtypes.

    schema = FeatureSchema(
        name="user_churn",
        features=[
            FeatureField(name="age", dtype="float64"),
            FeatureField(name="tenure_months", dtype="float64"),
            FeatureField(name="monthly_charges", dtype="float64"),
        ],
        entity_id_column="user_id",
    )

    # ── 4. Register the schema ──────────────────────────────────────────
    # register_features() creates the backing table.  Idempotent --
    # re-registering with the same schema is a no-op.

    await fs.register_features(schema)

    # Re-registering the same schema is safe (no error)
    await fs.register_features(schema)

    # ── 5. Verify schema listing ────────────────────────────────────────

    schemas = await fs.list_schemas()
    assert isinstance(schemas, list)
    assert len(schemas) >= 1
    names = [s["schema_name"] for s in schemas]
    assert "user_churn" in names

    # ── 6. Create synthetic feature data ────────────────────────────────

    raw_data = pl.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(20)],
            "age": [25.0 + i for i in range(20)],
            "tenure_months": [1.0 + i * 2 for i in range(20)],
            "monthly_charges": [50.0 + i * 5.0 for i in range(20)],
        }
    )

    # ── 7. compute() — validate and project to schema ───────────────────
    # compute() checks that all required columns exist, validates
    # nullable constraints, and projects to schema columns only.

    computed = fs.compute(raw_data, schema)
    assert isinstance(computed, pl.DataFrame)
    assert "user_id" in computed.columns
    assert "age" in computed.columns
    assert computed.height == 20

    # ── 8. store() — persist features to the database ───────────────────

    stored_count = await fs.store(computed, schema)
    assert stored_count == 20

    # ── 9. get_features() — retrieve by entity IDs ──────────────────────

    features = await fs.get_features(
        entity_ids=["u0", "u1", "u2"],
        feature_names=["age", "tenure_months"],
        schema=schema,
    )

    assert isinstance(features, pl.DataFrame)
    assert features.height == 3
    assert "user_id" in features.columns
    assert "age" in features.columns
    assert "tenure_months" in features.columns

    # ── 10. get_features() — entity not found returns row ────────────────

    missing = await fs.get_features(
        entity_ids=["nonexistent"],
        feature_names=["age"],
        schema=schema,
    )
    # Non-matching entities return empty DataFrame
    assert missing.height == 0

    # ── 11. get_training_set() — retrieve by time window ─────────────────
    # Returns all feature rows where created_at falls within [start, end].

    now = datetime.now(timezone.utc)
    one_hour_ago = now - timedelta(hours=1)
    one_hour_later = now + timedelta(hours=1)

    training_set = await fs.get_training_set(
        schema,
        start=one_hour_ago,
        end=one_hour_later,
    )
    assert isinstance(training_set, pl.DataFrame)
    assert training_set.height == 20  # All rows within time window

    # Empty window returns empty DataFrame
    far_past = datetime(2020, 1, 1, tzinfo=timezone.utc)
    far_past_end = datetime(2020, 1, 2, tzinfo=timezone.utc)
    empty_training = await fs.get_training_set(schema, start=far_past, end=far_past_end)
    assert empty_training.height == 0

    # ── 12. get_features_lazy() — LazyFrame for streaming ────────────────

    lazy_features = await fs.get_features_lazy(
        entity_ids=["u0", "u5"],
        feature_names=["age", "monthly_charges"],
        schema=schema,
    )
    assert isinstance(lazy_features, pl.LazyFrame)

    # Collect to verify contents
    collected = lazy_features.collect()
    assert isinstance(collected, pl.DataFrame)
    assert collected.height == 2

    # ── 13. compute() — validates missing columns ────────────────────────

    incomplete_data = pl.DataFrame(
        {
            "user_id": ["u0"],
            "age": [30.0],
            # Missing: tenure_months, monthly_charges
        }
    )

    try:
        fs.compute(incomplete_data, schema)
        assert False, "Should raise ValueError for missing columns"
    except ValueError as e:
        assert "missing" in str(e).lower()

    # ── 14. compute() — validates nullable constraints ───────────────────

    strict_schema = FeatureSchema(
        name="strict_features",
        features=[
            FeatureField(name="value", dtype="float64", nullable=False),
        ],
        entity_id_column="entity_id",
    )

    null_data = pl.DataFrame(
        {
            "entity_id": ["a", "b"],
            "value": [1.0, None],
        }
    )

    try:
        fs.compute(null_data, strict_schema)
        assert False, "Should raise ValueError for nulls in non-nullable column"
    except ValueError as e:
        assert "null" in str(e).lower()

    # ── 15. Re-registering different schema raises ───────────────────────

    different_schema = FeatureSchema(
        name="user_churn",  # Same name as before
        features=[
            FeatureField(name="totally_different", dtype="float64"),
        ],
        entity_id_column="user_id",
    )

    try:
        await fs.register_features(different_schema)
        assert False, "Should raise ValueError for schema conflict"
    except ValueError as e:
        assert "already registered" in str(e).lower()

    # ── 16. Invalid table prefix ─────────────────────────────────────────

    try:
        FeatureStore(conn, table_prefix="1invalid!")
        assert False, "Should raise ValueError for invalid prefix"
    except ValueError:
        pass  # Expected: invalid table_prefix

    # ── 17. Clean up ─────────────────────────────────────────────────────

    await conn.close()

    print("PASS: 05-ml/05_feature_store")


asyncio.run(main())
