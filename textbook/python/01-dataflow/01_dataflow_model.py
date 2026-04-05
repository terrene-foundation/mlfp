# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — DataFlow / Model Definition
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Define database models using @db.model decorator and field types
# LEVEL: Basic
# PARITY: Full — Rust uses ModelDefinition::new() with FieldType enum;
#         Python uses @db.model decorator with type annotations
# VALIDATES: DataFlow(), @db.model decorator, field type mapping, initialize()
#
# Run: uv run python textbook/python/01-dataflow/01_dataflow_model.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os
import tempfile

from dataflow import DataFlow

# ── 1. Create a DataFlow instance ───────────────────────────────────
# DataFlow connects to a database. SQLite is simplest for tutorials.
# In production, use PostgreSQL via DATABASE_URL environment variable.

db_path = os.path.join(tempfile.gettempdir(), "textbook_dataflow_01.db")
db = DataFlow(database_url=f"sqlite:///{db_path}")

# ── 2. Define a model with @db.model ───────────────────────────────
# The decorator registers the class, creates CRUD nodes, and maps
# Python type annotations to database column types.
#
# Supported types: str, int, float, bool, datetime
# Fields with defaults are optional; fields without are required.


@db.model
class Product:
    name: str
    price: float
    category: str
    in_stock: bool = True


# Verify the model was registered
registered_models = db.get_models()
assert "Product" in registered_models, "Product model should be registered"

# Inspect model fields
fields = db.get_model_fields("Product")
assert "name" in fields, "name field registered"
assert "price" in fields, "price field registered"
assert "category" in fields, "category field registered"
assert "in_stock" in fields, "in_stock field registered"

# Required vs optional: fields with defaults are not required
assert fields["name"]["required"] is True, "name has no default — required"
assert fields["in_stock"]["required"] is False, "in_stock has default — optional"
assert fields["in_stock"]["default"] is True

print(f"Registered models: {list(registered_models.keys())}")
print(f"Product fields: {list(fields.keys())}")

# ── 3. Multiple models ─────────────────────────────────────────────


@db.model
class Customer:
    email: str
    name: str
    tier: str = "free"


@db.model
class Order:
    customer_email: str
    product_name: str
    quantity: int
    total: float


all_models = db.get_models()
assert len(all_models) >= 3, "Three models registered"
assert "Customer" in all_models
assert "Order" in all_models

# ── 4. Initialize — create tables in the database ──────────────────
# initialize() is async — it creates tables, runs migrations,
# and sets up the connection pool.


async def main():
    success = await db.initialize()
    assert success, "DataFlow initialization should succeed"

    # After initialize, tables exist and CRUD is available
    print(f"DataFlow initialized with {len(all_models)} models")

    # ── 5. Table name mapping ───────────────────────────────────────
    # By default, class names are converted to snake_case table names:
    #   Product → products
    #   Customer → customers
    #   Order → orders
    # Override with __tablename__ class attribute if needed.

    # Cleanup
    await db.stop()


asyncio.run(main())

# ── 6. Edge case: custom table name ────────────────────────────────

db2 = DataFlow(database_url=f"sqlite:///{db_path}")


@db2.model
class AuditLog:
    __tablename__ = "custom_audit_logs"
    action: str
    details: str


# The table name override is respected
model_info = db2._models.get("AuditLog", {})
assert model_info.get("table_name") == "custom_audit_logs"

# Cleanup temp file
try:
    os.unlink(db_path)
except OSError:
    pass

print("PASS: 01-dataflow/01_dataflow_model")
