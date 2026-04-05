# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — DataFlow / Express CRUD
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Perform Create, Read, Update, Delete operations using
#            the Express API (db.express)
# LEVEL: Basic
# PARITY: Full — Rust has equivalent express module with same operations
# VALIDATES: ExpressDataFlow.create(), read(), update(), delete()
#
# Run: uv run python textbook/python/01-dataflow/02_express_crud.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os
import tempfile

from dataflow import DataFlow

# ── Setup ───────────────────────────────────────────────────────────

db_path = os.path.join(tempfile.gettempdir(), "textbook_dataflow_02.db")
db = DataFlow(database_url=f"sqlite:///{db_path}")


@db.model
class Task:
    title: str
    description: str
    priority: int
    done: bool = False


async def main():
    await db.initialize()

    # ── 1. CREATE — insert a new record ─────────────────────────────
    # express.create(model_name, data_dict) returns the created record
    # with an auto-generated id.

    task1 = await db.express.create(
        "Task",
        {
            "title": "Learn DataFlow",
            "description": "Complete the textbook tutorial",
            "priority": 1,
        },
    )

    assert "id" in task1, "Created record gets an auto-generated id"
    assert task1["title"] == "Learn DataFlow"
    assert task1["done"] is False, "Default value applied"
    print(f"Created: {task1}")

    # Create more records
    task2 = await db.express.create(
        "Task",
        {
            "title": "Build ML Pipeline",
            "description": "Use TrainingPipeline engine",
            "priority": 2,
        },
    )
    task3 = await db.express.create(
        "Task",
        {
            "title": "Deploy with Nexus",
            "description": "Multi-channel deployment",
            "priority": 3,
        },
    )

    # ── 2. READ — fetch a single record by id ──────────────────────
    # express.read(model_name, id) returns the record or None.

    fetched = await db.express.read("Task", task1["id"])

    assert fetched is not None, "Record found by id"
    assert fetched["title"] == "Learn DataFlow"
    assert fetched["id"] == task1["id"]
    print(f"Read: {fetched}")

    # ── 3. UPDATE — modify an existing record ───────────────────────
    # express.update(model_name, id, fields_dict) returns updated record.

    updated = await db.express.update(
        "Task",
        task1["id"],
        {"done": True, "priority": 0},
    )

    assert updated["done"] is True, "Field updated"
    assert updated["priority"] == 0, "Priority updated"
    print(f"Updated: {updated}")

    # Verify the update persisted
    refetched = await db.express.read("Task", task1["id"])
    assert refetched["done"] is True

    # ── 4. DELETE — remove a record ─────────────────────────────────
    # express.delete(model_name, id) returns True on success.

    deleted = await db.express.delete("Task", task3["id"])
    assert deleted is True, "Delete returns True"

    # Verify deletion
    gone = await db.express.read("Task", task3["id"])
    assert gone is None, "Deleted record returns None on read"

    # ── 5. LIST — fetch multiple records ────────────────────────────
    # express.list(model_name) returns all records.
    # Supports filters, ordering, limit, offset.

    all_tasks = await db.express.list("Task")
    assert len(all_tasks) == 2, "Two tasks remain after delete"

    # List with ordering
    ordered = await db.express.list("Task", order_by="priority")
    assert len(ordered) == 2
    print(f"Listed {len(ordered)} tasks")

    # ── 6. COUNT — count records ────────────────────────────────────

    count = await db.express.count("Task")
    assert count == 2, "Two records in total"

    # ── Cleanup ─────────────────────────────────────────────────────
    await db.stop()


asyncio.run(main())

try:
    os.unlink(db_path)
except OSError:
    pass

print("PASS: 01-dataflow/02_express_crud")
