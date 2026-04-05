# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — DataFlow / List with Filters
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Query records using filters, ordering, limit, and offset
# LEVEL: Basic
# PARITY: Full — Rust uses FilterCondition builder; Python uses dict filters
# VALIDATES: express.list() with filters, order_by, limit, offset
#
# Run: uv run python textbook/python/01-dataflow/03_list_filters.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os
import tempfile

from dataflow import DataFlow

# ── Setup ───────────────────────────────────────────────────────────

db_path = os.path.join(tempfile.gettempdir(), "textbook_dataflow_03.db")
db = DataFlow(database_url=f"sqlite:///{db_path}")


@db.model
class Employee:
    name: str
    department: str
    salary: float
    active: bool = True


async def main():
    await db.initialize()

    # Seed data
    employees = [
        {"name": "Alice", "department": "Engineering", "salary": 95000.0},
        {"name": "Bob", "department": "Engineering", "salary": 85000.0},
        {"name": "Carol", "department": "Marketing", "salary": 78000.0},
        {"name": "Dave", "department": "Marketing", "salary": 72000.0},
        {"name": "Eve", "department": "Engineering", "salary": 110000.0},
        {"name": "Frank", "department": "Sales", "salary": 68000.0, "active": False},
    ]
    for emp in employees:
        await db.express.create("Employee", emp)

    # ── 1. List all ─────────────────────────────────────────────────

    all_emps = await db.express.list("Employee")
    assert len(all_emps) == 6, "Six employees created"

    # ── 2. Filter by exact value ────────────────────────────────────
    # Pass filters as a dict: {field: value}

    engineers = await db.express.list("Employee", filters={"department": "Engineering"})
    assert len(engineers) == 3, "Three engineers"

    active_only = await db.express.list("Employee", filters={"active": True})
    assert len(active_only) == 5, "Five active employees"

    # ── 3. Order by field ───────────────────────────────────────────

    by_salary = await db.express.list("Employee", order_by="salary")
    salaries = [e["salary"] for e in by_salary]
    assert salaries == sorted(salaries), "Ordered by salary ascending"

    # ── 4. Limit and offset (pagination) ────────────────────────────

    page1 = await db.express.list("Employee", order_by="name", limit=2, offset=0)
    assert len(page1) == 2, "Page 1: 2 records"

    page2 = await db.express.list("Employee", order_by="name", limit=2, offset=2)
    assert len(page2) == 2, "Page 2: 2 records"

    # No overlap between pages
    page1_names = {e["name"] for e in page1}
    page2_names = {e["name"] for e in page2}
    assert len(page1_names & page2_names) == 0, "Pages don't overlap"

    # ── 5. Count with filters ───────────────────────────────────────

    eng_count = await db.express.count(
        "Employee", filters={"department": "Engineering"}
    )
    assert eng_count == 3

    total = await db.express.count("Employee")
    assert total == 6

    # ── 6. Combined: filter + order + limit ─────────────────────────

    top_engineers = await db.express.list(
        "Employee",
        filters={"department": "Engineering"},
        order_by="salary",
        limit=2,
    )
    assert len(top_engineers) == 2
    print(f"Top 2 engineers by salary: {[e['name'] for e in top_engineers]}")

    # ── Cleanup ─────────────────────────────────────────────────────
    await db.stop()


asyncio.run(main())

try:
    os.unlink(db_path)
except OSError:
    pass

print("PASS: 01-dataflow/03_list_filters")
