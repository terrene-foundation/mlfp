# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — DataFlow / SyncExpress CRUD
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Use SyncExpress for synchronous CRUD without async/await
# LEVEL: Intermediate
# PARITY: Full — both languages have sync wrappers
# VALIDATES: SyncExpress via db.express_sync, same CRUD operations as
#            async but synchronous (create, read, update, delete, list, count)
#
# Run: uv run python textbook/python/01-dataflow/04_sync_express.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os
import tempfile

from dataflow import DataFlow

# ── Setup ───────────────────────────────────────────────────────────
# SyncExpress wraps the async ExpressDataFlow methods with synchronous
# equivalents. Useful in CLI scripts, synchronous FastAPI handlers,
# pytest without asyncio, and other non-async code.
#
# Internally, SyncExpress maintains a persistent event loop in a
# background daemon thread so that database connections survive
# across multiple sync calls.

db_path = os.path.join(tempfile.gettempdir(), "textbook_dataflow_04.db")
db = DataFlow(database_url=f"sqlite:///{db_path}")


@db.model
class Book:
    title: str
    author: str
    pages: int
    available: bool = True


# initialize() is still async — call it once before using express_sync
async def setup():
    await db.initialize()


asyncio.run(setup())

# ── 1. ACCESS — db.express_sync property ────────────────────────────
# After initialize(), access SyncExpress through db.express_sync.
# No 'await' needed — all calls are plain synchronous functions.

sync = db.express_sync
assert sync is not None, "express_sync property returns SyncExpress instance"

# ── 2. CREATE — insert records synchronously ────────────────────────

book1 = sync.create(
    "Book",
    {
        "title": "The Pragmatic Programmer",
        "author": "Hunt & Thomas",
        "pages": 352,
    },
)

assert "id" in book1, "Created record gets an auto-generated id"
assert book1["title"] == "The Pragmatic Programmer"
assert book1["available"] is True, "Default value applied"
print(f"Created: {book1['title']}")

book2 = sync.create(
    "Book",
    {
        "title": "Clean Code",
        "author": "Robert C. Martin",
        "pages": 464,
    },
)
book3 = sync.create(
    "Book",
    {
        "title": "Refactoring",
        "author": "Martin Fowler",
        "pages": 448,
        "available": False,
    },
)

# ── 3. READ — fetch a single record by id ──────────────────────────

fetched = sync.read("Book", book1["id"])

assert fetched is not None, "Record found by id"
assert fetched["title"] == "The Pragmatic Programmer"
print(f"Read: {fetched['title']}")

# Read non-existent record returns None
missing = sync.read("Book", "nonexistent-id-999")
assert missing is None, "Missing record returns None"

# ── 4. UPDATE — modify an existing record ───────────────────────────

updated = sync.update("Book", book3["id"], {"available": True})

assert updated["available"] is True, "Field updated"
print(f"Updated: {updated['title']} is now available")

# Verify persistence
refetched = sync.read("Book", book3["id"])
assert refetched["available"] is True, "Update persisted"

# ── 5. DELETE — remove a record ─────────────────────────────────────

deleted = sync.delete("Book", book2["id"])
assert deleted is True, "Delete returns True on success"

gone = sync.read("Book", book2["id"])
assert gone is None, "Deleted record returns None on read"

# ── 6. LIST — query multiple records ────────────────────────────────

all_books = sync.list("Book")
assert len(all_books) == 2, "Two books remain after delete"

# List with ordering
ordered = sync.list("Book", order_by="pages")
assert len(ordered) == 2
pages = [b["pages"] for b in ordered]
assert pages == sorted(pages), "Ordered by pages ascending"

# ── 7. COUNT — count records ────────────────────────────────────────

total = sync.count("Book")
assert total == 2, "Two records in total"

# ── 8. Same instance reuse ──────────────────────────────────────────
# The SyncExpress instance is cached on the DataFlow object.
# Accessing db.express_sync again returns the same instance.

sync2 = db.express_sync
assert sync2 is sync, "Same SyncExpress instance returned"

# ── Cleanup ─────────────────────────────────────────────────────────


async def teardown():
    await db.stop()


asyncio.run(teardown())

try:
    os.unlink(db_path)
except OSError:
    pass

print("PASS: 01-dataflow/04_sync_express")
