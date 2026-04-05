# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Core / ConnectionManager
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Use ConnectionManager for database-backed workflow
#            infrastructure — the foundation for stateful engines
# LEVEL: Advanced
# PARITY: Python-only — Rust handles database infrastructure at the
#         native driver level with different pooling abstractions
# VALIDATES: ConnectionManager initialization, lifecycle, execute(),
#            fetch(), fetchone(), transaction(), close()
#
# Run: uv run python textbook/python/00-core/10_connection_manager.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

from kailash.db import ConnectionManager, DatabaseType, detect_dialect

# ── 1. Dialect detection ───────────────────────────────────────────
# ConnectionManager auto-detects the database type from the URL.
# Supported dialects: SQLite, PostgreSQL, MySQL.

sqlite_dialect = detect_dialect("sqlite:///local.db")
assert sqlite_dialect.database_type == DatabaseType.SQLITE

pg_dialect = detect_dialect("postgresql://user:pass@localhost/mydb")
assert pg_dialect.database_type == DatabaseType.POSTGRESQL

mysql_dialect = detect_dialect("mysql://user:pass@localhost/mydb")
assert mysql_dialect.database_type == DatabaseType.MYSQL

print(f"SQLite: {sqlite_dialect.database_type.value}")
print(f"PostgreSQL: {pg_dialect.database_type.value}")
print(f"MySQL: {mysql_dialect.database_type.value}")


# ── 2. ConnectionManager lifecycle ─────────────────────────────────
# Create with a URL, initialize to open the connection pool, then
# close when done. Always use async — ConnectionManager is async-only.


async def test_lifecycle():
    """Demonstrate the basic lifecycle: create, initialize, close."""
    conn = ConnectionManager("sqlite:///:memory:")

    # Before initialize(), the pool is None
    assert conn._pool is None, "Pool starts as None"
    assert conn.url == "sqlite:///:memory:"
    assert conn.dialect.database_type == DatabaseType.SQLITE

    # Initialize creates the connection pool
    await conn.initialize()
    assert conn._pool is not None, "Pool is created after initialize()"

    # Close releases all resources
    await conn.close()
    assert conn._pool is None, "Pool is None after close()"


asyncio.run(test_lifecycle())
print("Lifecycle: create -> initialize -> close")


# ── 3. Execute queries ─────────────────────────────────────────────
# execute() runs DDL/DML statements with dialect placeholder translation.
# Use canonical ? placeholders — ConnectionManager translates them to
# the target dialect automatically ($1 for PostgreSQL, %s for MySQL).


async def test_execute():
    """Create a table, insert rows, query them."""
    conn = ConnectionManager("sqlite:///:memory:")
    await conn.initialize()

    # DDL: create a table
    await conn.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, score REAL)"
    )

    # DML: insert rows using ? placeholders
    await conn.execute(
        "INSERT INTO users (id, name, score) VALUES (?, ?, ?)", 1, "Alice", 95.5
    )
    await conn.execute(
        "INSERT INTO users (id, name, score) VALUES (?, ?, ?)", 2, "Bob", 87.0
    )
    await conn.execute(
        "INSERT INTO users (id, name, score) VALUES (?, ?, ?)", 3, "Carol", 92.3
    )

    # fetch() returns a list of dicts
    rows = await conn.fetch("SELECT * FROM users ORDER BY id")
    assert len(rows) == 3, "Three rows inserted"
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"
    assert rows[2]["name"] == "Carol"
    print(f"Fetched {len(rows)} users: {[r['name'] for r in rows]}")

    # fetch() with parameters
    high_scorers = await conn.fetch("SELECT * FROM users WHERE score > ?", 90.0)
    assert len(high_scorers) == 2, "Alice and Carol score above 90"
    print(f"High scorers: {[r['name'] for r in high_scorers]}")

    # fetchone() returns a single dict or None
    alice = await conn.fetchone("SELECT * FROM users WHERE name = ?", "Alice")
    assert alice is not None
    assert alice["score"] == 95.5
    print(f"Alice: {alice}")

    nobody = await conn.fetchone("SELECT * FROM users WHERE name = ?", "Nobody")
    assert nobody is None, "fetchone() returns None when no match"
    print("fetchone() with no match returns None")

    await conn.close()


asyncio.run(test_execute())


# ── 4. Transactions ────────────────────────────────────────────────
# transaction() provides an async context manager for multi-statement
# atomic operations. On success, changes are committed. On exception,
# changes are rolled back.


async def test_transactions():
    """Demonstrate commit and rollback behavior."""
    conn = ConnectionManager("sqlite:///:memory:")
    await conn.initialize()

    await conn.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance REAL)")
    await conn.execute("INSERT INTO accounts (id, balance) VALUES (?, ?)", 1, 100.0)
    await conn.execute("INSERT INTO accounts (id, balance) VALUES (?, ?)", 2, 50.0)

    # Successful transaction: transfer 25 from account 1 to account 2
    async with conn.transaction() as tx:
        await tx.execute(
            "UPDATE accounts SET balance = balance - ? WHERE id = ?", 25.0, 1
        )
        await tx.execute(
            "UPDATE accounts SET balance = balance + ? WHERE id = ?", 25.0, 2
        )

    rows = await conn.fetch("SELECT * FROM accounts ORDER BY id")
    assert rows[0]["balance"] == 75.0, "Account 1 debited"
    assert rows[1]["balance"] == 75.0, "Account 2 credited"
    print(f"After transfer: {rows}")

    # Failed transaction: rollback on error
    try:
        async with conn.transaction() as tx:
            await tx.execute(
                "UPDATE accounts SET balance = balance - ? WHERE id = ?", 999.0, 1
            )
            raise ValueError("Simulated business logic error")
    except ValueError:
        pass  # Expected — transaction rolled back

    rows_after = await conn.fetch("SELECT * FROM accounts ORDER BY id")
    assert rows_after[0]["balance"] == 75.0, "Rollback preserved original balance"
    print(f"After rollback: {rows_after}")

    await conn.close()


asyncio.run(test_transactions())


# ── 5. Uninitialized access raises RuntimeError ────────────────────
# Calling execute/fetch/fetchone before initialize() raises RuntimeError
# with a clear message.


async def test_uninitialized():
    """Verify that uninitialized access is caught."""
    conn = ConnectionManager("sqlite:///:memory:")

    try:
        await conn.execute("SELECT 1")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not initialized" in str(e).lower()
        print(f"Uninitialized access caught: {e}")

    try:
        await conn.fetch("SELECT 1")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        print("fetch() also caught")

    try:
        await conn.fetchone("SELECT 1")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        print("fetchone() also caught")


asyncio.run(test_uninitialized())


# ── 6. Index creation helper ───────────────────────────────────────
# create_index() wraps CREATE INDEX IF NOT EXISTS portably.


async def test_create_index():
    """Create an index on an existing table."""
    conn = ConnectionManager("sqlite:///:memory:")
    await conn.initialize()

    await conn.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, category TEXT, price REAL)"
    )
    await conn.execute(
        "INSERT INTO items (id, category, price) VALUES (?, ?, ?)", 1, "A", 10.0
    )

    # Create an index — idempotent, safe to call multiple times
    await conn.create_index("idx_items_category", "items", "category")
    await conn.create_index(
        "idx_items_category", "items", "category"
    )  # No error on duplicate

    # Verify the index works by querying
    rows = await conn.fetch("SELECT * FROM items WHERE category = ?", "A")
    assert len(rows) == 1
    print("Index created and query works")

    await conn.close()


asyncio.run(test_create_index())


# ── 7. Infrastructure stores — the bigger picture ──────────────────
# ConnectionManager is the foundation for all database-backed
# infrastructure in Kailash. The infrastructure module provides:
#
#   DBEventStoreBackend   — Workflow event sourcing
#   DBCheckpointStore     — Node-level checkpointing for recovery
#   DBDeadLetterQueue     — Failed message storage
#   DBExecutionStore      — Workflow execution history
#   DBIdempotencyStore    — Exactly-once execution guarantees
#   StoreFactory          — Auto-detects URL and creates all stores
#
# Each store takes a ConnectionManager and calls initialize() to
# create its own tables.

from kailash.infrastructure import StoreFactory


async def test_store_factory():
    """Show how StoreFactory wraps ConnectionManager."""
    # StoreFactory auto-detects the database from environment or URL.
    # Here we use the explicit URL constructor for demonstration.
    conn = ConnectionManager("sqlite:///:memory:")
    await conn.initialize()

    # The infrastructure stores all accept a ConnectionManager
    from kailash.infrastructure import DBEventStoreBackend, DBExecutionStore

    event_store = DBEventStoreBackend(conn)
    await event_store.initialize()

    exec_store = DBExecutionStore(conn)
    await exec_store.initialize()

    print("Infrastructure stores initialized on shared ConnectionManager")

    await conn.close()


asyncio.run(test_store_factory())


# ── 8. Placeholder translation ─────────────────────────────────────
# The dialect layer translates canonical ? placeholders to the target
# dialect's format. This means you write queries once and they work
# across SQLite, PostgreSQL, and MySQL.

from kailash.db import SQLiteDialect, PostgresDialect, MySQLDialect

sqlite_d = SQLiteDialect()
pg_d = PostgresDialect()
mysql_d = MySQLDialect()

query = "SELECT * FROM users WHERE id = ? AND name = ?"

sqlite_translated = sqlite_d.translate_query(query)
pg_translated = pg_d.translate_query(query)
mysql_translated = mysql_d.translate_query(query)

# SQLite keeps ? as-is
assert "?" in sqlite_translated
print(f"SQLite:      {sqlite_translated}")

# PostgreSQL uses $1, $2, ...
assert "$1" in pg_translated and "$2" in pg_translated
print(f"PostgreSQL:  {pg_translated}")

# MySQL uses %s
assert "%s" in mysql_translated
print(f"MySQL:       {mysql_translated}")

# ── 9. Double close is safe ────────────────────────────────────────
# Calling close() on an already-closed ConnectionManager is a no-op.


async def test_double_close():
    """Verify that double-close does not raise."""
    conn = ConnectionManager("sqlite:///:memory:")
    await conn.initialize()
    await conn.close()
    await conn.close()  # Should not raise
    print("Double close is safe")


asyncio.run(test_double_close())

print("PASS: 00-core/10_connection_manager")
