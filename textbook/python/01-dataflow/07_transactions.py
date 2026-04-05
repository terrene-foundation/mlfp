# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — DataFlow / Transaction Management
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Use transaction management for atomic operations with
#            commit, rollback, and isolation level control
# LEVEL: Advanced
# PARITY: Full — Rust has DataFlowTransaction with RAII rollback
# VALIDATES: TransactionManager, transaction() context manager,
#            commit/rollback semantics, rollback_all(), get_active_transactions()
#
# Run: uv run python textbook/python/01-dataflow/07_transactions.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os
import tempfile

from dataflow import DataFlow
from dataflow.features.transactions import TransactionManager

# ── Setup ───────────────────────────────────────────────────────────

db_path = os.path.join(tempfile.gettempdir(), "textbook_dataflow_07.db")
db = DataFlow(database_url=f"sqlite:///{db_path}")


@db.model
class Account:
    owner: str
    balance: float
    currency: str = "SGD"


async def main():
    await db.initialize()

    # Seed data
    await db.express.create(
        "Account", {"owner": "Alice", "balance": 1000.0, "currency": "SGD"}
    )
    await db.express.create(
        "Account", {"owner": "Bob", "balance": 500.0, "currency": "SGD"}
    )

    # ── 1. ACCESS — db.transactions property ────────────────────────
    # TransactionManager is accessed via db.transactions.

    txn_mgr = db.transactions
    assert isinstance(txn_mgr, TransactionManager)
    print("TransactionManager accessible via db.transactions")

    # ── 2. TRANSACTION — context manager for atomic operations ──────
    # transaction() yields a transaction context dict with:
    #   id, isolation_level, status, operations
    #
    # On normal exit: status becomes "committed"
    # On exception:   status becomes "rolled_back"

    with txn_mgr.transaction() as txn:
        assert txn["status"] == "active"
        assert txn["isolation_level"] == "READ_COMMITTED", "Default isolation"
        assert "id" in txn

        # Track operations inside the transaction
        txn["operations"].append({"type": "debit", "account": "Alice", "amount": 200})
        txn["operations"].append({"type": "credit", "account": "Bob", "amount": 200})

        print(f"Transaction {txn['id']}: {len(txn['operations'])} operations")

    # After the context manager exits normally, status is "committed"
    assert txn["status"] == "committed"
    print("Transaction committed successfully")

    # ── 3. ISOLATION LEVELS ─────────────────────────────────────────
    # Specify isolation level when creating a transaction.

    with txn_mgr.transaction(isolation_level="SERIALIZABLE") as txn:
        assert txn["isolation_level"] == "SERIALIZABLE"
        txn["operations"].append({"type": "audit", "details": "balance check"})

    assert txn["status"] == "committed"
    print("SERIALIZABLE transaction committed")

    # ── 4. ROLLBACK on exception ────────────────────────────────────
    # If an exception occurs inside the context manager, the transaction
    # is automatically rolled back.

    try:
        with txn_mgr.transaction() as txn:
            txn["operations"].append(
                {"type": "debit", "account": "Alice", "amount": 9999}
            )
            # Simulate a business rule violation
            raise ValueError("Insufficient funds")
    except ValueError:
        pass  # Expected

    assert txn["status"] == "rolled_back"
    assert txn.get("error") == "Insufficient funds"
    print("Transaction rolled back on exception")

    # ── 5. ACTIVE TRANSACTIONS — monitoring ─────────────────────────
    # get_active_transactions() returns currently active transactions.
    # After a transaction completes (commit or rollback), it is removed.

    active = txn_mgr.get_active_transactions()
    assert len(active) == 0, "No active transactions after all completed"

    # ── 6. ROLLBACK ALL — emergency operation ───────────────────────
    # rollback_all() rolls back every active transaction at once.
    # Useful for emergency shutdown or cleanup.

    result = txn_mgr.rollback_all()
    assert result["success"] is True
    assert result["count"] == 0, "No active transactions to roll back"
    assert isinstance(result["rolled_back_transactions"], list)
    print(f"Emergency rollback: {result['count']} transactions affected")

    # ── 7. Nested transaction pattern ───────────────────────────────
    # Transactions are independent — nesting creates separate contexts.

    with txn_mgr.transaction() as outer:
        outer["operations"].append({"type": "outer_op"})

        with txn_mgr.transaction() as inner:
            inner["operations"].append({"type": "inner_op"})

        # Inner committed, outer still active
        assert inner["status"] == "committed"

    assert outer["status"] == "committed"
    print("Nested transactions: both committed independently")

    # ── 8. Edge case: exception in nested transaction ───────────────
    # Inner rollback does not affect the outer transaction.

    with txn_mgr.transaction() as outer:
        outer["operations"].append({"type": "outer_op"})

        try:
            with txn_mgr.transaction() as inner:
                inner["operations"].append({"type": "risky_op"})
                raise RuntimeError("Inner failure")
        except RuntimeError:
            pass  # Inner rolled back

        assert inner["status"] == "rolled_back"
        # Outer continues
        outer["operations"].append({"type": "recovery_op"})

    assert outer["status"] == "committed"
    print("Inner rollback did not affect outer transaction")

    # ── 9. Transaction context data ─────────────────────────────────
    # The transaction context dict can carry arbitrary operation logs.

    with txn_mgr.transaction(isolation_level="READ_COMMITTED") as txn:
        txn["operations"].append(
            {
                "type": "transfer",
                "from": "Alice",
                "to": "Bob",
                "amount": 100.0,
                "currency": "SGD",
            }
        )

    assert len(txn["operations"]) == 1
    assert txn["operations"][0]["type"] == "transfer"
    assert txn["operations"][0]["amount"] == 100.0
    print("Transaction carries structured operation metadata")

    # ── Cleanup ─────────────────────────────────────────────────────
    await db.stop()


asyncio.run(main())

try:
    os.unlink(db_path)
except OSError:
    pass

print("PASS: 01-dataflow/07_transactions")
