# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — DataFlow / Multi-Tenant Context Switching
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Isolate data using multi-tenant context switching with
#            sync and async context managers
# LEVEL: Advanced
# PARITY: Full — Rust has QueryInterceptor for tenancy
# VALIDATES: TenantContextSwitch, TenantInfo, get_current_tenant_id,
#            register_tenant, switch(), aswitch(), require_tenant,
#            deactivate_tenant, activate_tenant, get_stats()
#
# Run: uv run python textbook/python/01-dataflow/08_multi_tenant.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os
import tempfile

from dataflow import DataFlow, TenantContextSwitch, TenantInfo, get_current_tenant_id

# ── Setup ───────────────────────────────────────────────────────────

db_path = os.path.join(tempfile.gettempdir(), "textbook_dataflow_08.db")
db = DataFlow(database_url=f"sqlite:///{db_path}")


@db.model
class Project:
    name: str
    status: str = "active"


async def main():
    await db.initialize()

    # ── 1. ACCESS — db.tenant_context property ──────────────────────
    # TenantContextSwitch is accessed via db.tenant_context.
    # It provides sync and async context managers for safely switching
    # between tenant contexts with guaranteed isolation.

    ctx = db.tenant_context
    assert isinstance(ctx, TenantContextSwitch)
    print("TenantContextSwitch accessible via db.tenant_context")

    # ── 2. REGISTER tenants ─────────────────────────────────────────
    # Register tenants before switching to them. Each tenant has an id,
    # a human-readable name, and optional metadata.

    tenant_a = ctx.register_tenant("acme", "Acme Corporation", {"region": "APAC"})
    assert isinstance(tenant_a, TenantInfo)
    assert tenant_a.tenant_id == "acme"
    assert tenant_a.name == "Acme Corporation"
    assert tenant_a.metadata == {"region": "APAC"}
    assert tenant_a.active is True
    print(f"Registered tenant: {tenant_a.name}")

    tenant_b = ctx.register_tenant("globex", "Globex Inc.", {"region": "EU"})
    assert tenant_b.tenant_id == "globex"

    # Duplicate registration raises ValueError
    try:
        ctx.register_tenant("acme", "Acme Again")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "already registered" in str(e)

    # Invalid tenant_id raises ValueError
    try:
        ctx.register_tenant("", "Empty ID")
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    # ── 3. LIST and GET tenants ─────────────────────────────────────

    all_tenants = ctx.list_tenants()
    assert len(all_tenants) == 2
    assert all(isinstance(t, TenantInfo) for t in all_tenants)

    found = ctx.get_tenant("acme")
    assert found is not None
    assert found.tenant_id == "acme"

    not_found = ctx.get_tenant("nonexistent")
    assert not_found is None

    # ── 4. SWITCH — sync context manager ────────────────────────────
    # with ctx.switch(tenant_id) sets the tenant for the block.
    # Previous context is restored on exit, even on exception.

    # Before any switch, no tenant is active
    assert ctx.get_current_tenant() is None
    assert get_current_tenant_id() is None, "Module-level helper also returns None"

    with ctx.switch("acme") as tenant_info:
        assert isinstance(tenant_info, TenantInfo)
        assert tenant_info.tenant_id == "acme"

        # Inside the context, the current tenant is set
        assert ctx.get_current_tenant() == "acme"
        assert get_current_tenant_id() == "acme", "Module-level helper reads context"
        print(f"Inside sync switch: tenant={ctx.get_current_tenant()}")

    # After the context manager exits, the previous context is restored
    assert ctx.get_current_tenant() is None
    print("After sync switch: tenant restored to None")

    # ── 5. ASWITCH — async context manager ──────────────────────────
    # async with ctx.aswitch(tenant_id) — same semantics for async code.
    # contextvars automatically propagates context to async tasks.

    async with ctx.aswitch("globex") as tenant_info:
        assert tenant_info.tenant_id == "globex"
        assert ctx.get_current_tenant() == "globex"
        print(f"Inside async switch: tenant={ctx.get_current_tenant()}")

    assert ctx.get_current_tenant() is None
    print("After async switch: tenant restored to None")

    # ── 6. NESTED switches ──────────────────────────────────────────
    # Nested switches properly track and restore the previous context.

    with ctx.switch("acme"):
        assert ctx.get_current_tenant() == "acme"

        with ctx.switch("globex"):
            assert ctx.get_current_tenant() == "globex"

        # Inner switch exited — restored to "acme"
        assert ctx.get_current_tenant() == "acme"

    # Outer switch exited — restored to None
    assert ctx.get_current_tenant() is None
    print("Nested switches: proper context restoration verified")

    # ── 7. REQUIRE_TENANT — strict enforcement ──────────────────────
    # require_tenant() raises RuntimeError if no tenant context is active.
    # Useful for operations that must run within a tenant scope.

    try:
        ctx.require_tenant()
        assert False, "Should raise RuntimeError"
    except RuntimeError as e:
        assert "No tenant context is active" in str(e)

    with ctx.switch("acme"):
        tenant_id = ctx.require_tenant()
        assert tenant_id == "acme"

    # ── 8. DEACTIVATE and ACTIVATE tenants ──────────────────────────
    # Deactivated tenants remain registered but cannot be switched to.

    ctx.deactivate_tenant("globex")
    assert ctx.is_tenant_registered("globex") is True
    assert ctx.is_tenant_active("globex") is False

    # Switching to a deactivated tenant raises ValueError
    try:
        with ctx.switch("globex"):
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "not active" in str(e)

    # Reactivate the tenant
    ctx.activate_tenant("globex")
    assert ctx.is_tenant_active("globex") is True

    # Now switching works again
    with ctx.switch("globex"):
        assert ctx.get_current_tenant() == "globex"
    print("Deactivate/activate: tenant lifecycle verified")

    # ── 9. SWITCH to unregistered tenant ────────────────────────────
    # Raises ValueError with a helpful message listing available tenants.

    try:
        with ctx.switch("unknown-corp"):
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "not registered" in str(e)
        assert "Available tenants" in str(e)

    # ── 10. UNREGISTER a tenant ─────────────────────────────────────

    ctx.register_tenant("temp", "Temporary Tenant")
    assert ctx.is_tenant_registered("temp") is True

    ctx.unregister_tenant("temp")
    assert ctx.is_tenant_registered("temp") is False

    # Cannot unregister while active context
    with ctx.switch("acme"):
        try:
            ctx.unregister_tenant("acme")
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "active context" in str(e)

    # ── 11. STATS — context switching statistics ────────────────────

    stats = ctx.get_stats()
    assert isinstance(stats, dict)
    assert stats["total_tenants"] == 2, "Two registered tenants"
    assert stats["active_tenants"] == 2, "Both are active"
    assert stats["total_switches"] > 0, "Switches have been performed"
    assert stats["active_switches"] == 0, "No switch currently active"
    assert stats["current_tenant"] is None, "No current tenant outside switch"
    print(f"Stats: {stats['total_switches']} total switches performed")

    # ── 12. Context restoration on exception ────────────────────────
    # Even when an exception occurs, the previous context is restored.

    with ctx.switch("acme"):
        try:
            with ctx.switch("globex"):
                assert ctx.get_current_tenant() == "globex"
                raise RuntimeError("Simulated failure")
        except RuntimeError:
            pass

        # Restored to "acme" despite exception in inner block
        assert ctx.get_current_tenant() == "acme"

    assert ctx.get_current_tenant() is None
    print("Exception safety: context restored after error")

    # ── Cleanup ─────────────────────────────────────────────────────
    await db.stop()


asyncio.run(main())

try:
    os.unlink(db_path)
except OSError:
    pass

print("PASS: 01-dataflow/08_multi_tenant")
