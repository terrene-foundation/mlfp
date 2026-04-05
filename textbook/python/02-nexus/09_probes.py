# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Nexus / Probes
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Configure health and readiness probes for Kubernetes
# LEVEL: Advanced
# PARITY: Python-only
# VALIDATES: ProbeManager, ProbeState, ProbeResponse, state transitions,
#            liveness/readiness/startup checks, readiness callbacks
#
# Run: uv run python textbook/python/02-nexus/09_probes.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from nexus.probes import ProbeManager, ProbeResponse, ProbeState

# ── 1. ProbeState enum ─────────────────────────────────────────────
# ProbeState models the Kubernetes probe lifecycle. Transitions are
# monotonic: STARTING -> READY -> DRAINING. FAILED is terminal
# (only reset() can recover from it).

assert ProbeState.STARTING.value == "starting"
assert ProbeState.READY.value == "ready"
assert ProbeState.DRAINING.value == "draining"
assert ProbeState.FAILED.value == "failed"

# ── 2. Create a ProbeManager ──────────────────────────────────────
# ProbeManager is thread-safe. All state transitions are atomic
# and validated against the allowed transition graph.

probes = ProbeManager()

assert probes.state == ProbeState.STARTING
assert probes.is_alive is True  # Not FAILED, so alive
assert probes.is_ready is False  # Not READY yet
assert probes.is_started is False  # Not past STARTING yet

# ── 3. Liveness probe (/healthz) ──────────────────────────────────
# check_liveness() returns 200 for all states except FAILED.
# This tells Kubernetes the process is alive and should not be
# restarted (even if it's still starting up).

liveness = probes.check_liveness()

assert isinstance(liveness, ProbeResponse)
assert liveness.status == "ok"
assert liveness.http_status == 200
assert "uptime_seconds" in liveness.details

# ProbeResponse.to_dict() serializes for JSON responses
liveness_dict = liveness.to_dict()
assert liveness_dict["status"] == "ok"

# ── 4. Startup probe (/startup) ───────────────────────────────────
# check_startup() returns 200 once past STARTING state.
# Kubernetes uses this to know when the initial boot is done.

startup = probes.check_startup()

assert startup.status == "starting"  # Still in STARTING state
assert startup.http_status == 503  # Not started yet
assert "elapsed_seconds" in startup.details

# ── 5. Readiness probe (/readyz) ──────────────────────────────────
# check_readiness() returns 200 only in READY state AND when all
# readiness callbacks pass. Kubernetes removes the pod from the
# load balancer when readiness fails.

readiness = probes.check_readiness()

assert readiness.status == "not_ready"
assert readiness.http_status == 503
assert readiness.details["state"] == "starting"

# ── 6. State transition: STARTING -> READY ─────────────────────────
# mark_ready() transitions to READY state. Returns True on success.

success = probes.mark_ready()

assert success is True
assert probes.state == ProbeState.READY
assert probes.is_alive is True
assert probes.is_ready is True
assert probes.is_started is True

# Now all three probes return 200
assert probes.check_liveness().http_status == 200
assert probes.check_readiness().http_status == 200
assert probes.check_startup().http_status == 200

# Startup probe includes the startup duration
startup_resp = probes.check_startup()
assert startup_resp.status == "started"
assert startup_resp.details.get("startup_duration_seconds") is not None

# ── 7. Workflow count tracking ─────────────────────────────────────
# ProbeManager tracks the number of registered workflows.
# This appears in readiness and startup probe details.

probes.set_workflow_count(5)

readiness = probes.check_readiness()
assert readiness.details["workflows"] == 5

# ── 8. Readiness callbacks ────────────────────────────────────────
# Additional readiness checks can be registered as callbacks.
# Each callback must return True for readiness to pass.


def check_database_connection() -> bool:
    """Simulate a database connectivity check."""
    return True


def check_model_loaded() -> bool:
    """Simulate checking if ML model is loaded."""
    return True


probes.add_readiness_check(check_database_connection)
probes.add_readiness_check(check_model_loaded)

# Both callbacks pass, so readiness still returns 200
readiness = probes.check_readiness()
assert readiness.http_status == 200

# ── 9. Failing readiness callback ─────────────────────────────────
# If any callback returns False, readiness fails with details
# about which checks failed.

probes2 = ProbeManager()
probes2.mark_ready()


def always_fails() -> bool:
    return False


probes2.add_readiness_check(always_fails)

readiness = probes2.check_readiness()
assert readiness.http_status == 503
assert readiness.status == "not_ready"
assert "failed_checks" in readiness.details
assert "always_fails" in readiness.details["failed_checks"]

# ── 10. State transition: READY -> DRAINING ────────────────────────
# mark_draining() signals graceful shutdown. Kubernetes stops sending
# new traffic but lets existing requests complete.

success = probes.mark_draining()

assert success is True
assert probes.state == ProbeState.DRAINING
assert probes.is_alive is True  # Still alive (don't restart)
assert probes.is_ready is False  # No longer accepting traffic
assert probes.is_started is True  # Past startup

# Liveness still 200 (process is alive, just draining)
assert probes.check_liveness().http_status == 200
# Readiness now 503 (stop sending new requests)
assert probes.check_readiness().http_status == 503

# ── 11. State transition: -> FAILED ────────────────────────────────
# mark_failed() is reachable from any non-FAILED state.
# It's a terminal state -- the process should be restarted.

probes3 = ProbeManager()
probes3.mark_ready()
success = probes3.mark_failed(reason="Out of memory")

assert success is True
assert probes3.state == ProbeState.FAILED
assert probes3.is_alive is False  # Liveness fails
assert probes3.is_ready is False

liveness = probes3.check_liveness()
assert liveness.http_status == 503
assert liveness.status == "failed"
assert liveness.details["reason"] == "Out of memory"

# ── 12. Invalid transitions are rejected ───────────────────────────
# The state machine enforces valid transitions. Invalid ones return
# False without changing state.

probes4 = ProbeManager()

# Can't go STARTING -> DRAINING (must be READY first)
assert probes4.mark_draining() is False
assert probes4.state == ProbeState.STARTING

# Can't go FAILED -> READY (FAILED is terminal)
probes4.mark_failed(reason="crash")
assert probes4.mark_ready() is False
assert probes4.state == ProbeState.FAILED

# ── 13. Reset for recovery/testing ─────────────────────────────────
# reset() is the only way to recover from FAILED state.
# It returns to STARTING so the full lifecycle can restart.

probes4.reset()

assert probes4.state == ProbeState.STARTING
assert probes4.is_alive is True

# Can now transition normally again
assert probes4.mark_ready() is True
assert probes4.state == ProbeState.READY

# ── 14. ProbeResponse serialization ────────────────────────────────
# to_dict() produces a clean JSON-serializable dict.
# The "details" key is omitted when empty.

simple_resp = ProbeResponse(status="ok", http_status=200)
d = simple_resp.to_dict()
assert d == {"status": "ok"}  # No details key when empty

detailed_resp = ProbeResponse(
    status="ready",
    http_status=200,
    details={"workflows": 3, "state": "ready"},
)
d2 = detailed_resp.to_dict()
assert d2["status"] == "ready"
assert d2["details"]["workflows"] == 3

# ── 15. Key concepts ──────────────────────────────────────────────
# - ProbeState: STARTING -> READY -> DRAINING (FAILED is terminal)
# - ProbeManager: thread-safe, atomic state transitions
# - check_liveness() -> /healthz: 200 unless FAILED
# - check_readiness() -> /readyz: 200 only in READY + callbacks pass
# - check_startup() -> /startup: 200 once past STARTING
# - mark_ready(), mark_draining(), mark_failed(): state transitions
# - add_readiness_check(callback): custom readiness checks
# - set_workflow_count(n): track registered workflows
# - reset(): recover from FAILED (for testing/recovery)
# - install(app): mounts /healthz, /readyz, /startup routes
# - NOTE: We don't call install() or app.start() because they block

print("PASS: 02-nexus/09_probes")
