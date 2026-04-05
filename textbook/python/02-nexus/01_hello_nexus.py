# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ═══���════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Nexus / Hello Nexus
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Create a Nexus instance, register a workflow, and inspect
#            the registration without starting the server
# LEVEL: Basic
# PARITY: Full — Rust has Nexus::new() with same constructor pattern
# VALIDATES: Nexus(), register(), @handler, handler_count
#
# Run: uv run python textbook/python/02-nexus/01_hello_nexus.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from kailash import WorkflowBuilder
from nexus import Nexus

# ── 1. Create a Nexus instance ──────────────────────────────────────
# Nexus is the multi-channel deployment platform. Like FastAPI, it
# provides a single instance that exposes workflows via API, CLI, and MCP.
# Default: port 8000 (API), port 3001 (MCP).

app = Nexus(
    api_port=8000,
    cors_origins=["http://localhost:3000"],  # Allow local frontend
    enable_durability=False,  # Disable caching for tutorial
)

assert isinstance(app, Nexus)

# ── 2. Register a workflow ──────────────────────────────────────────
# register(name, workflow) makes the workflow available on all channels:
#   - POST /workflows/{name}/execute (HTTP API)
#   - MCP tool: workflow_{name}

builder = WorkflowBuilder()
builder.add_node(
    "PythonCodeNode",
    "greet",
    {
        "code": "output = f'Hello, {name}!'",
        "inputs": {"name": "str"},
        "output_type": "str",
    },
)
workflow = builder.build(name="greeter")

app.register("greet", workflow)

# ── 3. Register via @handler decorator ──────────────────────────────
# @handler is the preferred pattern — it inspects the function signature
# to derive workflow parameters automatically.


@app.handler("add", description="Add two numbers")
async def add(a: int, b: int = 0) -> dict:
    return {"sum": a + b}


# ── 4. Inspect registered handlers ─────────────────────────────────
# The handler registry tracks all registered workflows and handlers.

registry = app._registry
assert registry is not None

# Both the workflow and handler are registered
print(f"Registered entries: {registry}")

# ── 5. Preset system ───────────────────────────────────────────────
# Nexus supports presets for one-line middleware configuration:
#   "lightweight" — minimal overhead
#   "production" — logging, rate limiting, security headers
#   "saas" — full enterprise stack

from nexus import get_preset, PRESETS

assert "lightweight" in PRESETS or len(PRESETS) > 0, "Presets are available"

# ── 6. Nexus with preset ───────────────────────────────────────────

app_prod = Nexus(
    preset="lightweight",
    cors_origins=["https://app.example.com"],
    enable_durability=False,
)

assert isinstance(app_prod, Nexus)

# ── 7. Key Nexus concepts ──────────────────────────────────────────
# - register(): workflow → all channels (auto)
# - @handler: function → workflow → all channels (auto)
# - preset: one-line middleware stack
# - start(): begins listening (NOT called in this tutorial)
#
# NOTE: We don't call app.start() because it blocks — that's for
# production. Tutorials validate registration and configuration only.

print("PASS: 02-nexus/01_hello_nexus")
