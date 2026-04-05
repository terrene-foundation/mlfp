# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Nexus / MCP Transport
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Configure MCP transport for AI agent integration
# LEVEL: Intermediate
# PARITY: Full — Rust has MCPTransport with same namespace/port pattern
# VALIDATES: MCPTransport, MCP tool registration, Transport ABC
#
# Run: uv run python textbook/python/02-nexus/03_mcp_transport.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from nexus.transports import MCPTransport, Transport

# ── 1. MCPTransport overview ──────────────────────────────────────
# MCPTransport registers Nexus handlers as MCP tools via FastMCP.
# AI agents (Claude, GPT, etc.) can call these tools over the MCP
# protocol. The transport runs in a background thread with its own
# event loop, matching the MCP server's async requirements.

assert issubclass(MCPTransport, Transport)

# ── 2. Create an MCPTransport ─────────────────────────────────────
# Key parameters:
#   port:        WebSocket port for MCP server (default 3001)
#   namespace:   Tool name prefix to avoid collisions (default "nexus")
#   server_name: FastMCP server identity (default "kailash-nexus")

mcp = MCPTransport(
    port=4001,
    namespace="myapp",
    server_name="my-nexus-service",
)

assert isinstance(mcp, MCPTransport)
assert isinstance(mcp, Transport)

# ── 3. Inspect transport properties ───────────────────────────────
# MCPTransport identifies itself as "mcp".

assert mcp.name == "mcp"
assert mcp.port == 4001
assert mcp.is_running is False

# Before start(), no FastMCP server exists.
assert mcp._server is None

# ── 4. Tool naming convention ─────────────────────────────────────
# When a handler named "greet" is registered, MCPTransport creates
# an MCP tool named "{namespace}_{handler_name}".
# With namespace="myapp", the tool becomes "myapp_greet".
#
# This prevents collisions when multiple Nexus instances serve
# different tools through the same MCP platform.

assert mcp._namespace == "myapp"

# ── 5. Health check ───────────────────────────────────────────────
# Like all transports, MCPTransport provides health_check().

health = mcp.health_check()

assert health["transport"] == "mcp"
assert health["running"] is False
assert health["port"] == 4001
assert health["server"] is False  # No FastMCP server yet

# ── 6. MCPTransport with default namespace ────────────────────────
# The default namespace is "nexus", so a handler called "add"
# becomes the MCP tool "nexus_add".

mcp_default = MCPTransport()

assert mcp_default.port == 3001
assert mcp_default._namespace == "nexus"
assert mcp_default._server_name == "kailash-nexus"

# ── 7. Adding MCPTransport to Nexus ───────────────────────────────
# Nexus doesn't add MCPTransport by default (it's opt-in).
# Use app.add_transport() to register it for lifecycle management.
# Nexus.start() calls transport.start(registry) on all transports.

from nexus import Nexus

app = Nexus(
    api_port=8000,
    enable_durability=False,
)

custom_mcp = MCPTransport(port=5001, namespace="tutorial")
app.add_transport(custom_mcp)

# The transport is now managed by Nexus
assert custom_mcp in app._transports

# ── 8. Handler registration and MCP tools ─────────────────────────
# When Nexus registers a handler, it notifies all transports via
# on_handler_registered(). MCPTransport uses this to hot-register
# new MCP tools at runtime (if the server is already running).
#
# In this tutorial, the server isn't running, so on_handler_registered
# is a no-op. But the handler is still in the registry for when
# start() is eventually called.

from nexus.registry import HandlerDef

dummy_def = HandlerDef(
    name="example",
    description="An example handler",
)

# Before start, on_handler_registered is safe to call (no-op)
custom_mcp.on_handler_registered(dummy_def)
assert custom_mcp._server is None  # Still no server, call was a no-op

# ── 9. Background thread architecture ─────────────────────────────
# MCPTransport runs the FastMCP server in a daemon thread with its
# own asyncio event loop. This is because the main thread is blocked
# by uvicorn (HTTP), so MCP needs its own loop.
#
# Thread lifecycle:
#   start() -> creates FastMCP, registers tools, spawns thread
#   stop()  -> signals the loop to stop, joins the thread (5s timeout)
#
# The daemon flag ensures the thread dies with the process.

assert mcp._thread is None  # No thread until start() is called

# ── 10. Key concepts ──────────────────────────────────────────────
# - MCPTransport registers handlers as MCP tools for AI agents
# - Tool names are "{namespace}_{handler_name}" to avoid collisions
# - Runs in a background daemon thread with its own event loop
# - Added to Nexus via app.add_transport(MCPTransport(...))
# - health_check() reports transport status
# - Hot-registers new tools via on_handler_registered()
# - NOTE: We don't call start() because it requires FastMCP + blocks

print("PASS: 02-nexus/03_mcp_transport")
