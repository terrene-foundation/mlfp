# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Core / Connections and Wiring
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Wire nodes together using connect(), add_connection(), and
#            mapping-based port routing
# LEVEL: Basic
# PARITY: Equivalent — Python uses connect()/add_connection();
#         Rust uses connect() only (DIV-020: different method names)
# VALIDATES: connect(), add_connection(), mapping, port defaults
#
# Run: uv run python textbook/python/00-core/04_connections.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from kailash import WorkflowBuilder
from kailash.sdk_exceptions import ConnectionError, WorkflowValidationError

# ── 1. Default connection: "data" → "data" ─────────────────────────
# When no mapping is specified, connect() uses "data" as both the
# output port and input port.

builder = WorkflowBuilder()
builder.add_node("PythonCodeNode", "a", {"code": "output = 42", "output_type": "int"})
builder.add_node(
    "PythonCodeNode",
    "b",
    {"code": "output = data * 2", "inputs": {"data": "int"}, "output_type": "int"},
)
builder.connect("a", "b")  # defaults to data → data

assert len(builder.connections) == 1
conn = builder.connections[0]
assert conn["from_output"] == "data"
assert conn["to_input"] == "data"

# ── 2. Mapping-based connection ─────────────────────────────────────
# mapping={"from_port": "to_port"} gives explicit control over which
# output feeds which input. Multiple ports can be mapped at once.

builder2 = WorkflowBuilder()
builder2.add_node(
    "PythonCodeNode",
    "splitter",
    {
        "code": "first = text[:5]\nrest = text[5:]",
        "inputs": {"text": "str"},
        "output_type": "dict",
    },
)
builder2.add_node(
    "PythonCodeNode",
    "joiner",
    {
        "code": "output = f'{a} | {b}'",
        "inputs": {"a": "str", "b": "str"},
        "output_type": "str",
    },
)

# Map two outputs to two inputs in one call
builder2.connect("splitter", "joiner", mapping={"first": "a", "rest": "b"})
assert len(builder2.connections) == 2, "mapping with 2 keys creates 2 connections"

# ── 3. Named port connection (from_output / to_input) ──────────────
# Alternative to mapping for single-port connections.

builder3 = WorkflowBuilder()
builder3.add_node(
    "PythonCodeNode", "gen", {"code": "result = 99", "output_type": "int"}
)
builder3.add_node(
    "PythonCodeNode",
    "recv",
    {"code": "output = value + 1", "inputs": {"value": "int"}, "output_type": "int"},
)

builder3.connect("gen", "recv", from_output="result", to_input="value")
assert builder3.connections[0]["from_output"] == "result"
assert builder3.connections[0]["to_input"] == "value"

# ── 4. add_connection() — low-level 4-argument form ────────────────
# connect() is syntactic sugar. add_connection() is the underlying API:
#   add_connection(from_node, from_output, to_node, to_input)

builder4 = WorkflowBuilder()
builder4.add_node("PythonCodeNode", "x", {"code": "output = 1", "output_type": "int"})
builder4.add_node(
    "PythonCodeNode",
    "y",
    {"code": "output = n + 1", "inputs": {"n": "int"}, "output_type": "int"},
)

builder4.add_connection("x", "output", "y", "n")
assert len(builder4.connections) == 1
assert builder4.connections[0]["from_node"] == "x"
assert builder4.connections[0]["to_node"] == "y"

# ── 5. Fan-out: one node feeds multiple downstream nodes ───────────

builder5 = WorkflowBuilder()
builder5.add_node(
    "PythonCodeNode", "source", {"code": "output = 10", "output_type": "int"}
)
builder5.add_node(
    "PythonCodeNode",
    "double",
    {"code": "output = data * 2", "inputs": {"data": "int"}, "output_type": "int"},
)
builder5.add_node(
    "PythonCodeNode",
    "triple",
    {"code": "output = data * 3", "inputs": {"data": "int"}, "output_type": "int"},
)

builder5.connect("source", "double")
builder5.connect("source", "triple")
assert len(builder5.connections) == 2, "Fan-out: one source, two targets"

wf5 = builder5.build(name="fan-out")
assert len(wf5.node_instances) == 3

# ── 6. Edge case: self-connection ───────────────────────────────────

try:
    builder.connect("a", "a")
    assert False, "Self-connection should raise ConnectionError"
except ConnectionError:
    pass  # Expected

# ── 7. Edge case: duplicate connection ──────────────────────────────

builder6 = WorkflowBuilder()
builder6.add_node("PythonCodeNode", "p", {"code": "output = 1", "output_type": "int"})
builder6.add_node(
    "PythonCodeNode",
    "q",
    {"code": "output = data", "inputs": {"data": "int"}, "output_type": "int"},
)
builder6.connect("p", "q")

try:
    builder6.connect("p", "q")  # Same connection again
    assert False, "Duplicate connection should raise ConnectionError"
except ConnectionError:
    pass  # Expected

print("PASS: 00-core/04_connections")
