# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Core / WorkflowBuilder Basics
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Create, build, and inspect a workflow using WorkflowBuilder
# LEVEL: Basic
# PARITY: Full — Rust has WorkflowBuilder::new() (DIV-007: Rust does
#         more pre-computation at build time)
# VALIDATES: WorkflowBuilder(), add_node(), connect(), build()
#
# Run: uv run python textbook/python/00-core/01_workflow_builder.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from kailash import WorkflowBuilder
from kailash.workflow.graph import Workflow

# ── 1. Create an empty builder ──────────────────────────────────────

builder = WorkflowBuilder()

# The builder starts with no nodes and no connections
assert len(builder.nodes) == 0, "Builder should start empty"
assert len(builder.connections) == 0, "No connections yet"

# ── 2. Add nodes ────────────────────────────────────────────────────
# Preferred pattern: add_node("NodeType", "node_id", {config_dict})
# Returns the node_id for chaining convenience.

node_id_1 = builder.add_node(
    "PythonCodeNode",
    "generate",
    {
        "code": "output = 'Hello from Kailash!'",
        "output_type": "str",
    },
)

node_id_2 = builder.add_node(
    "PythonCodeNode",
    "transform",
    {
        "code": "output = text.upper()",
        "inputs": {"text": "str"},
        "output_type": "str",
    },
)

assert node_id_1 == "generate", "add_node returns the node_id"
assert node_id_2 == "transform"
assert len(builder.nodes) == 2, "Two nodes added"

# ── 3. Connect nodes ────────────────────────────────────────────────
# connect() supports three patterns:
#   connect("src", "dst")                    → default "data" → "data"
#   connect("src", "dst", mapping={"a": "b"})→ explicit port mapping
#   connect("src", "dst", from_output="a", to_input="b") → named ports

builder.connect("generate", "transform", mapping={"output": "text"})

assert len(builder.connections) == 1, "One connection added"

# ── 4. Build the workflow ───────────────────────────────────────────
# build() validates the graph and returns an immutable Workflow object.
# Optional: pass workflow_id, name, description, version.

workflow = builder.build(
    name="hello-workflow",
    description="A minimal two-node workflow",
    version="1.0.0",
)

assert isinstance(workflow, Workflow), "build() returns a Workflow"
assert workflow.name == "hello-workflow"
assert workflow.version == "1.0.0"

# ── 5. Inspect the workflow ─────────────────────────────────────────
# The Workflow object has node_instances and connections.

assert len(workflow.node_instances) == 2
assert "generate" in workflow.node_instances
assert "transform" in workflow.node_instances
assert len(workflow.connections) == 1

# ── 6. Edge case: duplicate node IDs ───────────────────────────────
# Adding a node with an existing ID raises WorkflowValidationError.

from kailash.sdk_exceptions import WorkflowValidationError

try:
    builder.add_node("PythonCodeNode", "generate", {"code": "x = 1"})
    assert False, "Should have raised WorkflowValidationError"
except WorkflowValidationError:
    pass  # Expected: duplicate node ID

# ── 7. Edge case: connect non-existent node ────────────────────────

try:
    builder.connect("nonexistent", "transform")
    assert False, "Should have raised WorkflowValidationError"
except WorkflowValidationError:
    pass  # Expected: source node not found

# ── 8. Edge case: build with no nodes ──────────────────────────────

empty_builder = WorkflowBuilder()
empty_wf = empty_builder.build(name="empty")
assert len(empty_wf.node_instances) == 0, "Empty workflow is valid"

print("PASS: 00-core/01_workflow_builder")
