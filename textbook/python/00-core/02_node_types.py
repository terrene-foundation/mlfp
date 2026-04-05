# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Core / Built-in Node Types
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Discover and use the built-in node types from NodeRegistry
# LEVEL: Basic
# PARITY: Equivalent — Python has ~18 specialized node types via
#         @register_node; Rust uses Agent struct with AgentConfig (DIV-011)
# VALIDATES: NodeRegistry, NodeMetadata, NodeParameter, built-in nodes
#
# Run: uv run python textbook/python/00-core/02_node_types.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from kailash.nodes.base import Node, NodeMetadata, NodeParameter, NodeRegistry

# ── 1. NodeRegistry: discover available nodes ───────────────────────
# The registry is populated by @register_node decorators at import time.
# It maps string names → Node classes.

registered_names = NodeRegistry.list()
assert isinstance(registered_names, (list, dict)), "list() returns registered nodes"
print(f"Registered node types: {len(registered_names)}")

# PythonCodeNode is always available (built-in)
python_code_cls = NodeRegistry.get("PythonCodeNode")
assert python_code_cls is not None, "PythonCodeNode must be registered"
assert issubclass(python_code_cls, Node), "Registered class extends Node"

# ── 2. NodeMetadata: inspect node information ───────────────────────
# Every Node class has metadata (name, description, version, tags).

metadata = NodeMetadata(name="ExampleNode", description="A test node", version="1.0.0")
assert metadata.name == "ExampleNode"
assert metadata.version == "1.0.0"
assert isinstance(metadata.tags, set), "Tags default to empty set"

# ── 3. NodeParameter: define inputs and outputs ─────────────────────
# Parameters describe what data flows into and out of a node.

param = NodeParameter(
    name="input_data",
    type="str",
    description="The input string to process",
    required=True,
)
assert param.name == "input_data"
assert param.required is True

optional_param = NodeParameter(
    name="separator",
    type="str",
    description="Optional separator",
    required=False,
    default=",",
)
assert optional_param.required is False
assert optional_param.default == ","

# ── 4. Using PythonCodeNode in a workflow ───────────────────────────
# PythonCodeNode executes Python code strings in a sandboxed environment.
# Config keys: "code" (str), "inputs" (dict), "output_type" (str)

from kailash import WorkflowBuilder, LocalRuntime

builder = WorkflowBuilder()

# A node that generates a list
builder.add_node(
    "PythonCodeNode",
    "make_list",
    {
        "code": "output = [1, 2, 3, 4, 5]",
        "output_type": "list",
    },
)

# A node that sums the list
builder.add_node(
    "PythonCodeNode",
    "sum_it",
    {
        "code": "output = sum(numbers)",
        "inputs": {"numbers": "list"},
        "output_type": "int",
    },
)

builder.connect("make_list", "sum_it", mapping={"output": "numbers"})

workflow = builder.build(name="node-types-demo")
assert len(workflow.node_instances) == 2

# Execute and verify
runtime = LocalRuntime()
results, run_id = runtime.execute(workflow)
runtime.close()

assert run_id is not None, "execute() returns a run_id"
assert isinstance(results, dict), "Results are a dict keyed by node_id"

# Results are nested: {node_id: {output_name: value}}
sum_result = results.get("sum_it", {})
print(f"Sum result: {sum_result}")

print("PASS: 00-core/02_node_types")
