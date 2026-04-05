# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Core / Conditional Branching with SwitchNode
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Implement workflow branching with SwitchNode for conditional
#            routing based on data values
# LEVEL: Intermediate
# PARITY: Equivalent — DIV-017: Rust uses structured dict config for
#         conditions; DIV-018: cases defined in config vs runtime
# VALIDATES: SwitchNode (boolean mode, multi-case mode), MergeNode
#
# Run: uv run python textbook/python/00-core/07_conditional_node.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from typing import Any

from kailash import WorkflowBuilder
from kailash.nodes.base import Node, NodeMetadata, NodeParameter, NodeRegistry
from kailash.nodes.logic.operations import MergeNode, SwitchNode

# ── 1. SwitchNode basics — boolean mode ─────────────────────────────
# SwitchNode routes data to "true_output" or "false_output" based on
# evaluating a condition. Config keys:
#   condition_field — which field in the input dict to check
#   operator       — comparison operator (==, !=, >, <, >=, <=, in, contains)
#   value          — the value to compare against

switch = SwitchNode()

# Boolean mode: check if score >= 80
result = switch.run(
    input_data={"score": 85, "name": "Alice"},
    condition_field="score",
    operator=">=",
    value=80,
)

assert result["true_output"] is not None, "Score 85 >= 80, should route to true"
assert result["false_output"] is None, "false_output should be None"
assert result["true_output"]["score"] == 85
print(f"Boolean true: {result['true_output']}")

# Condition that evaluates to false
result_fail = switch.run(
    input_data={"score": 50, "name": "Bob"},
    condition_field="score",
    operator=">=",
    value=80,
)

assert result_fail["true_output"] is None, "Score 50 < 80, true_output is None"
assert result_fail["false_output"] is not None, "Should route to false_output"
assert result_fail["false_output"]["score"] == 50
print(f"Boolean false: {result_fail['false_output']}")

# ── 2. Supported operators ──────────────────────────────────────────
# SwitchNode supports: ==, !=, >, <, >=, <=, in, contains, is_null, is_not_null

# Equality
eq_result = switch.run(
    input_data={"status": "active"},
    condition_field="status",
    operator="==",
    value="active",
)
assert eq_result["true_output"] is not None

# Inequality
neq_result = switch.run(
    input_data={"status": "inactive"},
    condition_field="status",
    operator="!=",
    value="active",
)
assert neq_result["true_output"] is not None

# Membership (in)
in_result = switch.run(
    input_data={"role": "admin"},
    condition_field="role",
    operator="in",
    value=["admin", "superadmin"],
)
assert in_result["true_output"] is not None

# Null check
null_result = switch.run(
    input_data={"email": None},
    condition_field="email",
    operator="is_null",
    value=None,
)
assert null_result["true_output"] is not None
print("All operators validated")

# ── 3. Multi-case switching ─────────────────────────────────────────
# Instead of boolean true/false, SwitchNode can route to multiple
# named outputs using the "cases" parameter. Each case creates an
# output field named "case_<value>".

multi_result = switch.run(
    input_data={"priority": "high", "task": "deploy"},
    condition_field="priority",
    cases=["high", "medium", "low"],
)

assert multi_result["case_high"] is not None, "Input matches 'high' case"
assert multi_result["case_medium"] is None, "No match for medium"
assert multi_result["case_low"] is None, "No match for low"
assert multi_result["default"] is not None, "Default always has the data"
assert multi_result["condition_result"] == "high"
print(f"Multi-case result: matched '{multi_result['condition_result']}'")

# ── 4. MergeNode — rejoining branches ───────────────────────────────
# After conditional branching, MergeNode combines results back together.
# Merge types: "concat" (lists), "zip", "merge_dict" (dicts)

merge = MergeNode()

# Concatenation merge
concat_result = merge.run(
    data1=[1, 2, 3],
    data2=[4, 5, 6],
    merge_type="concat",
)
assert concat_result["merged_data"] == [1, 2, 3, 4, 5, 6]
print(f"Concat merge: {concat_result['merged_data']}")

# Dictionary merge
dict_result = merge.run(
    data1={"name": "Alice", "age": 30},
    data2={"age": 31, "role": "engineer"},
    merge_type="merge_dict",
)
assert dict_result["merged_data"]["age"] == 31, "Later dict wins on conflict"
assert dict_result["merged_data"]["name"] == "Alice"
assert dict_result["merged_data"]["role"] == "engineer"
print(f"Dict merge: {dict_result['merged_data']}")

# ── 5. Conditional branching with PythonCodeNode ────────────────────
# For custom conditional logic beyond SwitchNode's operators, use
# PythonCodeNode with if/else in the code string.

builder = WorkflowBuilder()

builder.add_node(
    "PythonCodeNode",
    "categorize",
    {
        "code": (
            "if score >= 90:\n"
            "    output = 'excellent'\n"
            "elif score >= 70:\n"
            "    output = 'good'\n"
            "else:\n"
            "    output = 'needs_improvement'"
        ),
        "inputs": {"score": "int"},
        "output_type": "str",
    },
)

workflow = builder.build(name="python-conditional")

from kailash import LocalRuntime

with LocalRuntime() as rt:
    results, _ = rt.execute(workflow, parameters={"categorize": {"score": 85}})

categorize_out = results.get("categorize", {})
print(f"Python conditional: {categorize_out}")

# ── 6. SwitchNode output schema ─────────────────────────────────────
# Inspect what outputs SwitchNode declares.

output_schema = switch.get_output_schema()
assert "true_output" in output_schema, "Boolean mode: true_output"
assert "false_output" in output_schema, "Boolean mode: false_output"
assert "default" in output_schema, "Multi-case mode: default"
assert "condition_result" in output_schema, "Condition result is always available"
print(f"SwitchNode output fields: {list(output_schema.keys())}")

# ── 7. SwitchNode input parameters ──────────────────────────────────
# Inspect the full parameter set.

params = switch.get_parameters()
assert "input_data" in params
assert "condition_field" in params
assert "operator" in params
assert "value" in params
assert "cases" in params
assert "case_prefix" in params
print(f"SwitchNode parameters: {list(params.keys())}")

# ── 8. Edge case: missing condition field ───────────────────────────
# When condition_field is not present in the data, SwitchNode uses
# the input data directly as the value to check.

direct_result = switch.run(
    input_data=42,
    operator=">=",
    value=40,
)
assert direct_result["true_output"] == 42, "Uses input_data directly when no field"
print("Direct value comparison works")

# ── 9. Edge case: unknown operator ──────────────────────────────────
# An unknown operator returns False (routes to false_output).

unknown_op_result = switch.run(
    input_data={"val": 10},
    condition_field="val",
    operator="~=",
    value=10,
)
assert (
    unknown_op_result["false_output"] is not None
), "Unknown operator defaults to False"
print("Unknown operator handled gracefully")

print("PASS: 00-core/07_conditional_node")
