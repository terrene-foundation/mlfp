# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Core / Value Types in Workflows
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Understand how Python values map to Kailash's internal type
#            system and how types are preserved through workflow execution
# LEVEL: Intermediate
# PARITY: Equivalent — DIV-001: Python uses float for all numbers, Rust
#         distinguishes i64/f64; DIV-006: Rust has Value::Bytes, Python
#         uses bytes natively
# VALIDATES: Value type handling in workflow execution — int, float, str,
#            bool, list, dict, None flowing through PythonCodeNode
#
# Run: uv run python textbook/python/00-core/08_value_types.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from kailash import LocalRuntime, WorkflowBuilder

# ── 1. Integer values ──────────────────────────────────────────────
# Integers pass through PythonCodeNode and are preserved as int.

builder = WorkflowBuilder()
builder.add_node(
    "PythonCodeNode",
    "int_source",
    {
        "code": "output = 42",
        "output_type": "int",
    },
)
builder.add_node(
    "PythonCodeNode",
    "int_check",
    {
        "code": "output = {'value': n, 'type_name': type(n).__name__}",
        "inputs": {"n": "int"},
        "output_type": "dict",
    },
)
builder.connect("int_source", "int_check", mapping={"output": "n"})

workflow = builder.build(name="int-types")

with LocalRuntime() as rt:
    results, _ = rt.execute(workflow)

int_out = results.get("int_check", {})
print(f"Integer: {int_out}")

# ── 2. Float values ────────────────────────────────────────────────
# Floats are preserved. DIV-001: In Rust, all numbers may be f64.
# In Python, int and float remain distinct types.

builder2 = WorkflowBuilder()
builder2.add_node(
    "PythonCodeNode",
    "float_source",
    {
        "code": "output = 3.14159",
        "output_type": "float",
    },
)
builder2.add_node(
    "PythonCodeNode",
    "float_check",
    {
        "code": "output = {'value': n, 'type_name': type(n).__name__}",
        "inputs": {"n": "float"},
        "output_type": "dict",
    },
)
builder2.connect("float_source", "float_check", mapping={"output": "n"})

workflow2 = builder2.build(name="float-types")

with LocalRuntime() as rt:
    results2, _ = rt.execute(workflow2)

float_out = results2.get("float_check", {})
print(f"Float: {float_out}")

# ── 3. String values ───────────────────────────────────────────────
# Strings pass through unchanged, including Unicode content.

builder3 = WorkflowBuilder()
builder3.add_node(
    "PythonCodeNode",
    "str_source",
    {
        "code": "output = 'Kailash SDK'",
        "output_type": "str",
    },
)
builder3.add_node(
    "PythonCodeNode",
    "str_transform",
    {
        "code": "output = {'original': s, 'length': len(s), 'upper': s.upper()}",
        "inputs": {"s": "str"},
        "output_type": "dict",
    },
)
builder3.connect("str_source", "str_transform", mapping={"output": "s"})

workflow3 = builder3.build(name="str-types")

with LocalRuntime() as rt:
    results3, _ = rt.execute(workflow3)

str_out = results3.get("str_transform", {})
print(f"String: {str_out}")

# ── 4. Boolean values ──────────────────────────────────────────────
# Booleans are preserved as Python bool (True/False), not coerced
# to integers.

builder4 = WorkflowBuilder()
builder4.add_node(
    "PythonCodeNode",
    "bool_source",
    {
        "code": "output = True",
        "output_type": "bool",
    },
)
builder4.add_node(
    "PythonCodeNode",
    "bool_check",
    {
        "code": "output = {'value': flag, 'type_name': type(flag).__name__, 'negated': not flag}",
        "inputs": {"flag": "bool"},
        "output_type": "dict",
    },
)
builder4.connect("bool_source", "bool_check", mapping={"output": "flag"})

workflow4 = builder4.build(name="bool-types")

with LocalRuntime() as rt:
    results4, _ = rt.execute(workflow4)

bool_out = results4.get("bool_check", {})
print(f"Boolean: {bool_out}")

# ── 5. List values ─────────────────────────────────────────────────
# Lists of any element type flow through workflows. Lists can contain
# mixed types (though homogeneous lists are recommended).

builder5 = WorkflowBuilder()
builder5.add_node(
    "PythonCodeNode",
    "list_source",
    {
        "code": "output = [1, 'two', 3.0, True, None]",
        "output_type": "list",
    },
)
builder5.add_node(
    "PythonCodeNode",
    "list_check",
    {
        "code": (
            "output = {"
            "'length': len(items), "
            "'types': [type(x).__name__ for x in items]"
            "}"
        ),
        "inputs": {"items": "list"},
        "output_type": "dict",
    },
)
builder5.connect("list_source", "list_check", mapping={"output": "items"})

workflow5 = builder5.build(name="list-types")

with LocalRuntime() as rt:
    results5, _ = rt.execute(workflow5)

list_out = results5.get("list_check", {})
print(f"List: {list_out}")

# ── 6. Dict values ─────────────────────────────────────────────────
# Dicts are the primary structured data type in Kailash workflows.
# Node results are always dicts keyed by output name.

builder6 = WorkflowBuilder()
builder6.add_node(
    "PythonCodeNode",
    "dict_source",
    {
        "code": "output = {'name': 'Alice', 'scores': [95, 87, 92], 'active': True}",
        "output_type": "dict",
    },
)
builder6.add_node(
    "PythonCodeNode",
    "dict_check",
    {
        "code": (
            "output = {"
            "'keys': list(data.keys()), "
            "'name': data['name'], "
            "'avg_score': sum(data['scores']) / len(data['scores'])"
            "}"
        ),
        "inputs": {"data": "dict"},
        "output_type": "dict",
    },
)
builder6.connect("dict_source", "dict_check", mapping={"output": "data"})

workflow6 = builder6.build(name="dict-types")

with LocalRuntime() as rt:
    results6, _ = rt.execute(workflow6)

dict_out = results6.get("dict_check", {})
print(f"Dict: {dict_out}")

# ── 7. None values ─────────────────────────────────────────────────
# None is a valid value that flows through workflows. Nodes must
# handle None inputs gracefully.

builder7 = WorkflowBuilder()
builder7.add_node(
    "PythonCodeNode",
    "none_source",
    {
        "code": "output = None",
        "output_type": "None",
    },
)
builder7.add_node(
    "PythonCodeNode",
    "none_check",
    {
        "code": "output = {'is_none': val is None, 'type_name': type(val).__name__}",
        "inputs": {"val": "None"},
        "output_type": "dict",
    },
)
builder7.connect("none_source", "none_check", mapping={"output": "val"})

workflow7 = builder7.build(name="none-types")

with LocalRuntime() as rt:
    results7, _ = rt.execute(workflow7)

none_out = results7.get("none_check", {})
print(f"None: {none_out}")

# ── 8. Nested structures ──────────────────────────────────────────
# Complex nested structures (list of dicts, dict of lists, etc.)
# pass through workflows without flattening or transformation.

builder8 = WorkflowBuilder()
builder8.add_node(
    "PythonCodeNode",
    "nested_source",
    {
        "code": (
            "output = {"
            "'users': ["
            "{'name': 'Alice', 'tags': ['admin', 'active']}, "
            "{'name': 'Bob', 'tags': ['user']}"
            "], "
            "'metadata': {'version': 1, 'counts': [2, 0]}"
            "}"
        ),
        "output_type": "dict",
    },
)
builder8.add_node(
    "PythonCodeNode",
    "nested_check",
    {
        "code": (
            "user_count = len(data['users'])\n"
            "all_tags = []\n"
            "for u in data['users']:\n"
            "    all_tags.extend(u['tags'])\n"
            "output = {'user_count': user_count, 'all_tags': all_tags, "
            "'meta_version': data['metadata']['version']}"
        ),
        "inputs": {"data": "dict"},
        "output_type": "dict",
    },
)
builder8.connect("nested_source", "nested_check", mapping={"output": "data"})

workflow8 = builder8.build(name="nested-types")

with LocalRuntime() as rt:
    results8, _ = rt.execute(workflow8)

nested_out = results8.get("nested_check", {})
print(f"Nested: {nested_out}")

# ── 9. Type coercion between nodes ────────────────────────────────
# Kailash does not perform automatic type coercion between nodes.
# If a node outputs an int and the next node expects a string,
# the code must handle the conversion explicitly.

builder9 = WorkflowBuilder()
builder9.add_node(
    "PythonCodeNode",
    "int_producer",
    {
        "code": "output = 42",
        "output_type": "int",
    },
)
builder9.add_node(
    "PythonCodeNode",
    "str_consumer",
    {
        "code": "output = f'The answer is {str(val)}'",
        "inputs": {"val": "int"},
        "output_type": "str",
    },
)
builder9.connect("int_producer", "str_consumer", mapping={"output": "val"})

workflow9 = builder9.build(name="coercion-demo")

with LocalRuntime() as rt:
    results9, _ = rt.execute(workflow9)

coercion_out = results9.get("str_consumer", {})
print(f"Type coercion: {coercion_out}")

# ── 10. Result structure recap ─────────────────────────────────────
# Every workflow execution returns results as:
#   {node_id: {output_name: value, ...}, ...}
# The outer dict is keyed by node_id. The inner dict is keyed by
# output variable names. This structure is consistent regardless
# of the value types flowing through the workflow.

with LocalRuntime() as rt:
    all_results, run_id = rt.execute(workflow)

assert isinstance(all_results, dict), "Top-level results is always a dict"
for node_id, node_outputs in all_results.items():
    assert isinstance(node_id, str), "Node IDs are always strings"
    print(f"  Node '{node_id}': {node_outputs}")

print("PASS: 00-core/08_value_types")
