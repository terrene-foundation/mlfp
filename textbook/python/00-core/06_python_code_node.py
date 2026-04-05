# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Core / PythonCodeNode Deep Dive
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Use PythonCodeNode for custom logic — code strings, inputs,
#            output types, from_function, and from_class patterns
# LEVEL: Intermediate
# PARITY: Python-only — DIV-009: Rust uses WASM + native cdylib for
#         custom code execution instead of Python code strings
# VALIDATES: PythonCodeNode configuration: "code", "inputs",
#            "output_type" keys; from_function(); from_class()
#
# Run: uv run python textbook/python/00-core/06_python_code_node.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from kailash import LocalRuntime, WorkflowBuilder
from kailash.nodes.code.python import PythonCodeNode

# ── 1. Simple code string — no inputs ──────────────────────────────
# The most basic pattern: a code string that sets the "output" variable.
# Config keys: "code" (str), "output_type" (str)

builder = WorkflowBuilder()
builder.add_node(
    "PythonCodeNode",
    "constant",
    {
        "code": "output = 42",
        "output_type": "int",
    },
)

workflow = builder.build(name="simple-code")

with LocalRuntime() as rt:
    results, run_id = rt.execute(workflow)

constant_out = results.get("constant", {})
print(f"Simple code output: {constant_out}")
assert run_id is not None, "Execution must produce a run_id"

# ── 2. Code with inputs — the "inputs" config key ──────────────────
# Use "inputs" to declare named input variables the code expects.
# Input variables are injected directly into the execution namespace.

builder2 = WorkflowBuilder()
builder2.add_node(
    "PythonCodeNode",
    "source",
    {
        "code": "output = 'kailash sdk'",
        "output_type": "str",
    },
)
builder2.add_node(
    "PythonCodeNode",
    "shout",
    {
        "code": "output = text.upper() + '!'",
        "inputs": {"text": "str"},
        "output_type": "str",
    },
)
builder2.connect("source", "shout", mapping={"output": "text"})

workflow2 = builder2.build(name="code-with-inputs")

with LocalRuntime() as rt:
    results2, _ = rt.execute(workflow2)

shout_out = results2.get("shout", {})
print(f"Code with inputs: {shout_out}")

# ── 3. Multiple outputs — exporting several variables ──────────────
# When you set multiple variables (not just "output"), all become
# available as named outputs for downstream connections.

builder3 = WorkflowBuilder()
builder3.add_node(
    "PythonCodeNode",
    "splitter",
    {
        "code": "first_half = text[:len(text)//2]\nsecond_half = text[len(text)//2:]",
        "inputs": {"text": "str"},
        "output_type": "dict",
    },
)
builder3.add_node(
    "PythonCodeNode",
    "source3",
    {
        "code": "output = 'HelloWorld'",
        "output_type": "str",
    },
)
builder3.connect("source3", "splitter", mapping={"output": "text"})

workflow3 = builder3.build(name="multi-output")

with LocalRuntime() as rt:
    results3, _ = rt.execute(workflow3)

splitter_out = results3.get("splitter", {})
print(f"Splitter outputs: {splitter_out}")

# ── 4. Code importing stdlib modules ───────────────────────────────
# PythonCodeNode allows importing from an allowlist of safe modules
# (math, json, datetime, collections, itertools, etc.).

builder4 = WorkflowBuilder()
builder4.add_node(
    "PythonCodeNode",
    "math_node",
    {
        "code": "import math\noutput = math.sqrt(144) + math.pi",
        "output_type": "float",
    },
)

workflow4 = builder4.build(name="stdlib-import")

with LocalRuntime() as rt:
    results4, _ = rt.execute(workflow4)

math_out = results4.get("math_node", {})
print(f"Math node output: {math_out}")

# ── 5. Chaining multiple PythonCodeNodes ───────────────────────────
# A three-node pipeline: generate list -> filter -> sum.

builder5 = WorkflowBuilder()
builder5.add_node(
    "PythonCodeNode",
    "gen_list",
    {
        "code": "output = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
        "output_type": "list",
    },
)
builder5.add_node(
    "PythonCodeNode",
    "filter_evens",
    {
        "code": "output = [x for x in numbers if x % 2 == 0]",
        "inputs": {"numbers": "list"},
        "output_type": "list",
    },
)
builder5.add_node(
    "PythonCodeNode",
    "sum_them",
    {
        "code": "output = sum(values)",
        "inputs": {"values": "list"},
        "output_type": "int",
    },
)
builder5.connect("gen_list", "filter_evens", mapping={"output": "numbers"})
builder5.connect("filter_evens", "sum_them", mapping={"output": "values"})

workflow5 = builder5.build(name="chain-demo")

with LocalRuntime() as rt:
    results5, _ = rt.execute(workflow5)

sum_out = results5.get("sum_them", {})
print(f"Chained pipeline result: {sum_out}")

# ── 6. Direct instantiation — PythonCodeNode constructor ───────────
# Beyond the config-dict pattern, you can instantiate PythonCodeNode
# directly with typed parameters and add the instance to a workflow.

direct_node = PythonCodeNode(
    name="direct_greet",
    code="result = f'Hello, {name}!'",
    input_types={"name": str},
    output_type=str,
)

assert direct_node.code == "result = f'Hello, {name}!'"
assert direct_node.output_type is str
assert "name" in direct_node.input_types

# Use the instance in a workflow
builder6 = WorkflowBuilder()
builder6.add_node(direct_node, "greeter")

workflow6 = builder6.build(name="direct-instance")
assert "greeter" in workflow6.node_instances

# ── 7. from_function — wrap a Python function as a node ────────────
# PythonCodeNode.from_function() introspects the function signature
# to build input_types and output_type automatically.


def fahrenheit_to_celsius(temp_f: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (temp_f - 32) * 5 / 9


fn_node = PythonCodeNode.from_function(fahrenheit_to_celsius)

assert fn_node.function is fahrenheit_to_celsius
assert "temp_f" in fn_node.input_types

# from_function nodes work in workflows like any other node
builder7 = WorkflowBuilder()
builder7.add_node(fn_node, "converter")

workflow7 = builder7.build(name="from-function-demo")
assert "converter" in workflow7.node_instances

# ── 8. from_class — wrap a stateful class as a node ────────────────
# Classes maintain state across invocations within the same workflow run.


class RunningTotal:
    """Accumulates values across calls."""

    def __init__(self):
        self.total = 0

    def process(self, value: float) -> dict:
        self.total += value
        return {"total": self.total, "last_value": value}


cls_node = PythonCodeNode.from_class(RunningTotal)

assert cls_node.class_type is RunningTotal
assert cls_node.instance is not None, "Class is pre-instantiated"

# ── 9. Edge case: code with no output variable ─────────────────────
# If the code does not set "output" or "result", the runtime captures
# all local variables as outputs.

builder9 = WorkflowBuilder()
builder9.add_node(
    "PythonCodeNode",
    "no_output_var",
    {
        "code": "x = 10\ny = 20\ntotal = x + y",
        "output_type": "dict",
    },
)

workflow9 = builder9.build(name="no-output-var")

with LocalRuntime() as rt:
    results9, _ = rt.execute(workflow9)

no_output = results9.get("no_output_var", {})
print(f"No explicit output variable: {no_output}")

print("PASS: 00-core/06_python_code_node")
