# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Core / LocalRuntime Execution
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Execute workflows with LocalRuntime, inspect results, and
#            use parameter overrides
# LEVEL: Basic
# PARITY: Equivalent (DIV-005: Python has separate sync/async runtimes;
#         Rust has single Runtime with sync wrapper)
#         (DIV-008: Rust ExecutionResult has metadata; Python returns tuple)
# VALIDATES: LocalRuntime(), execute(), context manager, parameters
#
# Run: uv run python textbook/python/00-core/05_local_runtime.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from kailash import WorkflowBuilder, LocalRuntime

# ── Helper: build a simple workflow ─────────────────────────────────


def build_math_workflow() -> "Workflow":
    """Two-node workflow: generate a number, then double it."""
    builder = WorkflowBuilder()
    builder.add_node(
        "PythonCodeNode",
        "generate",
        {"code": "output = 21", "output_type": "int"},
    )
    builder.add_node(
        "PythonCodeNode",
        "double",
        {
            "code": "output = value * 2",
            "inputs": {"value": "int"},
            "output_type": "int",
        },
    )
    builder.connect("generate", "double", mapping={"output": "value"})
    return builder.build(name="math-demo")


# ── 1. Basic execution ──────────────────────────────────────────────
# LocalRuntime.execute() returns (results_dict, run_id).
# Results are keyed by node_id.

workflow = build_math_workflow()

runtime = LocalRuntime()
results, run_id = runtime.execute(workflow)
runtime.close()

assert isinstance(results, dict), "Results is a dict"
assert run_id is not None, "run_id is always returned"
assert "generate" in results, "Each node produces output"
assert "double" in results

print(f"Run ID: {run_id}")
print(f"Generate: {results['generate']}")
print(f"Double: {results['double']}")

# ── 2. Context manager pattern (recommended) ────────────────────────
# Ensures proper cleanup of the runtime's event loop and resources.

with LocalRuntime() as rt:
    results2, run_id2 = rt.execute(workflow)

assert results2 is not None
assert run_id2 != run_id, "Each execution gets a unique run_id"

# ── 3. Parameter overrides ──────────────────────────────────────────
# Pass parameters={node_id: {param: value}} to override node inputs
# at execution time without rebuilding the workflow.

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
wf_greet = builder.build(name="param-demo")

with LocalRuntime() as rt:
    results3, _ = rt.execute(wf_greet, parameters={"greet": {"name": "Kailash"}})

greet_output = results3.get("greet", {})
print(f"Greeting: {greet_output}")

# ── 4. Multiple executions on same runtime ──────────────────────────
# A single runtime instance can execute multiple workflows.
# Connection pools and async resources are reused.

with LocalRuntime() as rt:
    r1, id1 = rt.execute(workflow)
    r2, id2 = rt.execute(workflow)

assert id1 != id2, "Each execution gets a unique run_id"

# ── 5. Return structure ─────────────────────────────────────────────
# DIV-008: In Rust, execute() returns ExecutionResult { results, run_id,
# metadata } with nodes_executed, levels_executed, duration. Python
# returns a simpler (dict, str) tuple.

assert isinstance(results, dict)
assert isinstance(run_id, (str, type(None)))

# ── 6. Edge case: empty workflow ────────────────────────────────────

empty_wf = WorkflowBuilder().build(name="empty")
with LocalRuntime() as rt:
    empty_results, empty_id = rt.execute(empty_wf)

assert isinstance(empty_results, dict)
assert len(empty_results) == 0, "Empty workflow produces no results"

print("PASS: 00-core/05_local_runtime")
