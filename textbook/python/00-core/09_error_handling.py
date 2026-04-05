# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Core / Error Handling
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Handle workflow errors — build errors, execution errors,
#            node errors, and the full exception hierarchy
# LEVEL: Advanced
# PARITY: Equivalent — DIV-004: Python uses exceptions, Rust uses typed
#         Result<T, E> errors with KailashError enum variants
# VALIDATES: WorkflowValidationError, WorkflowExecutionError,
#            NodeExecutionError, NodeConfigurationError, ConnectionError,
#            SafetyViolationError, KailashException hierarchy
#
# Run: uv run python textbook/python/00-core/09_error_handling.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from kailash import LocalRuntime, WorkflowBuilder
from kailash.sdk_exceptions import (
    CodeExecutionError,
    ConnectionError,
    KailashException,
    NodeConfigurationError,
    NodeException,
    NodeExecutionError,
    NodeValidationError,
    RuntimeExecutionError,
    SafetyViolationError,
    WorkflowException,
    WorkflowExecutionError,
    WorkflowValidationError,
)

# ── 1. Exception hierarchy ─────────────────────────────────────────
# All Kailash exceptions inherit from KailashException, enabling
# catch-all handling at the top level while allowing specific catches
# for known error types.

assert issubclass(WorkflowValidationError, WorkflowException)
assert issubclass(WorkflowExecutionError, WorkflowException)
assert issubclass(WorkflowException, KailashException)

assert issubclass(NodeExecutionError, NodeException)
assert issubclass(NodeValidationError, NodeException)
assert issubclass(NodeConfigurationError, NodeException)
assert issubclass(SafetyViolationError, NodeException)
assert issubclass(CodeExecutionError, NodeException)
assert issubclass(NodeException, KailashException)

assert issubclass(ConnectionError, WorkflowException)
assert issubclass(RuntimeExecutionError, KailashException)

print("Exception hierarchy validated")

# ── 2. Build errors — WorkflowValidationError ──────────────────────
# WorkflowValidationError is raised when the workflow graph is invalid.
# Common causes: duplicate node IDs, connecting non-existent nodes.

# Duplicate node ID
builder = WorkflowBuilder()
builder.add_node(
    "PythonCodeNode", "my_node", {"code": "output = 1", "output_type": "int"}
)

try:
    builder.add_node(
        "PythonCodeNode", "my_node", {"code": "output = 2", "output_type": "int"}
    )
    assert False, "Should have raised WorkflowValidationError"
except WorkflowValidationError as e:
    assert "my_node" in str(e), "Error message includes the duplicate ID"
    print(f"Duplicate node ID caught: {e}")

# Connecting non-existent source node
builder2 = WorkflowBuilder()
builder2.add_node(
    "PythonCodeNode", "real_node", {"code": "output = 1", "output_type": "int"}
)

try:
    builder2.connect("ghost_node", "real_node")
    assert False, "Should have raised WorkflowValidationError"
except WorkflowValidationError as e:
    print(f"Non-existent source caught: {e}")

# Connecting non-existent target node
try:
    builder2.connect("real_node", "ghost_target")
    assert False, "Should have raised WorkflowValidationError"
except WorkflowValidationError as e:
    print(f"Non-existent target caught: {e}")

# ── 3. Connection errors — ConnectionError ─────────────────────────
# ConnectionError is raised for invalid connection patterns like
# self-connections and duplicate connections.

builder3 = WorkflowBuilder()
builder3.add_node("PythonCodeNode", "a", {"code": "output = 1", "output_type": "int"})
builder3.add_node(
    "PythonCodeNode",
    "b",
    {"code": "output = data", "inputs": {"data": "int"}, "output_type": "int"},
)

# Self-connection
try:
    builder3.connect("a", "a")
    assert False, "Self-connection should raise ConnectionError"
except ConnectionError:
    print("Self-connection caught")

# Duplicate connection
builder3.connect("a", "b")
try:
    builder3.connect("a", "b")
    assert False, "Duplicate connection should raise ConnectionError"
except ConnectionError:
    print("Duplicate connection caught")

# ── 4. Node configuration errors — NodeConfigurationError ──────────
# NodeConfigurationError is raised when creating a node with invalid
# configuration parameters.

from kailash.nodes.code.python import PythonCodeNode

# No code, function, or class provided
try:
    node = PythonCodeNode(name="broken")
    assert False, "Should have raised NodeConfigurationError"
except NodeConfigurationError as e:
    assert "code" in str(e).lower() or "function" in str(e).lower()
    print(f"Missing code/function caught: {e}")

# Both code and function provided (ambiguous)
try:
    node = PythonCodeNode(
        name="ambiguous",
        code="output = 1",
        function=lambda x: x,
    )
    assert False, "Should have raised NodeConfigurationError"
except NodeConfigurationError as e:
    print(f"Ambiguous config caught: {e}")

# ── 5. Catch-all pattern with KailashException ─────────────────────
# For production code, catch KailashException at the top level to
# ensure no SDK error goes unhandled.

errors_caught = []

for scenario in ["valid", "duplicate_node", "bad_config"]:
    try:
        b = WorkflowBuilder()
        if scenario == "valid":
            b.add_node(
                "PythonCodeNode",
                "ok",
                {"code": "output = 'works'", "output_type": "str"},
            )
            wf = b.build(name="ok")
            with LocalRuntime() as rt:
                rt.execute(wf)
        elif scenario == "duplicate_node":
            b.add_node("PythonCodeNode", "dup", {"code": "output = 1"})
            b.add_node("PythonCodeNode", "dup", {"code": "output = 2"})
        elif scenario == "bad_config":
            PythonCodeNode(name="no_code_at_all")
    except KailashException as e:
        errors_caught.append((scenario, type(e).__name__, str(e)))

assert len(errors_caught) == 2, "Two scenarios should produce errors"
assert errors_caught[0][0] == "duplicate_node"
assert errors_caught[1][0] == "bad_config"
print(f"Catch-all pattern caught {len(errors_caught)} errors:")
for scenario, error_type, msg in errors_caught:
    print(f"  {scenario}: {error_type}")

# ── 6. Granular error handling pattern ─────────────────────────────
# For detailed diagnostics, catch specific exceptions first, then
# fall back to the base class. Python's exception matching is
# order-dependent — put specific catches before general ones.


def safe_build_and_run(builder: WorkflowBuilder, name: str) -> dict | str:
    """Build and run a workflow with comprehensive error handling."""
    try:
        workflow = builder.build(name=name)
        with LocalRuntime() as rt:
            results, run_id = rt.execute(workflow)
        return results
    except WorkflowValidationError as e:
        return f"BUILD_ERROR: {e}"
    except NodeExecutionError as e:
        return f"NODE_ERROR: {e}"
    except WorkflowExecutionError as e:
        return f"EXEC_ERROR: {e}"
    except KailashException as e:
        return f"SDK_ERROR ({type(e).__name__}): {e}"


# Test with a valid workflow
valid_builder = WorkflowBuilder()
valid_builder.add_node(
    "PythonCodeNode",
    "hello",
    {"code": "output = 'hello'", "output_type": "str"},
)
result = safe_build_and_run(valid_builder, "safe-test")
assert isinstance(result, dict), "Valid workflow returns results dict"
print(f"Safe build succeeded: {result}")

# ── 7. Legacy compatibility names ──────────────────────────────────
# The SDK provides legacy aliases for backward compatibility.
# These are re-exports, not separate classes.

from kailash.sdk_exceptions import KailashRuntimeError, KailashValidationError

assert KailashRuntimeError is RuntimeExecutionError, "Legacy alias"
assert KailashValidationError is NodeValidationError, "Legacy alias"
print("Legacy aliases confirmed")

# ── 8. Specialized exception attributes ────────────────────────────
# Some exceptions carry extra context. RetryExhaustedException, for
# example, includes the operation name, attempt count, and last error.

from kailash.sdk_exceptions import RetryExhaustedException

original_error = TimeoutError("Connection timed out")
retry_error = RetryExhaustedException(
    operation="database_connect",
    attempts=3,
    last_error=original_error,
    total_wait_time=4.5,
)

assert retry_error.operation == "database_connect"
assert retry_error.attempts == 3
assert retry_error.last_error is original_error
assert retry_error.total_wait_time == 4.5
assert "3 retry attempts" in str(retry_error)
assert "4.50s" in str(retry_error)
print(f"RetryExhaustedException: {retry_error}")

# ── 9. Best practices summary ─────────────────────────────────────
# 1. Use specific exception types for expected error conditions
# 2. Use KailashException as a catch-all at the top level
# 3. Never catch bare Exception in production — it hides non-SDK bugs
# 4. Check error messages — they include actionable suggestions
# 5. WorkflowValidationError fires at build time, not execution time
# 6. NodeExecutionError fires during runtime node processing
# 7. ConnectionError fires when wiring nodes incorrectly
# 8. SafetyViolationError fires when code uses blocked operations

print("PASS: 00-core/09_error_handling")
