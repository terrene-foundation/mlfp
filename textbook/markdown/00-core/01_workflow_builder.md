# Chapter 1: WorkflowBuilder Basics

## Overview

Every Kailash application begins with a **workflow** -- a directed graph of processing nodes wired together to form a data pipeline. The `WorkflowBuilder` is your entry point: it provides a fluent API for adding nodes, connecting them, and compiling the graph into an immutable `Workflow` object ready for execution. This chapter teaches you how to create, wire, validate, and inspect workflows from scratch.

## Prerequisites

- Python 3.10+ installed
- Kailash SDK installed (`pip install kailash`)
- No prior Kailash knowledge required -- this is where you start

## Concepts

### Concept 1: The Builder Pattern

WorkflowBuilder uses the **builder pattern** -- you accumulate configuration (nodes, connections) through method calls, then finalize everything with a single `.build()` call. This separates construction from validation: you can add nodes in any order, wire them up, and the builder only checks the graph's validity when you call `build()`.

- **What**: A mutable container that collects nodes and connections before producing an immutable workflow
- **Why**: Separating construction from validation lets you build workflows incrementally without worrying about ordering constraints
- **How**: Internally, the builder maintains two lists -- `nodes` and `connections` -- that are validated and frozen into a `Workflow` object on `build()`
- **When**: Use WorkflowBuilder whenever you need a multi-step processing pipeline. For simple single-record CRUD, use DataFlow Express instead

### Concept 2: Nodes

Nodes are the processing units in a workflow. Each node has a **type** (what it does), a **node ID** (a unique string identifier), and a **configuration dictionary** (parameters that control its behavior). The simplest built-in node type is `PythonCodeNode`, which executes a Python code string.

### Concept 3: Connections

Connections define how data flows between nodes. A connection maps an **output port** of one node to an **input port** of another. The default port name is `"data"`, but you can use explicit port names via the `mapping` parameter or the `from_output`/`to_input` keyword arguments.

### Concept 4: The Workflow Object

The `Workflow` returned by `build()` is **immutable** -- it cannot be modified after creation. It contains the validated graph (node instances and connections) plus optional metadata (name, description, version). Immutability guarantees that a workflow can be executed repeatedly with consistent behavior.

### Key API

| Method                | Parameters                                                                                              | Returns             | Description                          |
| --------------------- | ------------------------------------------------------------------------------------------------------- | ------------------- | ------------------------------------ |
| `WorkflowBuilder()`   | --                                                                                                      | `WorkflowBuilder`   | Create an empty builder              |
| `add_node()`          | `node_type: str`, `node_id: str`, `config: dict`                                                        | `str` (the node_id) | Add a node to the builder            |
| `connect()`           | `source: str`, `target: str`, `mapping: dict = None`, `from_output: str = None`, `to_input: str = None` | `None`              | Wire an output port to an input port |
| `build()`             | `name: str = None`, `description: str = None`, `version: str = None`, `workflow_id: str = None`         | `Workflow`          | Validate and compile the graph       |
| `builder.nodes`       | --                                                                                                      | `list`              | Current list of added nodes          |
| `builder.connections` | --                                                                                                      | `list`              | Current list of connections          |

## Code Walkthrough

```python
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
#   connect("src", "dst")                    -> default "data" -> "data"
#   connect("src", "dst", mapping={"a": "b"})-> explicit port mapping
#   connect("src", "dst", from_output="a", to_input="b") -> named ports

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
```

### Step-by-Step Explanation

1. **Import**: `WorkflowBuilder` is the primary import from `kailash`. The `Workflow` class lives in `kailash.workflow.graph` and is what `build()` returns.

2. **Empty builder**: A fresh `WorkflowBuilder()` has zero nodes and zero connections. This is your blank canvas.

3. **Adding nodes**: `add_node("PythonCodeNode", "generate", {...})` adds a node of type `PythonCodeNode` with the ID `"generate"`. The config dict sets the code to execute and declares the output type. The method returns the node ID string, which is convenient for storing references.

4. **Connecting nodes**: `connect("generate", "transform", mapping={"output": "text"})` wires the `"output"` port of `"generate"` to the `"text"` input port of `"transform"`. The mapping dict lets you route named ports explicitly.

5. **Building**: `build()` validates the entire graph -- checking for dangling references, type mismatches, and structural issues -- then returns an immutable `Workflow`. You can attach metadata like `name`, `description`, and `version`.

6. **Inspection**: The `Workflow` object exposes `node_instances` (a dict of node ID to node) and `connections` (a list of connection descriptors).

7. **Duplicate IDs**: Adding a second node with the same ID raises `WorkflowValidationError` immediately -- the builder catches this at add-time, not build-time.

8. **Non-existent connections**: Connecting to a node that does not exist also raises `WorkflowValidationError` at connect-time.

9. **Empty workflows**: An empty workflow is valid -- `build()` succeeds and returns a workflow with zero nodes.

## Common Mistakes

| Mistake                                      | Correct Pattern                                 | Why                                                                                                                       |
| -------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `runtime.execute(builder)`                   | `runtime.execute(builder.build())`              | The runtime expects a `Workflow`, not a `WorkflowBuilder`. Missing `.build()` skips validation and causes cryptic errors. |
| Using variable names for node IDs            | `add_node("PythonCodeNode", "my_node", {...})`  | Dynamic node IDs break graph analysis, checkpoint recovery, and debugging. Always use string literals.                    |
| Forgetting `mapping` when ports do not match | `connect("a", "b", mapping={"output": "text"})` | Without `mapping`, the default ports `"data"` to `"data"` are used, which may not match your node's actual port names.    |
| Modifying the workflow after `build()`       | Build a new workflow from a new builder         | The `Workflow` object is immutable. Create a fresh `WorkflowBuilder` if you need a different graph.                       |

## Exercises

1. Create a three-node workflow: one that generates a list of numbers, one that filters for even numbers, and one that sums them. Use `PythonCodeNode` for all three.
2. What happens if you call `build()` twice on the same builder? Does it return the same `Workflow` instance or a new one?
3. Try connecting two nodes with `connect("a", "b")` (no mapping) when the source node's output is named `"output"` instead of `"data"`. What error do you get at execution time?

## Key Takeaways

- `WorkflowBuilder` is the entry point for all workflow construction in Kailash
- The builder pattern separates construction from validation -- `build()` is where validation happens
- `add_node()` takes a type string, a node ID string, and a config dict
- `connect()` wires output ports to input ports, with `mapping` for explicit routing
- The resulting `Workflow` is immutable and can be executed repeatedly
- Duplicate node IDs and references to non-existent nodes are caught immediately

## Next Chapter

[Chapter 2: Node Types](02_node_types.md) -- Discover the built-in node types available through the `NodeRegistry`, and learn how to inspect node metadata and parameters.
