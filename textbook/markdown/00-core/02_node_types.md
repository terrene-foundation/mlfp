# Chapter 2: Built-in Node Types

## Overview

Kailash ships with a registry of built-in node types that cover common processing tasks -- from executing Python code to making API calls, transforming data, and managing control flow. This chapter teaches you how to discover available node types through the `NodeRegistry`, inspect their metadata and parameters, and use them in workflows. Understanding the node catalog is essential before you start building custom nodes.

## Prerequisites

- [Chapter 1: WorkflowBuilder Basics](01_workflow_builder.md) -- you should know how to create builders, add nodes, and build workflows
- Kailash SDK installed

## Concepts

### Concept 1: NodeRegistry

The `NodeRegistry` is a global dictionary that maps string names to `Node` classes. When you call `add_node("PythonCodeNode", ...)`, the builder looks up `"PythonCodeNode"` in the registry to find the corresponding class. Built-in nodes are registered automatically at import time via the `@register_node` decorator.

- **What**: A singleton registry mapping node type names (strings) to Node classes
- **Why**: Decouples workflow definitions from concrete node implementations -- you can reference nodes by name in config files, databases, or user interfaces without importing the class directly
- **How**: The `@register_node` decorator calls `NodeRegistry.register(name, cls)` at module load time. `NodeRegistry.get(name)` retrieves the class, and `NodeRegistry.list()` returns all registered names
- **When**: Use it to discover available nodes, verify a node type exists, or dynamically construct workflows

### Concept 2: NodeMetadata

Every node class carries metadata: a human-readable name, description, version, and a set of tags for categorization. This metadata is used by tooling (visual editors, documentation generators) and is accessible at runtime for introspection.

### Concept 3: NodeParameter

`NodeParameter` describes a single input or output of a node -- its name, type, whether it is required, a description, and an optional default value. The collection of input and output parameters defines the node's contract: what data it expects and what data it produces.

### Key API

| Method / Class            | Parameters                                                                | Returns                | Description                       |
| ------------------------- | ------------------------------------------------------------------------- | ---------------------- | --------------------------------- |
| `NodeRegistry.list()`     | --                                                                        | `list` or `dict`       | All registered node names         |
| `NodeRegistry.get()`      | `name: str`                                                               | `Node class` or `None` | Look up a node class by name      |
| `NodeRegistry.register()` | `name: str`, `cls: type`                                                  | `None`                 | Register a node class             |
| `NodeMetadata()`          | `name: str`, `description: str`, `version: str`, `tags: set`              | `NodeMetadata`         | Node metadata descriptor          |
| `NodeParameter()`         | `name: str`, `type: str`, `description: str`, `required: bool`, `default` | `NodeParameter`        | Input/output parameter descriptor |

## Code Walkthrough

```python
from __future__ import annotations

from kailash.nodes.base import Node, NodeMetadata, NodeParameter, NodeRegistry

# ── 1. NodeRegistry: discover available nodes ───────────────────────
# The registry is populated by @register_node decorators at import time.

registered_names = NodeRegistry.list()
assert isinstance(registered_names, (list, dict)), "list() returns registered nodes"
print(f"Registered node types: {len(registered_names)}")

# PythonCodeNode is always available (built-in)
python_code_cls = NodeRegistry.get("PythonCodeNode")
assert python_code_cls is not None, "PythonCodeNode must be registered"
assert issubclass(python_code_cls, Node), "Registered class extends Node"

# ── 2. NodeMetadata: inspect node information ───────────────────────
# Every Node class has metadata (name, description, version, tags).

metadata = NodeMetadata(
    name="ExampleNode", description="A test node", version="1.0.0"
)
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

from kailash import WorkflowBuilder, LocalRuntime

builder = WorkflowBuilder()

builder.add_node(
    "PythonCodeNode",
    "make_list",
    {
        "code": "output = [1, 2, 3, 4, 5]",
        "output_type": "list",
    },
)

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

# Execute and verify
runtime = LocalRuntime()
results, run_id = runtime.execute(workflow)
runtime.close()

assert run_id is not None, "execute() returns a run_id"
assert isinstance(results, dict), "Results are a dict keyed by node_id"

sum_result = results.get("sum_it", {})
print(f"Sum result: {sum_result}")
```

### Step-by-Step Explanation

1. **NodeRegistry.list()**: Returns all node types currently registered in the SDK. The count depends on which modules have been imported -- Kailash lazy-loads node modules.

2. **NodeRegistry.get()**: Retrieves the class for a given node type name. Returns `None` if the name is not registered. The returned class is always a subclass of `Node`.

3. **NodeMetadata**: A data object that describes a node for humans and tooling. The `tags` field defaults to an empty set and is useful for filtering nodes by category (e.g., `"transform"`, `"ai"`, `"data"`).

4. **NodeParameter**: Defines a single port on a node. `required=True` means the workflow must provide this input (via a connection or parameter override). `default` provides a fallback value for optional parameters.

5. **Workflow execution**: The two-node workflow generates a list `[1, 2, 3, 4, 5]` and pipes it to a node that sums it. The result is stored in the results dict under the node ID `"sum_it"`.

## Common Mistakes

| Mistake                             | Correct Pattern                                        | Why                                                                              |
| ----------------------------------- | ------------------------------------------------------ | -------------------------------------------------------------------------------- |
| Looking up a misspelled node name   | `NodeRegistry.get("PythonCodeNode")`                   | `get()` returns `None` for unknown names, leading to confusing errors downstream |
| Ignoring `required=True` parameters | Always connect or provide defaults for required inputs | Missing required inputs cause `NodeExecutionError` at runtime                    |
| Assuming all nodes are loaded       | Import the relevant module first                       | Nodes are registered at import time; unimported modules are not in the registry  |

## Exercises

1. Call `NodeRegistry.list()` and count how many node types are available. Try importing `kailash.nodes` submodules and check if the count increases.
2. Create a `NodeParameter` with `required=False` and no `default`. What value does the node receive when no input is connected?
3. Build a workflow that uses `PythonCodeNode` to generate a dictionary and another node to extract a specific key from it.

## Key Takeaways

- `NodeRegistry` is the global catalog of all available node types
- Built-in nodes are registered automatically via `@register_node` at import time
- `NodeMetadata` describes a node (name, description, version, tags) for introspection
- `NodeParameter` defines the input/output contract: name, type, required flag, and default
- `PythonCodeNode` is the most versatile built-in node -- it executes arbitrary Python code strings
- Results from `runtime.execute()` are returned as a `(dict, run_id)` tuple

## Next Chapter

[Chapter 3: Custom Nodes](03_custom_nodes.md) -- Learn how to create your own node types by subclassing `Node` and registering them with the `NodeRegistry`.
