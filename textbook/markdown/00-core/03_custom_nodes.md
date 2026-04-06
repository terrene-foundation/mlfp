# Chapter 3: Custom Nodes

## Overview

While built-in nodes cover many common tasks, real-world workflows often need domain-specific processing logic. This chapter teaches you how to create custom nodes by subclassing `Node`, defining input and output parameters, implementing the `process()` method, and registering your node with the `NodeRegistry` so it can be referenced by name in workflows. You will also learn the async node pattern and how to wrap existing Python functions and classes as nodes.

## Prerequisites

- [Chapter 1: WorkflowBuilder Basics](01_workflow_builder.md)
- [Chapter 2: Node Types](02_node_types.md) -- understanding of `NodeMetadata`, `NodeParameter`, and `NodeRegistry`

## Concepts

### Concept 1: Node Subclassing

Every custom node inherits from the `Node` base class. You define three things on your subclass:

1. **`metadata`** -- a `NodeMetadata` instance with the node's name, description, version, and optional tags
2. **`input_parameters`** -- a dict mapping parameter names to `NodeParameter` instances
3. **`output_parameters`** -- a dict mapping output names to `NodeParameter` instances
4. **`process()`** -- a method that receives inputs as keyword arguments and returns a dict of outputs

- **What**: A Python class that extends `Node` with custom processing logic
- **Why**: Custom nodes encapsulate domain logic into reusable, testable units that can be wired into any workflow
- **How**: The runtime calls `process(**inputs)` (or `process_async()` for async nodes), passing connected input values as keyword arguments. The returned dict becomes the node's output.
- **When**: When no built-in node handles your specific transformation or computation

### Concept 2: Node Registration

After defining a custom node class, you register it with `NodeRegistry.register("NodeName", NodeClass)` so the builder can reference it by string name. The `@register_node` decorator is the declarative equivalent. Once registered, your node is indistinguishable from built-in nodes.

### Concept 3: Async Nodes

Nodes can implement `process_async()` instead of (or in addition to) `process()` for I/O-bound operations. The runtime automatically detects and awaits async nodes. This is important for nodes that call external APIs, read files, or perform database queries.

### Key API

| Method / Attribute                 | Description                                               |
| ---------------------------------- | --------------------------------------------------------- |
| `Node` (base class)                | Import from `kailash.nodes.base`                          |
| `metadata` (class attr)            | `NodeMetadata` instance                                   |
| `input_parameters` (class attr)    | Dict of `NodeParameter` -- the inputs this node accepts   |
| `output_parameters` (class attr)   | Dict of `NodeParameter` -- the outputs this node produces |
| `process(**kwargs)`                | Sync processing method; returns `dict[str, Any]`          |
| `process_async(**kwargs)`          | Async processing method; returns `dict[str, Any]`         |
| `NodeRegistry.register(name, cls)` | Register a node class by string name                      |

## Code Walkthrough

```python
from __future__ import annotations

from typing import Any

from kailash import WorkflowBuilder, LocalRuntime
from kailash.nodes.base import Node, NodeMetadata, NodeParameter, NodeRegistry

# ── 1. Define a custom Node by subclassing ──────────────────────────

class UpperCaseNode(Node):
    """Converts input text to uppercase."""

    metadata = NodeMetadata(
        name="UpperCaseNode",
        description="Converts text to uppercase",
        version="1.0.0",
        tags={"text", "transform"},
    )

    input_parameters = {
        "text": NodeParameter(
            name="text",
            type="str",
            description="Input text to transform",
            required=True,
        )
    }

    output_parameters = {
        "result": NodeParameter(
            name="result",
            type="str",
            description="Uppercased text",
        )
    }

    def process(self, **kwargs: Any) -> dict[str, Any]:
        """Process the input and return uppercased text."""
        text = kwargs.get("text", "")
        return {"result": text.upper()}


# Validate the custom node's structure
assert issubclass(UpperCaseNode, Node)
assert "text" in UpperCaseNode.input_parameters
assert "result" in UpperCaseNode.output_parameters

# Instantiate and test directly
node = UpperCaseNode()
output = node.process(text="hello kailash")
assert output["result"] == "HELLO KAILASH"

# ── 2. Register the node with NodeRegistry ──────────────────────────

NodeRegistry.register("UpperCaseNode", UpperCaseNode)

retrieved = NodeRegistry.get("UpperCaseNode")
assert retrieved is UpperCaseNode

# ── 3. Use custom node in a workflow ────────────────────────────────

builder = WorkflowBuilder()

builder.add_node(
    "PythonCodeNode",
    "source",
    {
        "code": "output = 'the kailash sdk is powerful'",
        "output_type": "str",
    },
)

# Reference by class (alternative pattern for custom nodes)
builder.add_node(UpperCaseNode, "upper")

builder.connect("source", "upper", mapping={"output": "text"})

workflow = builder.build(name="custom-node-demo")

runtime = LocalRuntime()
results, run_id = runtime.execute(workflow)
runtime.close()

upper_result = results.get("upper", {}).get("result", "")
print(f"Custom node result: {upper_result}")

# ── 4. Edge case: Node without required input ───────────────────────

output_missing = node.process()  # No text provided
assert output_missing["result"] == ""  # defaults to empty string

# ── 5. Async node pattern ───────────────────────────────────────────

class AsyncGreetNode(Node):
    """An async node that greets by name."""

    metadata = NodeMetadata(
        name="AsyncGreetNode", description="Async greeting"
    )

    input_parameters = {
        "name": NodeParameter(name="name", type="str", required=True)
    }

    output_parameters = {
        "greeting": NodeParameter(name="greeting", type="str")
    }

    async def process_async(self, **kwargs: Any) -> dict[str, Any]:
        """Async process -- the runtime awaits this automatically."""
        name = kwargs.get("name", "World")
        return {"greeting": f"Hello, {name}!"}


NodeRegistry.register("AsyncGreetNode", AsyncGreetNode)
```

### Step-by-Step Explanation

1. **Subclassing**: `UpperCaseNode(Node)` inherits the node lifecycle. The class defines `metadata`, `input_parameters`, `output_parameters`, and `process()`.

2. **Direct testing**: You can instantiate a custom node and call `process()` directly for unit testing, without needing a workflow or runtime.

3. **Registration**: `NodeRegistry.register("UpperCaseNode", UpperCaseNode)` makes the node available by string name. After this, `add_node("UpperCaseNode", ...)` works just like built-in nodes.

4. **Class-based add_node**: `builder.add_node(UpperCaseNode, "upper")` is an alternative to string-based references. It passes the class directly, which is useful during development before you register the node.

5. **Graceful defaults**: Using `kwargs.get("text", "")` ensures the node handles missing inputs without crashing.

6. **Async nodes**: `process_async()` is the async equivalent of `process()`. The runtime detects it automatically and uses `await`. This is the standard pattern for I/O-bound operations.

## Common Mistakes

| Mistake                                      | Correct Pattern                              | Why                                                                           |
| -------------------------------------------- | -------------------------------------------- | ----------------------------------------------------------------------------- |
| Forgetting to return a dict from `process()` | `return {"result": value}`                   | The runtime expects a dict keyed by output parameter names                    |
| Using `self.text` instead of `kwargs`        | `text = kwargs.get("text", "")`              | Inputs are passed as keyword arguments, not set as attributes                 |
| Registering the same name twice              | Check with `NodeRegistry.get()` first        | Duplicate registration overwrites the previous class silently                 |
| Mixing sync and async in one node            | Choose one: `process()` or `process_async()` | The runtime calls whichever is defined; if both exist, async takes precedence |

## Exercises

1. Create a `ReverseStringNode` that reverses input text. Register it and use it in a workflow.
2. Create an async node that simulates a network delay using `asyncio.sleep(0.1)` before returning a result. Wire it into a workflow and execute it.
3. What happens if `process()` returns a dict with keys that do not match `output_parameters`? Does the runtime raise an error or silently pass?

## Key Takeaways

- Custom nodes extend the `Node` base class with `metadata`, `input_parameters`, `output_parameters`, and `process()`
- `process()` receives inputs as keyword arguments and must return a dict of outputs
- `NodeRegistry.register()` makes custom nodes available by string name in workflows
- You can pass the class directly to `add_node()` as an alternative to string-based references
- Async nodes implement `process_async()` instead of `process()` -- the runtime handles both transparently
- Custom nodes can be unit-tested directly without a workflow or runtime

## Next Chapter

[Chapter 4: Connections and Wiring](04_connections.md) -- Deep dive into the connection system: default ports, explicit mappings, fan-out patterns, and edge cases.
