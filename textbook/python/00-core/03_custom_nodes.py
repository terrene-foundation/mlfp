# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Core / Custom Nodes
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Create custom nodes by subclassing Node and using @register_node
# LEVEL: Intermediate
# PARITY: Equivalent — Python uses class + @register_node decorator;
#         Rust uses impl Node for T trait (DIV-014: from_function/from_class
#         not applicable to Rust)
# VALIDATES: Node subclass, @register_node, process() method
#
# Run: uv run python textbook/python/00-core/03_custom_nodes.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from typing import Any

from kailash import WorkflowBuilder, LocalRuntime
from kailash.nodes.base import Node, NodeMetadata, NodeParameter, NodeRegistry

# ── 1. Define a custom Node by subclassing ──────────────────────────
# Every custom node must:
#   - Inherit from Node
#   - Define input_parameters and output_parameters as class attributes
#   - Implement the process() method (sync) or process_async() (async)


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
assert output["result"] == "HELLO KAILASH", "Direct process() call works"

# ── 2. Register the node with @register_node ────────────────────────
# Once registered, the node can be referenced by string name in workflows.
# NodeRegistry.register() is the programmatic equivalent of @register_node.

NodeRegistry.register("UpperCaseNode", UpperCaseNode)

# Verify registration
retrieved = NodeRegistry.get("UpperCaseNode")
assert retrieved is UpperCaseNode, "Registered class is retrievable by name"

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
assert len(workflow.node_instances) == 2

# Execute
runtime = LocalRuntime()
results, run_id = runtime.execute(workflow)
runtime.close()

upper_result = results.get("upper", {}).get("result", "")
print(f"Custom node result: {upper_result}")

# ── 4. Edge case: Node without required input ───────────────────────
# Calling process() without a required parameter should still work
# (the node handles missing input gracefully via kwargs.get)

output_missing = node.process()  # No text provided
assert output_missing["result"] == "", "Missing input defaults to empty string"

# ── 5. Async node pattern ───────────────────────────────────────────
# Nodes can also implement process_async() for async operations.
# The runtime handles both sync and async nodes transparently.
# DIV-013: In Rust, ALL nodes are async (no sync variant).


class AsyncGreetNode(Node):
    """An async node that greets by name."""

    metadata = NodeMetadata(name="AsyncGreetNode", description="Async greeting")

    input_parameters = {"name": NodeParameter(name="name", type="str", required=True)}

    output_parameters = {"greeting": NodeParameter(name="greeting", type="str")}

    async def process_async(self, **kwargs: Any) -> dict[str, Any]:
        """Async process — the runtime awaits this automatically."""
        name = kwargs.get("name", "World")
        return {"greeting": f"Hello, {name}!"}


# Async nodes work in workflows just like sync nodes
NodeRegistry.register("AsyncGreetNode", AsyncGreetNode)

print("PASS: 00-core/03_custom_nodes")
