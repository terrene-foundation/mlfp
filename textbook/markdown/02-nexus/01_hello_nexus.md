# Chapter 1: Hello Nexus

## Overview

Nexus is the Kailash SDK's multi-channel deployment platform. It provides a single entry point that exposes workflows simultaneously as HTTP APIs, CLI commands, and MCP tools for AI agents. This chapter introduces the core Nexus concepts: creating an instance, registering workflows, using the `@handler` decorator, and understanding the preset system. By the end, you will know how to set up a Nexus application and register your first handlers without starting a server.

## Prerequisites

- Kailash Core SDK installed (`pip install kailash`)
- Kailash Nexus installed (`pip install kailash-nexus`)
- Familiarity with `WorkflowBuilder` from the Core SDK (Package 00)
- Python 3.10+ with `async`/`await` syntax

## Concepts

### What Is Nexus?

Nexus is analogous to web frameworks like FastAPI, but with a key difference: instead of exposing only HTTP endpoints, Nexus exposes every registered workflow across **three channels** simultaneously:

| Channel  | Protocol               | Endpoint Pattern                 |
| -------- | ---------------------- | -------------------------------- |
| HTTP API | REST over HTTP         | `POST /workflows/{name}/execute` |
| CLI      | Command-line           | `nexus run {name} --param value` |
| MCP      | Model Context Protocol | Tool `workflow_{name}`           |

This means you write the workflow once and it becomes available to web clients, terminal users, and AI agents at the same time.

### Why Use Nexus?

Without Nexus, deploying a workflow requires writing separate HTTP routes, CLI argument parsers, and MCP tool registrations. Nexus eliminates this boilerplate by auto-generating all three interfaces from a single registration call.

### How Does Registration Work?

Nexus provides two registration mechanisms:

1. **`register(name, workflow)`** -- registers an existing Kailash workflow by name. The workflow's nodes define the input/output schema.
2. **`@handler` decorator** -- registers an async function directly. Nexus inspects the function's type annotations to derive parameters automatically.

### When to Use Each Registration Method

Use `register()` when you have a pre-built workflow with multiple nodes and complex logic. Use `@handler` when you want to expose a simple async function without building a full workflow graph.

## Key API

| Method / Property                 | Parameters                                                                                  | Returns                   | Description                                |
| --------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------- | ------------------------------------------ |
| `Nexus()`                         | `api_port: int = 8000`, `cors_origins: list[str]`, `enable_durability: bool`, `preset: str` | `Nexus`                   | Create a new Nexus instance                |
| `app.register(name, workflow)`    | `name: str`, `workflow: Workflow`                                                           | `None`                    | Register a workflow on all channels        |
| `@app.handler(name, description)` | `name: str`, `description: str`                                                             | decorator                 | Register an async function as a handler    |
| `app._registry`                   | --                                                                                          | `HandlerRegistry`         | Access the handler registry                |
| `app.start()`                     | --                                                                                          | `None`                    | Start listening on all channels (blocking) |
| `get_preset(name)`                | `name: str`                                                                                 | `PresetConfig`            | Retrieve a named middleware preset         |
| `PRESETS`                         | --                                                                                          | `dict[str, PresetConfig]` | All available preset configurations        |

## Code Walkthrough

### Step 1: Import and Create Nexus

```python
from kailash import WorkflowBuilder
from nexus import Nexus

app = Nexus(
    api_port=8000,
    cors_origins=["http://localhost:3000"],
    enable_durability=False,
)
```

The `Nexus()` constructor creates the application instance. `api_port` sets the HTTP port (default 8000). `cors_origins` configures Cross-Origin Resource Sharing for browser clients. `enable_durability` controls response caching -- disable it for tutorials.

### Step 2: Register a Workflow

```python
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
workflow = builder.build(name="greeter")

app.register("greet", workflow)
```

`register()` makes the workflow available on all channels. After this call, the workflow is accessible at `POST /workflows/greet/execute` (HTTP), as a CLI command, and as an MCP tool.

### Step 3: Register via @handler

```python
@app.handler("add", description="Add two numbers")
async def add(a: int, b: int = 0) -> dict:
    return {"sum": a + b}
```

The `@handler` decorator is the preferred pattern for simple functions. Nexus inspects the function signature to derive:

- Required parameters: `a: int` (no default)
- Optional parameters: `b: int = 0` (has default)
- Return type: `dict`

### Step 4: Inspect the Registry

```python
registry = app._registry
print(f"Registered entries: {registry}")
```

The handler registry tracks all registered workflows and handlers. Both the workflow-based `greet` and the function-based `add` appear in the registry.

### Step 5: Use Presets

```python
from nexus import get_preset, PRESETS

app_prod = Nexus(
    preset="lightweight",
    cors_origins=["https://app.example.com"],
    enable_durability=False,
)
```

Presets provide one-line middleware configuration. The `"lightweight"` preset adds CORS and security headers. Other presets (`"standard"`, `"saas"`, `"enterprise"`) add progressively more middleware.

> **Important:** `app.start()` is not called in this tutorial because it blocks the process, listening for incoming requests. In production, `start()` is the final call.

## Common Mistakes

| Mistake                                           | Problem                                           | Fix                                                   |
| ------------------------------------------------- | ------------------------------------------------- | ----------------------------------------------------- |
| Calling `app.start()` before `register()`         | Handlers are not available when the server starts | Always register all handlers before calling `start()` |
| Using synchronous functions with `@handler`       | Nexus handlers must be async                      | Define handler functions with `async def`             |
| Hardcoding port numbers across environments       | Port conflicts in deployment                      | Use environment variables for port configuration      |
| Forgetting `enable_durability=False` in tutorials | Response caching interferes with testing          | Disable durability for development and tutorials      |

## Exercises

1. **Register Multiple Handlers**: Create a Nexus instance and register three handlers: `multiply`, `subtract`, and `divide`. Inspect the registry to verify all three are registered. Ensure `divide` handles the division-by-zero case.

2. **Preset Exploration**: Iterate over `PRESETS` and print the name, description, and number of middleware factories for each preset. Which preset has the most middleware?

3. **Workflow vs Handler**: Register the same logic (string concatenation) both as a workflow via `register()` and as a function via `@handler`. Compare the registration process and discuss when each approach is appropriate.

## Key Takeaways

- Nexus is a multi-channel deployment platform: one registration exposes workflows via HTTP, CLI, and MCP simultaneously.
- `register(name, workflow)` registers existing workflows; `@handler` registers async functions directly.
- The `@handler` decorator derives parameters from Python type annotations automatically.
- Presets provide one-line middleware stacks (lightweight, standard, saas, enterprise).
- Never call `start()` until all handlers are registered.

## Next Chapter

[Chapter 2: HTTP Transport](02_http_transport.md) -- Learn how Nexus configures the HTTP transport layer, including middleware queuing, endpoint registration, and the Transport ABC.
