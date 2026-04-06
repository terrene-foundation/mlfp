# Chapter 3: MCP Transport

## Overview

The MCP (Model Context Protocol) transport enables AI agents to interact with Nexus handlers as MCP tools. When registered, each handler becomes an MCP tool that agents like Claude can invoke directly. This chapter covers `MCPTransport` configuration, the tool naming convention, the background thread architecture, and how to add MCP support to a Nexus application. MCP is the bridge between your workflows and the AI agent ecosystem.

## Prerequisites

- Completed Chapters 1-2 (Nexus basics, HTTP transport)
- Understanding of the Transport ABC
- Conceptual familiarity with MCP (Model Context Protocol)

## Concepts

### What Is MCP Transport?

MCPTransport registers Nexus handlers as MCP tools via FastMCP. AI agents (Claude, GPT, etc.) can call these tools over the MCP protocol, enabling them to use your workflows as capabilities.

### Tool Naming Convention

When a handler named `"greet"` is registered, MCPTransport creates an MCP tool named `"{namespace}_{handler_name}"`. With the default namespace `"nexus"`, the tool becomes `"nexus_greet"`. This namespace prevents collisions when multiple Nexus instances serve tools through the same MCP platform.

### Background Thread Architecture

MCPTransport runs the FastMCP server in a **daemon thread** with its own asyncio event loop. This is necessary because the main thread is blocked by uvicorn (HTTP), so MCP needs its own loop. The lifecycle is:

1. `start()` -- creates FastMCP, registers tools, spawns daemon thread
2. `stop()` -- signals the loop to stop, joins the thread (5s timeout)

The daemon flag ensures the thread dies with the process.

### Why Is MCP Opt-In?

Unlike HTTP (always present), MCP is opt-in via `app.add_transport(MCPTransport(...))`. This is because MCP requires FastMCP as a dependency and adds a background thread -- overhead that not every application needs.

## Key API

| Method / Property                        | Parameters                                                                           | Returns        | Description                                       |
| ---------------------------------------- | ------------------------------------------------------------------------------------ | -------------- | ------------------------------------------------- | ------------------------------------- |
| `MCPTransport()`                         | `port: int = 3001`, `namespace: str = "nexus"`, `server_name: str = "kailash-nexus"` | `MCPTransport` | Create an MCP transport                           |
| `mcp.name`                               | --                                                                                   | `str`          | Always returns `"mcp"`                            |
| `mcp.port`                               | --                                                                                   | `int`          | WebSocket port for MCP server                     |
| `mcp.is_running`                         | --                                                                                   | `bool`         | Whether the transport is active                   |
| `mcp._namespace`                         | --                                                                                   | `str`          | Tool name prefix                                  |
| `mcp._server`                            | --                                                                                   | `FastMCP       | None`                                             | The underlying FastMCP server         |
| `mcp._thread`                            | --                                                                                   | `Thread        | None`                                             | The daemon thread (None before start) |
| `mcp.health_check()`                     | --                                                                                   | `dict`         | Transport health status                           |
| `mcp.on_handler_registered(handler_def)` | `handler_def: HandlerDef`                                                            | `None`         | Hot-register a new MCP tool at runtime            |
| `app.add_transport(transport)`           | `transport: Transport`                                                               | `None`         | Add a transport to Nexus for lifecycle management |

## Code Walkthrough

### Step 1: Create an MCPTransport

```python
from nexus.transports import MCPTransport, Transport

mcp = MCPTransport(
    port=4001,
    namespace="myapp",
    server_name="my-nexus-service",
)

assert isinstance(mcp, MCPTransport)
assert isinstance(mcp, Transport)
assert mcp.name == "mcp"
assert mcp.port == 4001
assert mcp.is_running is False
assert mcp._server is None
```

Key parameters: `port` sets the WebSocket port (default 3001), `namespace` sets the tool name prefix, and `server_name` sets the FastMCP server identity.

### Step 2: Understand Tool Naming

```python
assert mcp._namespace == "myapp"
# Handler "greet" becomes MCP tool "myapp_greet"
```

The namespace prevents tool name collisions. If two Nexus instances both register a `"greet"` handler, namespaces `"app1"` and `"app2"` produce distinct tools `"app1_greet"` and `"app2_greet"`.

### Step 3: Default Configuration

```python
mcp_default = MCPTransport()

assert mcp_default.port == 3001
assert mcp_default._namespace == "nexus"
assert mcp_default._server_name == "kailash-nexus"
```

The defaults produce MCP tools like `"nexus_handler_name"`.

### Step 4: Add to Nexus

```python
from nexus import Nexus

app = Nexus(api_port=8000, enable_durability=False)
custom_mcp = MCPTransport(port=5001, namespace="tutorial")
app.add_transport(custom_mcp)

assert custom_mcp in app._transports
```

`add_transport()` registers the MCP transport for lifecycle management. When `Nexus.start()` is called, it calls `transport.start(registry)` on all transports, including MCP.

### Step 5: Hot-Registration

```python
from nexus.registry import HandlerDef

example_def = HandlerDef(name="example", description="An example handler")
custom_mcp.on_handler_registered(example_def)

assert custom_mcp._server is None  # No-op before start()
```

`on_handler_registered()` is called automatically when new handlers are added. If the server is already running, it hot-registers the new tool. Before `start()`, it is a safe no-op.

### Step 6: Health Check

```python
health = mcp.health_check()

assert health["transport"] == "mcp"
assert health["running"] is False
assert health["port"] == 4001
assert health["server"] is False
```

## Common Mistakes

| Mistake                                                 | Problem                   | Fix                                                     |
| ------------------------------------------------------- | ------------------------- | ------------------------------------------------------- |
| Expecting Nexus to include MCP by default               | MCP is opt-in             | Explicitly call `app.add_transport(MCPTransport(...))`  |
| Using the same namespace for multiple instances         | Tool name collisions      | Use unique namespaces per Nexus instance                |
| Forgetting the daemon thread runs a separate event loop | Shared async state issues | Keep MCP handler logic self-contained                   |
| Using the same port for HTTP and MCP                    | Port conflict             | Use different ports (e.g., 8000 for HTTP, 3001 for MCP) |

## Exercises

1. **Namespace Design**: Create three `MCPTransport` instances with namespaces `"analytics"`, `"auth"`, and `"payments"`. For each, write out what the MCP tool name would be for a handler called `"process"`.

2. **Health Check Comparison**: Create both an `HTTPTransport` and an `MCPTransport`. Call `health_check()` on each and compare the returned dictionaries. What keys are shared? What keys are unique to each transport type?

3. **Multi-Transport Nexus**: Create a Nexus instance and add two MCPTransport instances with different ports and namespaces. Verify both are in `app._transports`. Discuss when you would use multiple MCP transports.

## Key Takeaways

- MCPTransport registers handlers as MCP tools for AI agents.
- Tool names follow the pattern `"{namespace}_{handler_name}"` to avoid collisions.
- The transport runs in a background daemon thread with its own event loop.
- MCP is added to Nexus via `app.add_transport(MCPTransport(...))` -- it is opt-in.
- `health_check()` reports transport status for monitoring.
- Hot-registration via `on_handler_registered()` supports adding tools at runtime.

## Next Chapter

[Chapter 4: SSE Transport](04_sse_transport.md) -- Configure Server-Sent Events for real-time event streaming from the Nexus EventBus.
