# Chapter 2: HTTP Transport

## Overview

The HTTP transport is the primary channel through which Nexus serves workflows as REST API endpoints. Under the hood, Nexus wraps FastAPI via the Kailash Core SDK's enterprise gateway. This chapter covers the `Transport` abstract base class, `HTTPTransport` configuration, middleware queuing, endpoint registration, and how Nexus creates the transport internally. Understanding the transport layer is essential for customizing API behavior beyond the defaults.

## Prerequisites

- Completed Chapter 1: Hello Nexus
- Familiarity with HTTP concepts (ports, CORS, middleware)
- Understanding of abstract base classes in Python

## Concepts

### The Transport ABC

Every Nexus transport implements the `Transport` abstract base class, which defines a uniform contract across all channel types (HTTP, MCP, SSE). The ABC requires:

- **`name`** property -- a unique string identifier (e.g., `"http"`, `"mcp"`)
- **`start(registry)`** -- called by `Nexus.start()` with the handler registry
- **`stop()`** -- called by `Nexus.stop()` for graceful shutdown
- **`is_running`** -- boolean property indicating whether the transport is active
- **`on_handler_registered()`** -- optional hook for hot-reloading new handlers

This architecture mirrors the Rust SDK's transport system for cross-SDK parity.

### Middleware Queuing

HTTPTransport supports adding middleware and endpoints _before_ the underlying FastAPI gateway is created. These are queued internally and applied when the gateway initializes during `start()`. This follows the same pattern FastAPI uses: configure everything, then run.

### Why Does Nexus Create HTTPTransport Automatically?

When you construct `Nexus(api_port=8080, cors_origins=[...])`, the constructor parameters are forwarded to an internal `HTTPTransport`. You rarely need to create `HTTPTransport` directly -- Nexus handles it. Direct creation is useful only when you need multiple HTTP transports or fine-grained control.

## Key API

| Method / Property                                | Parameters                                                                                                                                                                          | Returns                | Description                                       |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- | ------------------------------------------------- | ---------------------------------------------- |
| `HTTPTransport()`                                | `port`, `cors_origins`, `cors_allow_methods`, `cors_allow_headers`, `cors_allow_credentials`, `cors_max_age`, `enable_auth`, `enable_monitoring`, `enable_durability`, `rate_limit` | `HTTPTransport`        | Create an HTTP transport                          |
| `http.name`                                      | --                                                                                                                                                                                  | `str`                  | Always returns `"http"`                           |
| `http.port`                                      | --                                                                                                                                                                                  | `int`                  | The configured port number                        |
| `http.is_running`                                | --                                                                                                                                                                                  | `bool`                 | Whether the transport is active                   |
| `http.app`                                       | --                                                                                                                                                                                  | `FastAPI               | None`                                             | The underlying FastAPI app (None before start) |
| `http.add_middleware(cls, **kwargs)`             | middleware class + config                                                                                                                                                           | `None`                 | Queue middleware for later application            |
| `http.register_endpoint(path, methods, handler)` | `path: str`, `methods: list`, `handler: callable`                                                                                                                                   | `None`                 | Queue a custom endpoint                           |
| `http.health_check()`                            | --                                                                                                                                                                                  | `dict`                 | Transport health status                           |
| `app.fastapi_app`                                | --                                                                                                                                                                                  | `FastAPI`              | Access the underlying FastAPI app via Nexus       |
| `app.add_middleware(cls, **kwargs)`              | middleware class + config                                                                                                                                                           | `Nexus`                | Add middleware via Nexus (delegates to transport) |
| `app.middleware`                                 | --                                                                                                                                                                                  | `list[MiddlewareInfo]` | Introspect the middleware stack                   |

## Code Walkthrough

### Step 1: Understand the Transport ABC

```python
from nexus.transports import HTTPTransport, Transport

assert issubclass(HTTPTransport, Transport)
```

Every transport implements `Transport`. This ensures all transports share a consistent lifecycle contract.

### Step 2: Create an HTTPTransport

```python
http = HTTPTransport(
    port=9090,
    cors_origins=["http://localhost:3000"],
    cors_allow_methods=["GET", "POST"],
    cors_allow_headers=["Authorization", "Content-Type"],
    cors_allow_credentials=False,
    cors_max_age=600,
    enable_auth=False,
    enable_monitoring=False,
    enable_durability=False,
    rate_limit=200,
)
```

The constructor accepts comprehensive CORS configuration. Before `start()` is called, the transport is not running and the FastAPI app is `None`.

```python
assert http.name == "http"
assert http.port == 9090
assert http.is_running is False
assert http.app is None
```

### Step 3: Queue Middleware

```python
from nexus.middleware.security_headers import (
    SecurityHeadersConfig,
    SecurityHeadersMiddleware,
)

http.add_middleware(SecurityHeadersMiddleware, config=SecurityHeadersConfig())
```

Middleware is queued internally. The queue is applied when the gateway is created during `start()`.

```python
assert len(http._middleware_queue) == 1
assert http._middleware_queue[0].middleware_class is SecurityHeadersMiddleware
```

### Step 4: Register Custom Endpoints

```python
async def custom_health(request):
    return {"status": "ok"}

http.register_endpoint("/custom-health", ["GET"], custom_health)
```

Custom endpoints are also queued. They are registered on the FastAPI router when the gateway starts.

### Step 5: Check Health

```python
health = http.health_check()

assert health["transport"] == "http"
assert health["running"] is False
assert health["port"] == 9090
assert health["gateway"] is False
```

`health_check()` returns a dictionary with transport status information, useful for monitoring systems.

### Step 6: HTTPTransport Inside Nexus

```python
from nexus import Nexus

app = Nexus(
    api_port=8080,
    cors_origins=["https://app.example.com"],
    enable_durability=False,
)

assert isinstance(app._http_transport, HTTPTransport)
assert app._http_transport.port == 8080
assert app.fastapi_app is not None
```

Nexus creates an `HTTPTransport` internally. The `fastapi_app` property provides access to the underlying FastAPI application.

### Step 7: Add Middleware via Nexus

```python
from nexus.middleware.cache import CacheConfig, ResponseCacheMiddleware

app.add_middleware(
    ResponseCacheMiddleware,
    config=CacheConfig(default_ttl=120, max_entries=500),
)

assert len(app.middleware) >= 1
last_mw = app.middleware[-1]
assert last_mw.name == "ResponseCacheMiddleware"
```

`Nexus.add_middleware()` delegates to the HTTP transport and also records `MiddlewareInfo` objects for introspection.

## Common Mistakes

| Mistake                                                         | Problem                       | Fix                                                            |
| --------------------------------------------------------------- | ----------------------------- | -------------------------------------------------------------- |
| Creating HTTPTransport directly when Nexus suffices             | Unnecessary complexity        | Use `Nexus()` constructor parameters instead                   |
| Expecting `http.app` to be non-None before `start()`            | Gateway is created lazily     | Access the app only after `start()` or via `Nexus.fastapi_app` |
| Adding middleware after `start()`                               | Middleware may not be applied | Add all middleware before calling `start()`                    |
| Setting `cors_allow_credentials=True` with `cors_origins=["*"]` | Browser security violation    | Use specific origins with credentials                          |

## Exercises

1. **Transport Properties**: Create an `HTTPTransport` with custom port and CORS settings. Verify all properties via assertions. Call `health_check()` and examine each key in the returned dictionary.

2. **Middleware Stack**: Create an `HTTPTransport`, queue three different middleware classes (security headers, CSRF, cache), and verify the queue length and order. Explain why order matters given the onion model.

3. **Nexus Integration**: Create a `Nexus` instance, access the internal `HTTPTransport`, add middleware both via the transport directly and via `Nexus.add_middleware()`. Compare the two approaches.

## Key Takeaways

- `HTTPTransport` implements the `Transport` ABC with the name `"http"`.
- Middleware, routers, and endpoints are queued before the gateway is created.
- Nexus creates `HTTPTransport` automatically from its constructor parameters.
- `app.fastapi_app` provides access to the underlying FastAPI application.
- `health_check()` returns transport status for monitoring.
- The gateway (and thus the FastAPI app) is created during `start()`.

## Next Chapter

[Chapter 3: MCP Transport](03_mcp_transport.md) -- Configure the MCP transport for AI agent integration, including tool naming conventions and the background thread architecture.
