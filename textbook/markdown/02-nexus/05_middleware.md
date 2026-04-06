# Chapter 5: Middleware

## Overview

Middleware in Nexus follows the onion model: each middleware wraps the handler, processing requests on the way in and responses on the way out. Nexus provides built-in middleware for CORS, security headers, CSRF protection, and response caching, plus introspection via `MiddlewareInfo`. This chapter covers how to add, configure, chain, and inspect middleware in your Nexus application.

## Prerequisites

- Completed Chapters 1-4 (Nexus basics, transports)
- Understanding of HTTP middleware concepts (request/response pipeline)
- Familiarity with the Starlette/FastAPI middleware model

## Concepts

### The Onion Model

Middleware executes in LIFO order (last added = outermost). The outermost middleware sees the request first and the response last:

```
Request  -> Middleware C -> Middleware B -> Middleware A -> Handler
Response <- Middleware C <- Middleware B <- Middleware A <- Handler
```

This is identical to Starlette's model.

### Why Middleware Ordering Matters

Security headers should be added before CSRF so that security headers are always present even on CSRF-rejected requests. Response caching should be outermost to cache final responses including all header modifications.

### Built-In Middleware

| Middleware                  | Purpose                                                         | Key Configuration                 |
| --------------------------- | --------------------------------------------------------------- | --------------------------------- |
| `SecurityHeadersMiddleware` | Secure-by-default response headers (CSP, HSTS, X-Frame-Options) | `SecurityHeadersConfig`           |
| `CSRFMiddleware`            | Stateless Origin/Referer validation for state-changing methods  | `allowed_origins`, `exempt_paths` |
| `ResponseCacheMiddleware`   | TTL-based cache with LRU eviction and ETag support              | `CacheConfig`                     |

## Key API

| Method / Property                   | Parameters                                                                 | Returns                | Description                     |
| ----------------------------------- | -------------------------------------------------------------------------- | ---------------------- | ------------------------------- |
| `app.add_middleware(cls, **kwargs)` | middleware class + config                                                  | `Nexus` (for chaining) | Add middleware to the stack     |
| `app.middleware`                    | --                                                                         | `list[MiddlewareInfo]` | Introspect the middleware stack |
| `SecurityHeadersConfig()`           | `frame_options`, `hsts_max_age`, `hsts_include_subdomains`, `hsts_preload` | config object          | Security header settings        |
| `CSRFMiddleware`                    | `allowed_origins: list[str]`, `exempt_paths: list[str]`                    | middleware class       | CSRF protection                 |
| `CacheConfig()`                     | `default_ttl: int`, `max_entries: int`, `no_cache_handlers: set[str]`      | config object          | Cache settings                  |
| `MiddlewareInfo`                    | `.name: str`, `.middleware_class`, `.added_at: datetime`                   | dataclass              | Middleware metadata             |

## Code Walkthrough

### Step 1: CORS via Constructor

```python
from nexus import Nexus

app = Nexus(
    api_port=8000,
    cors_origins=["https://app.example.com", "https://admin.example.com"],
    cors_allow_methods=["GET", "POST", "PUT", "DELETE"],
    cors_allow_headers=["Authorization", "Content-Type"],
    cors_allow_credentials=False,
    cors_expose_headers=["X-Request-Id"],
    cors_max_age=3600,
    enable_durability=False,
)
```

CORS is the most common middleware. Configuring it via the Nexus constructor is the recommended approach.

### Step 2: Add Security Headers

```python
from nexus.middleware.security_headers import (
    SecurityHeadersConfig,
    SecurityHeadersMiddleware,
)

app.add_middleware(
    SecurityHeadersMiddleware,
    config=SecurityHeadersConfig(
        frame_options="SAMEORIGIN",
        hsts_max_age=31536000,
        hsts_include_subdomains=True,
    ),
)
```

`SecurityHeadersConfig` has secure defaults: `frame_options="DENY"`, `content_type_options="nosniff"`, `hsts_max_age=31536000` (1 year).

### Step 3: Add CSRF Protection

```python
from nexus.middleware.csrf import CSRFMiddleware

app2 = Nexus(api_port=8001, enable_durability=False)
app2.add_middleware(
    CSRFMiddleware,
    allowed_origins=["https://app.example.com"],
    exempt_paths=["/webhooks/stripe", "/healthz"],
)
```

CSRFMiddleware validates Origin and Referer headers for state-changing methods (POST, PUT, DELETE, PATCH). Safe methods (GET, HEAD, OPTIONS) bypass validation.

### Step 4: Add Response Caching

```python
from nexus.middleware.cache import CacheConfig, ResponseCacheMiddleware

cache_config = CacheConfig(
    default_ttl=300,
    max_entries=5000,
    no_cache_handlers={"create_user", "delete_user"},
)

app2.add_middleware(ResponseCacheMiddleware, config=cache_config)
```

`no_cache_handlers` exempts specific handlers from caching -- essential for mutation operations.

### Step 5: Introspect the Stack

```python
for mw_info in app2.middleware:
    assert isinstance(mw_info, MiddlewareInfo)
    print(f"{mw_info.name} added at {mw_info.added_at}")

names = [m.name for m in app2.middleware]
assert "CSRFMiddleware" in names
assert "ResponseCacheMiddleware" in names
```

### Step 6: Method Chaining

```python
app3 = Nexus(api_port=8002, enable_durability=False)

(
    app3.add_middleware(SecurityHeadersMiddleware, config=SecurityHeadersConfig())
    .add_middleware(CSRFMiddleware, allowed_origins=["https://example.com"])
    .add_middleware(ResponseCacheMiddleware, config=CacheConfig())
)

assert len(app3.middleware) >= 3
```

`add_middleware()` returns `self`, enabling fluent configuration.

## Common Mistakes

| Mistake                                 | Problem                                        | Fix                                                   |
| --------------------------------------- | ---------------------------------------------- | ----------------------------------------------------- |
| Adding caching before security headers  | Security headers missing from cached responses | Add security headers first (innermost)                |
| Exempting too many paths from CSRF      | Attack surface widened                         | Only exempt webhook paths that verify via other means |
| Setting `no_cache_handlers` too broadly | All responses uncached                         | Only exempt mutation handlers                         |
| Forgetting middleware ordering is LIFO  | Unexpected execution order                     | Last added = outermost (sees request first)           |

## Exercises

1. **Security Audit**: Create a Nexus instance with `SecurityHeadersConfig()` defaults. List all the response headers that will be added and explain what each protects against.

2. **CSRF Configuration**: Create a Nexus instance with CSRFMiddleware. Explain which HTTP methods are validated and which are exempt. Why is GET exempt?

3. **Middleware Chain**: Build a Nexus instance with all three middleware types using method chaining. Print the middleware stack in execution order (outermost to innermost) and explain why the order matters for security.

## Key Takeaways

- CORS is best configured via the Nexus constructor; other middleware via `add_middleware()`.
- `add_middleware(Class, **kwargs)` follows the LIFO onion model.
- `SecurityHeadersMiddleware` provides secure-by-default response headers.
- `CSRFMiddleware` performs stateless Origin/Referer validation.
- `ResponseCacheMiddleware` provides TTL + LRU + ETag caching.
- `app.middleware` returns a list of `MiddlewareInfo` for runtime introspection.
- `add_middleware()` returns `self` for method chaining.

## Next Chapter

[Chapter 6: Presets](06_presets.md) -- Use built-in presets for one-line middleware configuration and the NexusEngine builder pattern.
