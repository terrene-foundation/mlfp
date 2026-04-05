# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Nexus / HTTP Transport
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Configure the HTTP transport (API endpoints, port, routing)
# LEVEL: Basic
# PARITY: Full — Rust has HTTPTransport with same constructor pattern
# VALIDATES: HTTPTransport, API endpoint registration, Transport ABC
#
# Run: uv run python textbook/python/02-nexus/02_http_transport.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from nexus.transports import HTTPTransport, Transport

# ── 1. Transport ABC ───────────────────────────────────────────────
# Every Nexus transport implements the Transport abstract base class.
# The ABC defines the contract: name, start(), stop(), is_running,
# and an optional on_handler_registered() hook for hot-reload.

assert issubclass(HTTPTransport, Transport)

# The Transport protocol mirrors kailash-rs Nexus transport architecture.
# start(registry) is called by Nexus.start() with the HandlerRegistry.
# stop() is called by Nexus.stop() for graceful shutdown.

# ── 2. Create an HTTPTransport ─────────────────────────────────────
# HTTPTransport wraps FastAPI via the Core SDK enterprise gateway.
# It manages middleware queuing, router inclusion, and endpoint
# registration, all before the gateway is actually started.

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

assert isinstance(http, HTTPTransport)
assert isinstance(http, Transport)

# ── 3. Inspect transport properties ────────────────────────────────
# Each transport has a name property that uniquely identifies it.
# HTTPTransport is always "http".

assert http.name == "http"
assert http.port == 9090

# Before start(), the transport is not running and the gateway is None.
assert http.is_running is False
assert http.app is None  # No FastAPI app until gateway is created

# ── 4. Middleware queuing ──────────────────────────────────────────
# Middleware can be added before the gateway exists. HTTPTransport
# queues it internally and applies it when the gateway is created.
# This is the same pattern FastAPI uses (add middleware, then run).

from nexus.middleware.security_headers import (
    SecurityHeadersConfig,
    SecurityHeadersMiddleware,
)

http.add_middleware(SecurityHeadersMiddleware, config=SecurityHeadersConfig())

# The middleware is queued (gateway not yet created)
assert len(http._middleware_queue) == 1
assert http._middleware_queue[0].middleware_class is SecurityHeadersMiddleware

# ── 5. Endpoint registration (queued) ─────────────────────────────
# Custom endpoints can be registered before start(). They are queued
# and applied once the gateway is ready.


async def custom_health(request):
    return {"status": "ok"}


http.register_endpoint("/custom-health", ["GET"], custom_health)

# Endpoint is queued since gateway doesn't exist yet.
assert len(http._endpoint_queue) == 1
assert http._endpoint_queue[0][0] == "/custom-health"

# ── 6. Health check ────────────────────────────────────────────────
# Every transport provides a health_check() method that returns
# a dict with transport status information.

health = http.health_check()

assert health["transport"] == "http"
assert health["running"] is False
assert health["port"] == 9090
assert health["gateway"] is False  # No gateway created yet

# ── 7. HTTPTransport inside Nexus ──────────────────────────────────
# Normally you don't create HTTPTransport directly. Nexus creates
# one automatically in its constructor and stores it as _http_transport.
# The Nexus constructor parameters (api_port, cors_origins, etc.)
# are forwarded to HTTPTransport.

from nexus import Nexus

app = Nexus(
    api_port=8080,
    cors_origins=["https://app.example.com"],
    enable_durability=False,
)

# Nexus wraps an HTTPTransport internally
assert isinstance(app._http_transport, HTTPTransport)
assert app._http_transport.port == 8080

# The fastapi_app property exposes the underlying FastAPI application.
# This is the preferred way to access it (instead of _gateway.app).
assert app.fastapi_app is not None

# ── 8. Adding middleware via Nexus ─────────────────────────────────
# Nexus.add_middleware() delegates to its HTTPTransport. It also
# records MiddlewareInfo for introspection.

from nexus.middleware.cache import CacheConfig, ResponseCacheMiddleware

app.add_middleware(
    ResponseCacheMiddleware,
    config=CacheConfig(default_ttl=120, max_entries=500),
)

# The middleware stack is available for introspection
assert len(app.middleware) >= 1
last_mw = app.middleware[-1]
assert last_mw.name == "ResponseCacheMiddleware"

# ── 9. Key concepts ───────────────────────────────────────────────
# - HTTPTransport implements the Transport ABC ("http")
# - Middleware, routers, and endpoints are queued before gateway creation
# - Nexus creates HTTPTransport automatically from constructor params
# - app.fastapi_app gives access to the underlying FastAPI app
# - health_check() returns transport status for monitoring
# - NOTE: We don't call start() because it blocks

print("PASS: 02-nexus/02_http_transport")
