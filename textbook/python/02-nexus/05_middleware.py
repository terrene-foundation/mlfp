# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Nexus / Middleware
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Add middleware to Nexus (CORS, security headers, CSRF,
#            response caching, custom middleware)
# LEVEL: Intermediate
# PARITY: Full — Rust has add_middleware with same onion model
# VALIDATES: Middleware configuration, CORS settings, MiddlewareInfo,
#            SecurityHeadersMiddleware, CSRFMiddleware, ResponseCacheMiddleware
#
# Run: uv run python textbook/python/02-nexus/05_middleware.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from nexus import MiddlewareInfo, Nexus

# ── 1. CORS via constructor ────────────────────────────────────────
# The most common middleware is CORS. Nexus applies it automatically
# when cors_origins is provided to the constructor. This is the
# recommended approach for most applications.

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

assert isinstance(app, Nexus)

# ── 2. The onion model ────────────────────────────────────────────
# Middleware executes in LIFO order (last added = outermost).
# This follows Starlette's onion model:
#
#   Request  -> Middleware C -> Middleware B -> Middleware A -> Handler
#   Response <- Middleware C <- Middleware B <- Middleware A <- Handler
#
# The outermost middleware sees the request first and the response last.

# ── 3. add_middleware() ────────────────────────────────────────────
# Nexus.add_middleware() accepts a middleware CLASS (not instance)
# and keyword arguments for the constructor.

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

# add_middleware returns self for chaining
result = app.add_middleware(
    SecurityHeadersMiddleware,
    config=SecurityHeadersConfig(),
)
assert result is app  # Chaining works

# ── 4. SecurityHeadersMiddleware ───────────────────────────────────
# Adds standard security response headers:
#   Content-Security-Policy, Strict-Transport-Security,
#   X-Content-Type-Options, X-Frame-Options, X-XSS-Protection,
#   Referrer-Policy, Permissions-Policy
#
# SecurityHeadersConfig has secure defaults. Override only when
# you have a specific reason to relax a policy.

config = SecurityHeadersConfig()

assert config.frame_options == "DENY"  # Default: deny all framing
assert config.content_type_options == "nosniff"
assert config.hsts_max_age == 31536000  # 1 year
assert config.hsts_include_subdomains is True
assert config.hsts_preload is False  # Opt-in for HSTS preload list

# ── 5. CSRFMiddleware ──────────────────────────────────────────────
# Validates Origin and Referer headers for state-changing methods
# (POST, PUT, DELETE, PATCH). Safe methods (GET, HEAD, OPTIONS)
# bypass validation entirely.
#
# This is a lightweight, stateless CSRF protection suitable for APIs.

from nexus.middleware.csrf import CSRFMiddleware

app2 = Nexus(api_port=8001, enable_durability=False)

app2.add_middleware(
    CSRFMiddleware,
    allowed_origins=["https://app.example.com"],
    exempt_paths=["/webhooks/stripe", "/healthz"],
)

# Verify it was added to the middleware stack
csrf_added = any(m.middleware_class is CSRFMiddleware for m in app2.middleware)
assert csrf_added

# ── 6. ResponseCacheMiddleware ─────────────────────────────────────
# TTL-based response cache with LRU eviction, ETag support, and
# Cache-Control header parsing. Thread-safe for concurrent access.

from nexus.middleware.cache import CacheConfig, ResponseCacheMiddleware

cache_config = CacheConfig(
    default_ttl=300,  # 5-minute TTL
    max_entries=5000,  # LRU eviction threshold
    no_cache_handlers={"create_user", "delete_user"},  # Exempt handlers
)

assert cache_config.default_ttl == 300
assert cache_config.max_entries == 5000
assert "create_user" in cache_config.no_cache_handlers

app2.add_middleware(ResponseCacheMiddleware, config=cache_config)

# ── 7. Middleware introspection ────────────────────────────────────
# Nexus tracks all added middleware in a list of MiddlewareInfo objects.
# This allows runtime inspection of the middleware stack.

assert len(app2.middleware) >= 2  # CSRF + cache at minimum

for mw_info in app2.middleware:
    assert isinstance(mw_info, MiddlewareInfo)
    assert isinstance(mw_info.name, str)  # Class name
    assert mw_info.added_at is not None  # Timestamp

# Check specific entries
names = [m.name for m in app2.middleware]
assert "CSRFMiddleware" in names
assert "ResponseCacheMiddleware" in names

# ── 8. Method chaining ────────────────────────────────────────────
# add_middleware() returns self, enabling fluent configuration.

app3 = Nexus(api_port=8002, enable_durability=False)

(
    app3.add_middleware(SecurityHeadersMiddleware, config=SecurityHeadersConfig())
    .add_middleware(CSRFMiddleware, allowed_origins=["https://example.com"])
    .add_middleware(ResponseCacheMiddleware, config=CacheConfig())
)

assert len(app3.middleware) >= 3

# ── 9. Middleware ordering matters ─────────────────────────────────
# Because of LIFO (onion model), the ORDER you add middleware
# determines execution order:
#
#   app.add_middleware(A)  # Innermost (closest to handler)
#   app.add_middleware(B)  # Middle
#   app.add_middleware(C)  # Outermost (sees request first)
#
# For security, add SecurityHeaders BEFORE CSRF so that security
# headers are always present even on CSRF-rejected requests.
# Response caching should be the outermost to cache final responses.

# ── 10. Key concepts ──────────────────────────────────────────────
# - CORS: configured via Nexus constructor (recommended) or add_middleware
# - add_middleware(Class, **kwargs): LIFO onion model
# - SecurityHeadersMiddleware: secure-by-default response headers
# - CSRFMiddleware: stateless Origin/Referer validation
# - ResponseCacheMiddleware: TTL + LRU + ETag caching
# - app.middleware: list of MiddlewareInfo for introspection
# - Method chaining: app.add_middleware(A).add_middleware(B)
# - NOTE: We don't call app.start() because it blocks

print("PASS: 02-nexus/05_middleware")
