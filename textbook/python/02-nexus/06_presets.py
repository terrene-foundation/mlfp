# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Nexus / Presets
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Use built-in presets (lightweight, standard, saas, enterprise)
#            and the NexusEngine builder pattern
# LEVEL: Intermediate
# PARITY: Full — Rust has Preset enum and NexusEngine::builder()
# VALIDATES: Preset, PresetConfig, NexusConfig, apply_preset, get_preset,
#            NexusEngine, NexusEngineBuilder, EnterpriseMiddlewareConfig
#
# Run: uv run python textbook/python/02-nexus/06_presets.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from nexus import Nexus
from nexus.presets import PRESETS, NexusConfig, PresetConfig, apply_preset, get_preset

# ── 1. Preset registry ─────────────────────────────────────────────
# Nexus ships with five named presets, each encoding a middleware stack
# as best practice. PRESETS is a dict mapping name -> PresetConfig.

assert isinstance(PRESETS, dict)
assert "none" in PRESETS
assert "lightweight" in PRESETS
assert "standard" in PRESETS
assert "saas" in PRESETS
assert "enterprise" in PRESETS

# ── 2. PresetConfig structure ──────────────────────────────────────
# Each preset has a name, description, and two lists of factory
# functions: middleware_factories and plugin_factories.
# Factories receive a NexusConfig and return middleware/plugin tuples.

lightweight = get_preset("lightweight")

assert isinstance(lightweight, PresetConfig)
assert lightweight.name == "lightweight"
assert "CORS" in lightweight.description
assert len(lightweight.middleware_factories) == 2  # CORS + Security Headers
assert len(lightweight.plugin_factories) == 0  # No plugins

# ── 3. Inspect all presets ─────────────────────────────────────────
# Each preset adds progressively more middleware and plugins.

none_preset = get_preset("none")
assert len(none_preset.middleware_factories) == 0
assert len(none_preset.plugin_factories) == 0

standard = get_preset("standard")
assert (
    len(standard.middleware_factories) == 5
)  # CORS + SecHeaders + CSRF + RateLimit + ErrorHandler

saas = get_preset("saas")
assert len(saas.middleware_factories) == 5
assert len(saas.plugin_factories) == 4  # JWT + RBAC + Tenant + Audit

enterprise = get_preset("enterprise")
assert len(enterprise.middleware_factories) == 5
assert len(enterprise.plugin_factories) == 6  # SaaS plugins + SSO + FeatureFlags

# ── 4. get_preset validation ──────────────────────────────────────
# get_preset() raises ValueError for unknown preset names.

try:
    get_preset("nonexistent")
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "Unknown preset" in str(e)
    assert "nonexistent" in str(e)

# ── 5. NexusConfig ─────────────────────────────────────────────────
# NexusConfig holds all configuration that preset factories need.
# It covers CORS, JWT, RBAC, rate limiting, audit, SSO, and more.
# Secrets are redacted in __repr__() for safety.

config = NexusConfig(
    cors_origins=["https://app.example.com"],
    cors_allow_credentials=False,
    jwt_secret="super-secret-key",
    jwt_algorithm="HS256",
    rate_limit=200,
    audit_enabled=True,
    environment="production",
)

assert config.cors_origins == ["https://app.example.com"]
assert config.jwt_algorithm == "HS256"
assert config.rate_limit == 200
assert config.environment == "production"

# Secrets are redacted in repr
repr_str = repr(config)
assert "[REDACTED]" in repr_str
assert "super-secret-key" not in repr_str

# ── 6. NexusConfig defaults ───────────────────────────────────────
# Default values are secure and development-friendly.

defaults = NexusConfig()

assert defaults.cors_origins == ["*"]  # Permissive for dev
assert defaults.cors_allow_credentials is False  # Secure default
assert defaults.jwt_secret is None  # No JWT by default
assert defaults.rate_limit == 100  # 100 req/min
assert defaults.audit_enabled is True
assert defaults.tenant_header == "X-Tenant-ID"
assert defaults.environment == "development"

# ── 7. Using presets with Nexus constructor ────────────────────────
# The simplest way to apply a preset: pass preset= to the Nexus
# constructor. Nexus builds a NexusConfig from its own parameters
# and calls apply_preset() internally.

app = Nexus(
    preset="lightweight",
    cors_origins=["https://app.example.com"],
    enable_durability=False,
)

assert isinstance(app, Nexus)
assert app._active_preset == "lightweight"

# ── 8. Applying presets manually ──────────────────────────────────
# For more control, create a NexusConfig and call apply_preset()
# directly. This lets you customize the config before applying.

app2 = Nexus(api_port=8001, enable_durability=False)

custom_config = NexusConfig(
    cors_origins=["https://custom.example.com"],
    rate_limit=500,
    environment="staging",
)

apply_preset(app2, "lightweight", custom_config)

# The preset middleware is now applied to app2.

# ── 9. NexusEngine builder pattern ─────────────────────────────────
# NexusEngine wraps Nexus with enterprise middleware and a fluent
# builder API. This matches kailash-rs NexusEngine for cross-SDK
# parity. Use NexusEngine for production deployments.

from nexus import NexusEngine, Preset
from nexus.engine import EnterpriseMiddlewareConfig, NexusEngineBuilder

# The Preset enum has three values
assert Preset.NONE.value == "none"
assert Preset.SAAS.value == "saas"
assert Preset.ENTERPRISE.value == "enterprise"

# ── 10. NexusEngine.builder() ──────────────────────────────────────
# builder() returns a NexusEngineBuilder with fluent methods.

builder = NexusEngine.builder()
assert isinstance(builder, NexusEngineBuilder)

# Chain configuration methods
engine = (
    NexusEngine.builder()
    .preset(Preset.SAAS)
    .bind("0.0.0.0:8080")
    .config(enable_durability=False)
    .build()
)

assert isinstance(engine, NexusEngine)
assert isinstance(engine.nexus, Nexus)
assert engine.bind_addr == "0.0.0.0:8080"

# ── 11. EnterpriseMiddlewareConfig ─────────────────────────────────
# For fine-grained control, pass an explicit EnterpriseMiddlewareConfig
# to the builder. This overrides the preset.

enterprise_config = EnterpriseMiddlewareConfig(
    enable_csrf=True,
    enable_audit=True,
    enable_metrics=True,
    enable_rate_limiting=True,
    rate_limit_rpm=500,
    enable_cors=True,
    cors_origins=["https://api.example.com"],
)

assert enterprise_config.rate_limit_rpm == 500
assert enterprise_config.enable_csrf is True

engine2 = (
    NexusEngine.builder()
    .enterprise(enterprise_config)
    .bind("0.0.0.0:9090")
    .config(enable_durability=False)
    .build()
)

assert engine2.enterprise_config is enterprise_config
assert engine2.enterprise_config.rate_limit_rpm == 500

# ── 12. NexusEngine registration ──────────────────────────────────
# NexusEngine delegates register() and start() to the underlying
# Nexus instance. Register workflows on the engine directly.

from kailash import WorkflowBuilder

wb = WorkflowBuilder()
wb.add_node(
    "PythonCodeNode",
    "double",
    {
        "code": "output = n * 2",
        "inputs": {"n": "int"},
        "output_type": "int",
    },
)

engine.register("double", wb.build(name="double"))

# The workflow is registered on the underlying Nexus instance
assert "double" in engine.nexus._workflows

# ── 13. Zero-config NexusEngine ────────────────────────────────────
# The simplest NexusEngine: no preset, default bind address.

simple = NexusEngine.builder().config(enable_durability=False).build()
assert isinstance(simple, NexusEngine)
assert simple.bind_addr == "0.0.0.0:3000"  # Default port 3000
assert simple.enterprise_config is None  # No enterprise features

# ── 14. Key concepts ──────────────────────────────────────────────
# - PRESETS: dict of 5 named presets (none, lightweight, standard, saas, enterprise)
# - get_preset(name) -> PresetConfig (raises ValueError for unknowns)
# - NexusConfig: all settings factories need (CORS, JWT, rate limit, etc.)
# - Nexus(preset="lightweight"): one-line middleware stack
# - apply_preset(app, name, config): manual preset application
# - NexusEngine.builder(): fluent builder with .preset(), .bind(), .build()
# - Preset enum: NONE, SAAS, ENTERPRISE (cross-SDK parity)
# - EnterpriseMiddlewareConfig: fine-grained override for the builder
# - NOTE: We don't call start() because it blocks

print("PASS: 02-nexus/06_presets")
