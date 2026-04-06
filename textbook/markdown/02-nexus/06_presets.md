# Chapter 6: Presets

## Overview

Presets provide one-line middleware configuration for common deployment scenarios. Instead of manually adding each middleware, choose a preset that encodes best practices: `none`, `lightweight`, `standard`, `saas`, or `enterprise`. This chapter also covers `NexusConfig` for centralized settings, the `NexusEngine` builder pattern for production deployments, and the `Preset` enum for cross-SDK parity with the Rust implementation.

## Prerequisites

- Completed Chapter 5: Middleware
- Understanding of middleware stacks and their purpose
- Familiarity with the builder pattern

## Concepts

### What Are Presets?

A preset is a named collection of middleware and plugin factories. Each factory receives a `NexusConfig` and returns a middleware or plugin tuple. Presets are progressive -- each level includes everything from the previous level plus more:

| Preset        | Middleware Count | Plugin Count | Includes                            |
| ------------- | ---------------- | ------------ | ----------------------------------- |
| `none`        | 0                | 0            | Nothing                             |
| `lightweight` | 2                | 0            | CORS + Security Headers             |
| `standard`    | 5                | 0            | + CSRF + Rate Limit + Error Handler |
| `saas`        | 5                | 4            | + JWT + RBAC + Tenant + Audit       |
| `enterprise`  | 5                | 6            | + SSO + Feature Flags               |

### NexusConfig

`NexusConfig` centralizes all settings that preset factories need: CORS origins, JWT secrets, rate limits, audit settings, and more. Secrets are redacted in `__repr__()` for safety.

### NexusEngine

`NexusEngine` wraps `Nexus` with enterprise middleware and a fluent builder API. It matches `kailash-rs NexusEngine` for cross-SDK parity and is the recommended entry point for production deployments.

## Key API

| Method / Property                 | Parameters                                                            | Returns                   | Description                                           |
| --------------------------------- | --------------------------------------------------------------------- | ------------------------- | ----------------------------------------------------- |
| `PRESETS`                         | --                                                                    | `dict[str, PresetConfig]` | All named presets                                     |
| `get_preset(name)`                | `name: str`                                                           | `PresetConfig`            | Get preset by name (raises `ValueError` for unknowns) |
| `PresetConfig`                    | `.name`, `.description`, `.middleware_factories`, `.plugin_factories` | dataclass                 | Preset definition                                     |
| `NexusConfig()`                   | `cors_origins`, `jwt_secret`, `rate_limit`, `environment`, etc.       | config object             | Centralized settings                                  |
| `apply_preset(app, name, config)` | `app: Nexus`, `name: str`, `config: NexusConfig`                      | `None`                    | Apply preset manually                                 |
| `NexusEngine.builder()`           | --                                                                    | `NexusEngineBuilder`      | Start building a NexusEngine                          |
| `.preset(p)`                      | `p: Preset`                                                           | builder                   | Set preset on builder                                 |
| `.bind(addr)`                     | `addr: str`                                                           | builder                   | Set bind address                                      |
| `.config(**kwargs)`               | keyword args                                                          | builder                   | Set Nexus config                                      |
| `.build()`                        | --                                                                    | `NexusEngine`             | Build the engine                                      |
| `.enterprise(config)`             | `EnterpriseMiddlewareConfig`                                          | builder                   | Set enterprise config                                 |
| `Preset` enum                     | `NONE`, `SAAS`, `ENTERPRISE`                                          | enum                      | Cross-SDK preset enum                                 |

## Code Walkthrough

### Step 1: Explore the Preset Registry

```python
from nexus.presets import PRESETS, get_preset

assert "none" in PRESETS
assert "lightweight" in PRESETS
assert "standard" in PRESETS
assert "saas" in PRESETS
assert "enterprise" in PRESETS

lightweight = get_preset("lightweight")
assert lightweight.name == "lightweight"
assert len(lightweight.middleware_factories) == 2
```

### Step 2: NexusConfig

```python
from nexus.presets import NexusConfig

config = NexusConfig(
    cors_origins=["https://app.example.com"],
    jwt_secret="super-secret-key",
    rate_limit=200,
    environment="production",
)

# Secrets are redacted in repr
repr_str = repr(config)
assert "[REDACTED]" in repr_str
assert "super-secret-key" not in repr_str
```

### Step 3: Using Presets with Nexus Constructor

```python
from nexus import Nexus

app = Nexus(
    preset="lightweight",
    cors_origins=["https://app.example.com"],
    enable_durability=False,
)

assert app._active_preset == "lightweight"
```

This is the simplest way to apply a preset -- pass `preset=` to the constructor.

### Step 4: NexusEngine Builder

```python
from nexus import NexusEngine, Preset

engine = (
    NexusEngine.builder()
    .preset(Preset.SAAS)
    .bind("0.0.0.0:8080")
    .config(enable_durability=False)
    .build()
)

assert isinstance(engine.nexus, Nexus)
assert engine.bind_addr == "0.0.0.0:8080"
```

The builder pattern provides fluent configuration for production deployments.

### Step 5: EnterpriseMiddlewareConfig

```python
from nexus.engine import EnterpriseMiddlewareConfig

enterprise_config = EnterpriseMiddlewareConfig(
    enable_csrf=True,
    enable_audit=True,
    enable_metrics=True,
    enable_rate_limiting=True,
    rate_limit_rpm=500,
    enable_cors=True,
    cors_origins=["https://api.example.com"],
)

engine2 = (
    NexusEngine.builder()
    .enterprise(enterprise_config)
    .bind("0.0.0.0:9090")
    .config(enable_durability=False)
    .build()
)

assert engine2.enterprise_config.rate_limit_rpm == 500
```

### Step 6: Register Workflows on NexusEngine

```python
from kailash import WorkflowBuilder

wb = WorkflowBuilder()
wb.add_node("PythonCodeNode", "double", {
    "code": "output = n * 2",
    "inputs": {"n": "int"},
    "output_type": "int",
})

engine.register("double", wb.build(name="double"))
assert "double" in engine.nexus._workflows
```

`NexusEngine.register()` delegates to the underlying Nexus instance.

## Common Mistakes

| Mistake                                                  | Problem                         | Fix                                    |
| -------------------------------------------------------- | ------------------------------- | -------------------------------------- |
| Using `saas` preset without JWT secret                   | JWT middleware fails at runtime | Set `jwt_secret` in NexusConfig        |
| Misspelling preset name                                  | `ValueError: Unknown preset`    | Check `PRESETS.keys()` for valid names |
| Overriding preset with `add_middleware()` after applying | Duplicate middleware            | Choose one approach: preset OR manual  |
| Logging NexusConfig with secrets                         | Security leak                   | Use `repr()` which redacts secrets     |

## Exercises

1. **Preset Inventory**: Iterate over all presets and print their name, middleware count, and plugin count. Create a table showing the progressive growth from `none` to `enterprise`.

2. **NexusConfig Defaults**: Create a `NexusConfig()` with no arguments. Print all default values. Which settings are development-friendly? Which are security-conscious?

3. **Engine Builder**: Build a `NexusEngine` with the `ENTERPRISE` preset, a bind address of `"0.0.0.0:443"`, and an `EnterpriseMiddlewareConfig` with rate limiting at 1000 RPM. Register a workflow and verify it appears in the engine's Nexus instance.

## Key Takeaways

- Five named presets provide progressive middleware stacks: none, lightweight, standard, saas, enterprise.
- `get_preset(name)` retrieves a preset; `ValueError` for unknown names.
- `NexusConfig` centralizes all settings with secret redaction in `repr()`.
- `Nexus(preset="lightweight")` is the simplest one-line middleware setup.
- `NexusEngine.builder()` provides a fluent builder for production deployments.
- The `Preset` enum (`NONE`, `SAAS`, `ENTERPRISE`) maintains cross-SDK parity.
- `EnterpriseMiddlewareConfig` provides fine-grained control over the middleware stack.

## Next Chapter

[Chapter 7: Event Bus](07_event_bus.md) -- Deep dive into the EventBus for event-driven patterns, including publishing, subscribing, filtering, and history management.
