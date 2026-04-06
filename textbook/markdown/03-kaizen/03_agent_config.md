# Chapter 3: Agent Configuration

## Overview

Kaizen provides framework-wide configuration through `KaizenConfig` and `kaizen.configure()`, lifecycle management via `AgentManager`, and environment-based configuration loading. This chapter covers the global configuration system, the `Kaizen`/`Framework` entry point class, and how to load settings from environment variables.

## Prerequisites

- Completed Chapters 1-2 (Signatures, Delegate)
- Understanding of Python module-level configuration patterns

## Concepts

### Framework-Wide Configuration

`kaizen.configure()` sets global defaults that apply to all new Kaizen instances. This is the central place to enable or disable framework features like signature programming, MCP integration, multi-agent coordination, and transparency.

### AgentManager

`AgentManager` tracks agent instances, their state, and provides discovery and shutdown capabilities. It acts as a registry for all agents created within a session.

### The Kaizen Class

`Kaizen` (aliased as `Framework`) is the framework entry point. It provides the runtime context for agent execution.

## Key API

| Method / Property                     | Parameters                | Returns        | Description                     |
| ------------------------------------- | ------------------------- | -------------- | ------------------------------- |
| `KaizenConfig()`                      | --                        | `KaizenConfig` | Framework-wide settings object  |
| `kaizen.configure(**kwargs)`          | feature flags             | `None`         | Set global defaults             |
| `AgentManager()`                      | --                        | `AgentManager` | Lifecycle management for agents |
| `kaizen.load_config_from_env(prefix)` | `prefix: str = "KAIZEN_"` | `dict`         | Load config from env vars       |
| `Kaizen()` / `Framework()`            | --                        | `Kaizen`       | Framework entry point           |

## Code Walkthrough

### Step 1: KaizenConfig

```python
import kaizen
from kaizen import KaizenConfig, AgentManager

config = KaizenConfig()
assert isinstance(config, KaizenConfig)
```

### Step 2: Global Configuration

```python
kaizen.configure(
    signature_programming_enabled=True,
    mcp_integration_enabled=False,
    multi_agent_coordination=True,
    transparency_enabled=True,
)
```

These settings apply to all new Kaizen instances unless overridden at the instance level.

### Step 3: AgentManager

```python
manager = AgentManager()
assert isinstance(manager, AgentManager)
```

AgentManager tracks agent instances and provides discovery/shutdown capabilities.

### Step 4: Environment-Based Configuration

```python
env_config = kaizen.load_config_from_env(prefix="KAIZEN_")
assert isinstance(env_config, dict)
```

Reads `KAIZEN_*` environment variables (e.g., `KAIZEN_SIGNATURE_PROGRAMMING_ENABLED=true`).

### Step 5: Framework Entry Point

```python
from kaizen import Kaizen, Framework

assert Kaizen is Framework  # Framework is alias for Kaizen

framework = Kaizen()
assert isinstance(framework, Kaizen)
```

## Common Mistakes

| Mistake                                     | Problem                            | Fix                                       |
| ------------------------------------------- | ---------------------------------- | ----------------------------------------- |
| Calling `configure()` after creating agents | Settings don't apply retroactively | Configure before creating agents          |
| Missing `KAIZEN_` prefix in env vars        | Variables not detected             | Use the `KAIZEN_` prefix consistently     |
| Creating multiple AgentManagers             | Agent tracking fragmented          | Use a single AgentManager per application |

## Exercises

1. **Feature Flags**: Call `kaizen.configure()` with different combinations of feature flags. Verify the configuration is stored correctly.

2. **Environment Loading**: Set `KAIZEN_SIGNATURE_PROGRAMMING_ENABLED=true` in your `.env` file. Load the config and verify the value is parsed as a boolean.

3. **Framework Alias**: Verify that `Kaizen` and `Framework` are the same class. Create instances of each and confirm they are of the same type.

## Key Takeaways

- `kaizen.configure()` sets framework-wide defaults for all new instances.
- `AgentManager` provides lifecycle management and agent discovery.
- `kaizen.load_config_from_env("KAIZEN_")` reads configuration from environment variables.
- `Kaizen` and `Framework` are aliases for the same class.
- Configure the framework before creating agents -- settings do not apply retroactively.

## Next Chapter

[Chapter 4: LLM Providers](04_llm_providers.md) -- Configure LLM providers via environment variables and understand provider detection.
