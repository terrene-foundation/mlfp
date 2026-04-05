# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Kaizen / Agent Configuration
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Configure agents with KaizenConfig and AgentManager
# LEVEL: Intermediate
# PARITY: Full — Rust has AgentConfig with same progressive-disclosure
# VALIDATES: KaizenConfig, configure(), AgentManager
#
# Run: uv run python textbook/python/03-kaizen/03_agent_config.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import kaizen
from kaizen import KaizenConfig, AgentManager

# ── 1. KaizenConfig — framework-wide settings ───────────────────────
# KaizenConfig stores global settings for the Kaizen framework.
# These apply to all new Kaizen instances unless overridden.

config = KaizenConfig()
assert isinstance(config, KaizenConfig)

# ── 2. Global configuration via kaizen.configure() ──────────────────
# configure() sets framework-wide defaults that apply to new instances.

kaizen.configure(
    signature_programming_enabled=True,
    mcp_integration_enabled=False,
    multi_agent_coordination=True,
    transparency_enabled=True,
)

# ── 3. AgentManager — lifecycle management ──────────────────────────
# AgentManager tracks agent instances, their state, and provides
# discovery/shutdown capabilities.

manager = AgentManager()
assert isinstance(manager, AgentManager)

# ── 4. Load config from environment variables ───────────────────────
# kaizen.load_config_from_env() reads KAIZEN_* environment variables.
# e.g., KAIZEN_SIGNATURE_PROGRAMMING_ENABLED=true

env_config = kaizen.load_config_from_env(prefix="KAIZEN_")
assert isinstance(env_config, dict), "Returns dict of loaded config"

# ── 5. Kaizen framework instance ────────────────────────────────────
# The Kaizen class is the framework entry point.
# Framework = Kaizen (backward compatibility alias)

from kaizen import Kaizen, Framework

assert Kaizen is Framework, "Framework is alias for Kaizen"

framework = Kaizen()
assert isinstance(framework, Kaizen)

print("PASS: 03-kaizen/03_agent_config")
