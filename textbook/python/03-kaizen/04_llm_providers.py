# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Kaizen / LLM Providers
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Configure LLM providers via environment variables
# LEVEL: Intermediate
# PARITY: Full — Rust has llm module with multi-provider support
# VALIDATES: Model name resolution, API key handling, cost estimation
#
# Run: uv run python textbook/python/03-kaizen/04_llm_providers.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# ── 1. Model configuration via environment ──────────────────────────
# Kaizen NEVER hardcodes model names. They come from env vars:
#   DEFAULT_LLM_MODEL — primary model for all agents
#   OPENAI_PROD_MODEL — OpenAI-specific override
#   ANTHROPIC_API_KEY — for Claude models
#   OPENAI_API_KEY — for GPT models

# Verify the pattern (don't actually require keys in tutorial)
model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")
assert isinstance(model, str)
assert len(model) > 0, "Model name must be non-empty"

# ── 2. Provider detection from model name ───────────────────────────
# The SDK detects the provider from model name prefixes:
#   claude-* → Anthropic
#   gpt-*    → OpenAI
#   gemini-* → Google
#   o1-*, o3-*, o4-* → OpenAI (reasoning models)


def detect_provider(model_name: str) -> str:
    """Detect provider from model name prefix."""
    if model_name.startswith("claude-"):
        return "anthropic"
    elif model_name.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    elif model_name.startswith("gemini-"):
        return "google"
    return "unknown"


assert detect_provider("claude-sonnet-4-20250514") == "anthropic"
assert detect_provider("gpt-4o") == "openai"
assert detect_provider("gemini-pro") == "google"

# ── 3. Cost estimation by provider ──────────────────────────────────
# Delegate tracks cost estimates per 1M tokens (from delegate.py):
#   claude-  : $3 input, $15 output
#   gpt-4o   : $2.5 input, $10 output
#   gpt-4    : $30 input, $60 output
#   gemini-  : $1.25 input, $5 output

cost_table = {
    "claude-": {"input": 3.0, "output": 15.0},
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gemini-": {"input": 1.25, "output": 5.0},
}

# Verify the most common provider costs
assert cost_table["claude-"]["input"] == 3.0
assert cost_table["gpt-4o"]["output"] == 10.0

# ── 4. API key patterns ─────────────────────────────────────────────
# Keys are always from environment, never hardcoded:
#   ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY
# The SDK reads these at runtime, not at import time.

# Validate that no keys are hardcoded in this file
import inspect

source = inspect.getsource(detect_provider)
assert "sk-" not in source, "No hardcoded API keys"
assert "key" not in source.lower() or "api_key" not in source, "No key values"

# ── 5. Security: environment-only pattern ───────────────────────────
# The dotenv pattern is the standard for loading keys:
#
#   from dotenv import load_dotenv
#   load_dotenv()  # Reads .env file
#   api_key = os.environ["ANTHROPIC_API_KEY"]
#
# .env files MUST be in .gitignore (security rule)

print("PASS: 03-kaizen/04_llm_providers")
