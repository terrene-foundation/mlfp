# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Align / AlignmentConfig Basics
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Create alignment configurations for different methods
# LEVEL: Basic
# PARITY: Full — Rust has equivalent AlignmentConfig struct
# VALIDATES: AlignmentConfig, method selection, base config, validate()
#
# Run: uv run python textbook/python/07-align/01_alignment_config.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from kailash_align import AlignmentConfig
from kailash_align.config import LoRAConfig, SFTConfig, DPOConfig
from kailash_align.method_registry import METHOD_REGISTRY

# ── 1. Default AlignmentConfig requires a base_model_id ─────────────

model_id = os.environ.get("DEFAULT_LLM_MODEL", "meta-llama/Llama-3.1-8B")

config = AlignmentConfig(
    method="sft",
    base_model_id=model_id,
)

assert config.method == "sft"
assert config.base_model_id == model_id
assert isinstance(config.lora, LoRAConfig), "LoRA config auto-populated"
assert isinstance(config.sft, SFTConfig), "SFT config auto-populated"

# ── 2. Missing base_model_id raises ValueError ─────────────────────

try:
    AlignmentConfig(method="sft", base_model_id="")
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: base_model_id is required

# ── 3. Invalid method raises ValueError ─────────────────────────────

try:
    AlignmentConfig(method="nonexistent_method", base_model_id=model_id)
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: unknown training method

# ── 4. All registered methods are accepted ──────────────────────────
# METHOD_REGISTRY contains all 12 built-in methods.

registered_methods = sorted(METHOD_REGISTRY.keys())
assert (
    len(registered_methods) >= 11
), f"Expected at least 11 methods, got {len(registered_methods)}"

for method_name in ["sft", "dpo", "kto", "orpo", "grpo", "rloo", "online_dpo"]:
    cfg = AlignmentConfig(method=method_name, base_model_id=model_id)
    assert cfg.method == method_name

# ── 5. The special 'sft_then_dpo' combo method ─────────────────────
# sft_then_dpo is not in METHOD_REGISTRY but is valid.

combo_config = AlignmentConfig(
    method="sft_then_dpo",
    base_model_id=model_id,
)
assert combo_config.method == "sft_then_dpo"
assert isinstance(combo_config.sft, SFTConfig)
assert isinstance(combo_config.dpo, DPOConfig)

# ── 6. validate() returns warnings for missing configs ──────────────

grpo_no_reward = AlignmentConfig(
    method="grpo",
    base_model_id=model_id,
)
warnings = grpo_no_reward.validate()
assert any(
    "reward_funcs" in w for w in warnings
), "GRPO without reward_funcs should produce a warning"

# ── 7. validate() returns empty list when config is complete ────────

sft_config = AlignmentConfig(
    method="sft",
    base_model_id=model_id,
)
warnings = sft_config.validate()
assert warnings == [], f"SFT config should be valid, got warnings: {warnings}"

# ── 8. Method-specific configs auto-created on demand ───────────────
# When method="kto" but kto=None, __post_init__ creates a default KTOConfig.

kto_config = AlignmentConfig(method="kto", base_model_id=model_id)
assert kto_config.kto is not None, "KTO config auto-created"

orpo_config = AlignmentConfig(method="orpo", base_model_id=model_id)
assert orpo_config.orpo is not None, "ORPO config auto-created"

# ── 9. get_method_config() dispatches to the right sub-config ───────

cfg = AlignmentConfig(
    method="dpo",
    base_model_id=model_id,
    dpo=DPOConfig(beta=0.2, learning_rate=3e-5),
)
dpo_sub = cfg.get_method_config("dpo")
assert dpo_sub is cfg.dpo
assert dpo_sub.beta == 0.2

# For unknown methods, falls back to SFT config
sft_fallback = cfg.get_method_config("cpo")
assert sft_fallback is cfg.sft, "Experimental methods fall back to SFT config"

# ── 10. QLoRA flag (cannot test import without bitsandbytes) ────────

config_no_qlora = AlignmentConfig(
    method="sft",
    base_model_id=model_id,
    use_qlora=False,
)
assert config_no_qlora.use_qlora is False

# ── 11. experiment_dir and local_files_only ─────────────────────────

config_opts = AlignmentConfig(
    method="sft",
    base_model_id=model_id,
    experiment_dir="/tmp/my-experiment",
    local_files_only=True,
    base_model_revision="abc123",
)
assert config_opts.experiment_dir == "/tmp/my-experiment"
assert config_opts.local_files_only is True
assert config_opts.base_model_revision == "abc123"

print("PASS: 07-align/01_alignment_config")
