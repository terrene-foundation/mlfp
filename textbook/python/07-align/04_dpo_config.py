# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Align / DPO Configuration
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Configure Direct Preference Optimization
# LEVEL: Intermediate
# PARITY: Full — Rust has equivalent DpoConfig struct
# VALIDATES: DPOConfig (beta, learning_rate, max_length, max_prompt_length)
#
# Run: uv run python textbook/python/07-align/04_dpo_config.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os

from kailash_align import AlignmentConfig, DPOConfig
from kailash_align.method_registry import METHOD_REGISTRY

# ── 1. Default DPOConfig ───────────────────────────────────────────

default = DPOConfig()

assert default.num_train_epochs == 1
assert default.per_device_train_batch_size == 4
assert default.gradient_accumulation_steps == 4
assert default.learning_rate == 5e-5
assert default.warmup_ratio == 0.1
assert default.max_length == 2048
assert default.max_prompt_length == 512
assert default.beta == 0.1
assert default.logging_steps == 10
assert default.save_steps == 100
assert default.gradient_checkpointing is True
assert default.bf16 is True
assert default.fp16 is False

# ── 2. Beta parameter — controls deviation from reference policy ────
# Lower beta = more deviation allowed (more aggressive optimization)
# Higher beta = stay closer to reference policy (more conservative)

conservative = DPOConfig(beta=0.5)
assert conservative.beta == 0.5

aggressive = DPOConfig(beta=0.01)
assert aggressive.beta == 0.01

# ── 3. Beta validation — must be positive and finite ────────────────

try:
    DPOConfig(beta=0.0)
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: beta must be positive

try:
    DPOConfig(beta=-0.1)
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: beta must be positive

try:
    DPOConfig(beta=float("nan"))
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: NaN rejected

try:
    DPOConfig(beta=float("inf"))
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: Inf rejected

# ── 4. Learning rate validation ─────────────────────────────────────

try:
    DPOConfig(learning_rate=0.0)
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: learning_rate must be positive

try:
    DPOConfig(learning_rate=float("inf"))
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: Inf rejected

# ── 5. warmup_ratio validation ──────────────────────────────────────

try:
    DPOConfig(warmup_ratio=1.0)
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: warmup_ratio must be in [0, 1)

try:
    DPOConfig(warmup_ratio=-0.1)
    assert False, "Should have raised ValueError"
except ValueError:
    pass

# ── 6. bf16/fp16 mutual exclusion ──────────────────────────────────

try:
    DPOConfig(bf16=True, fp16=True)
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "bf16" in str(e) and "fp16" in str(e)

# ── 7. Frozen dataclass ────────────────────────────────────────────

try:
    default.beta = 0.5  # type: ignore[misc]
    assert False, "Should have raised FrozenInstanceError"
except AttributeError:
    pass  # Expected: frozen dataclass

# ── 8. max_length and max_prompt_length relationship ────────────────
# max_prompt_length should be less than max_length to leave room for
# chosen/rejected completions.

cfg = DPOConfig(max_length=2048, max_prompt_length=512)
assert (
    cfg.max_prompt_length < cfg.max_length
), "Prompt length should be less than total max length"
completion_budget = cfg.max_length - cfg.max_prompt_length
assert completion_budget == 1536, "Completion budget = max_length - max_prompt_length"

# ── 9. DPO in AlignmentConfig with loss_type variants ──────────────
# kailash-align supports DPO loss variants via loss_type on AlignmentConfig.

model_id = os.environ.get("DEFAULT_LLM_MODEL", "meta-llama/Llama-3.1-8B")

# Standard DPO
standard_dpo = AlignmentConfig(
    method="dpo",
    base_model_id=model_id,
    dpo=DPOConfig(beta=0.1),
)
assert standard_dpo.loss_type is None, "Default: standard DPO loss"

# IPO variant (Identity Preference Optimization)
ipo_config = AlignmentConfig(
    method="dpo",
    base_model_id=model_id,
    dpo=DPOConfig(beta=0.1),
    loss_type="ipo",
)
assert ipo_config.loss_type == "ipo"

# SimPO variant
simpo_config = AlignmentConfig(
    method="dpo",
    base_model_id=model_id,
    loss_type="simpo",
)
assert simpo_config.loss_type == "simpo"

# ── 10. DPO method registry metadata ───────────────────────────────

dpo_method = METHOD_REGISTRY["dpo"]
assert dpo_method.requires_preference_data is True
assert dpo_method.supports_loss_type is True
assert dpo_method.requires_reward_func is False
assert dpo_method.category == "offline"
assert "prompt" in dpo_method.dataset_required_columns
assert "chosen" in dpo_method.dataset_required_columns
assert "rejected" in dpo_method.dataset_required_columns

# ── 11. sft_then_dpo combo uses both SFT and DPO configs ───────────

from kailash_align import SFTConfig

combo = AlignmentConfig(
    method="sft_then_dpo",
    base_model_id=model_id,
    sft=SFTConfig(num_train_epochs=2, learning_rate=2e-4),
    dpo=DPOConfig(num_train_epochs=1, beta=0.1),
)
assert combo.sft.num_train_epochs == 2
assert combo.dpo.beta == 0.1

print("PASS: 07-align/04_dpo_config")
