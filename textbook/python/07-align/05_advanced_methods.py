# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Align / Advanced Alignment Methods
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Configure advanced alignment methods (KTO, ORPO, GRPO, RLOO,
#            OnlineDPO)
# LEVEL: Advanced
# PARITY: Full — Rust has equivalent config structs
# VALIDATES: KTOConfig, ORPOConfig, GRPOConfig, RLOOConfig, OnlineDPOConfig
#
# Run: uv run python textbook/python/07-align/05_advanced_methods.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os

from kailash_align import (
    AlignmentConfig,
    GRPOConfig,
    KTOConfig,
    OnlineDPOConfig,
    ORPOConfig,
    RLOOConfig,
)
from kailash_align.method_registry import METHOD_REGISTRY

model_id = os.environ.get("DEFAULT_LLM_MODEL", "meta-llama/Llama-3.1-8B")

# ── 1. KTOConfig — Unpaired binary feedback ────────────────────────
# KTO works with (prompt, completion, label) triples instead of
# (prompt, chosen, rejected) pairs. Dramatically lowers the data barrier.

kto = KTOConfig()

assert kto.learning_rate == 5e-7, "KTO paper recommends very low LR"
assert kto.beta == 0.1
assert kto.desirable_weight == 1.0
assert kto.undesirable_weight == 1.0
assert kto.max_length == 1024
assert kto.max_prompt_length == 512

# Custom KTO with asymmetric weighting
kto_custom = KTOConfig(
    desirable_weight=1.5,
    undesirable_weight=0.5,
    learning_rate=1e-6,
    beta=0.05,
)
assert kto_custom.desirable_weight == 1.5
assert kto_custom.undesirable_weight == 0.5

# Weight validation — must be positive
try:
    KTOConfig(desirable_weight=0.0)
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: must be positive

try:
    KTOConfig(undesirable_weight=-1.0)
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: must be positive

# KTO method metadata
kto_method = METHOD_REGISTRY["kto"]
assert kto_method.category == "unpaired"
assert kto_method.requires_preference_data is False
assert kto_method.requires_reward_func is False
assert "prompt" in kto_method.dataset_required_columns
assert "completion" in kto_method.dataset_required_columns
assert "label" in kto_method.dataset_required_columns

# ── 2. ORPOConfig — Monolithic SFT + preference ────────────────────
# ORPO combines SFT and preference alignment in one pass,
# eliminating the need for sft_then_dpo two-stage pipeline.

orpo = ORPOConfig()

assert orpo.learning_rate == 8e-6, "ORPO paper recommends 8e-6"
assert orpo.beta == 0.1
assert orpo.max_length == 1024
assert orpo.max_prompt_length == 512

# ORPO method metadata
orpo_method = METHOD_REGISTRY["orpo"]
assert orpo_method.category == "monolithic"
assert orpo_method.requires_preference_data is True
assert "prompt" in orpo_method.dataset_required_columns
assert "chosen" in orpo_method.dataset_required_columns
assert "rejected" in orpo_method.dataset_required_columns

# ── 3. GRPOConfig — Online RL (DeepSeek-R1 method) ─────────────────
# GRPO generates completions online and scores them with reward functions.
# Requires reward functions from RewardRegistry.

grpo = GRPOConfig()

assert grpo.num_generations == 4, "Default: 4 completions per prompt"
assert grpo.temperature == 0.7
assert grpo.max_completion_length == 2048
assert grpo.learning_rate == 1e-5
assert grpo.kl_coef == 0.001
assert grpo.use_vllm is False
assert grpo.vllm_gpu_utilization == 0.5

# Custom GRPO for single-GPU training
grpo_custom = GRPOConfig(
    num_generations=8,
    temperature=1.0,
    kl_coef=0.01,
    max_completion_length=1024,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
)
assert grpo_custom.num_generations == 8
assert grpo_custom.temperature == 1.0
assert grpo_custom.kl_coef == 0.01

# num_generations validation
try:
    GRPOConfig(num_generations=0)
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: must be >= 1

# temperature validation — must be > 0
try:
    GRPOConfig(temperature=0.0)
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: must be > 0

# kl_coef validation — must be >= 0 and finite
try:
    GRPOConfig(kl_coef=-0.1)
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: must be >= 0

zero_kl = GRPOConfig(kl_coef=0.0)
assert zero_kl.kl_coef == 0.0, "Zero KL coef is valid (no penalty)"

# vllm_gpu_utilization validation — must be in (0, 1]
try:
    GRPOConfig(vllm_gpu_utilization=0.0)
    assert False, "Should have raised ValueError"
except ValueError:
    pass

try:
    GRPOConfig(vllm_gpu_utilization=1.5)
    assert False, "Should have raised ValueError"
except ValueError:
    pass

# GRPO method metadata
grpo_method = METHOD_REGISTRY["grpo"]
assert grpo_method.category == "online"
assert grpo_method.requires_reward_func is True
assert grpo_method.requires_generation_backend is True
assert "prompt" in grpo_method.dataset_required_columns

# ── 4. RLOOConfig — REINFORCE Leave-One-Out ────────────────────────
# RLOO uses a leave-one-out baseline for variance reduction.
# Same infrastructure as GRPO but different optimization.

rloo = RLOOConfig()

assert rloo.num_generations == 4
assert rloo.temperature == 0.7
assert rloo.kl_coef == 0.001
assert rloo.use_vllm is False

# RLOO shares the same validation rules as GRPO
try:
    RLOOConfig(num_generations=0)
    assert False, "Should have raised ValueError"
except ValueError:
    pass

try:
    RLOOConfig(temperature=0.0)
    assert False, "Should have raised ValueError"
except ValueError:
    pass

# RLOO method metadata
rloo_method = METHOD_REGISTRY["rloo"]
assert rloo_method.category == "online"
assert rloo_method.requires_reward_func is True

# ── 5. OnlineDPOConfig — DPO with online generation ────────────────
# Online DPO generates completions online and uses a reward model to
# score pairs, then applies DPO loss.

online_dpo = OnlineDPOConfig()

assert online_dpo.beta == 0.1
assert online_dpo.max_length == 2048
assert online_dpo.max_prompt_length == 512
assert online_dpo.max_completion_length == 512
assert online_dpo.use_vllm is False

# Custom OnlineDPO
online_dpo_custom = OnlineDPOConfig(
    beta=0.2,
    max_completion_length=1024,
    learning_rate=3e-5,
)
assert online_dpo_custom.beta == 0.2
assert online_dpo_custom.max_completion_length == 1024

# Beta validation (same as DPO)
try:
    OnlineDPOConfig(beta=0.0)
    assert False, "Should have raised ValueError"
except ValueError:
    pass

# OnlineDPO method metadata — does NOT require reward_func
online_dpo_method = METHOD_REGISTRY["online_dpo"]
assert online_dpo_method.category == "online"
assert online_dpo_method.requires_reward_func is False
assert online_dpo_method.requires_generation_backend is True

# ── 6. bf16/fp16 mutual exclusion on all configs ───────────────────

for ConfigClass in (KTOConfig, ORPOConfig, GRPOConfig, RLOOConfig, OnlineDPOConfig):
    try:
        ConfigClass(bf16=True, fp16=True)
        assert False, f"{ConfigClass.__name__} should reject bf16+fp16"
    except ValueError as e:
        assert "bf16" in str(e) and "fp16" in str(e)

# ── 7. All configs are frozen dataclasses ───────────────────────────

for config_obj in (kto, orpo, grpo, rloo, online_dpo):
    try:
        config_obj.learning_rate = 999.0  # type: ignore[misc]
        assert False, f"{type(config_obj).__name__} should be frozen"
    except AttributeError:
        pass  # Expected: all frozen

# ── 8. Auto-creation in AlignmentConfig ─────────────────────────────
# When method-specific config is None, AlignmentConfig creates defaults.

for method in ("kto", "orpo", "grpo", "rloo", "online_dpo"):
    cfg = AlignmentConfig(method=method, base_model_id=model_id)
    sub = getattr(cfg, method)
    assert sub is not None, f"{method} config should be auto-created"

print("PASS: 07-align/05_advanced_methods")
