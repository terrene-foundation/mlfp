# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Align / LoRA Configuration
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Configure LoRA adapters (rank, alpha, target modules)
# LEVEL: Intermediate
# PARITY: Full — Rust has equivalent LoraConfig struct
# VALIDATES: LoRAConfig fields (rank, alpha, target_modules, dropout, bias)
#
# Run: uv run python textbook/python/07-align/02_lora_config.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math

from kailash_align import LoRAConfig

# ── 1. Default LoRAConfig ───────────────────────────────────────────

default = LoRAConfig()

assert default.rank == 16, f"Default rank should be 16, got {default.rank}"
assert default.alpha == 32, f"Default alpha should be 32, got {default.alpha}"
assert default.target_modules == ("q_proj", "v_proj", "k_proj", "o_proj")
assert default.dropout == 0.05
assert default.bias == "none"
assert default.task_type == "CAUSAL_LM"

# ── 2. Custom LoRA configuration ───────────────────────────────────

custom = LoRAConfig(
    rank=64,
    alpha=128,
    target_modules=("q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj"),
    dropout=0.1,
    bias="lora_only",
)

assert custom.rank == 64
assert custom.alpha == 128
assert len(custom.target_modules) == 6
assert "gate_proj" in custom.target_modules
assert custom.dropout == 0.1
assert custom.bias == "lora_only"

# ── 3. Frozen dataclass — immutable after creation ──────────────────

try:
    default.rank = 32  # type: ignore[misc]
    assert False, "Should have raised FrozenInstanceError"
except AttributeError:
    pass  # Expected: frozen dataclass cannot be modified

# ── 4. Rank validation — must be >= 1 ──────────────────────────────

try:
    LoRAConfig(rank=0)
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "rank" in str(e)

try:
    LoRAConfig(rank=-1)
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "rank" in str(e)

# ── 5. Alpha validation — must be >= 1 ─────────────────────────────

try:
    LoRAConfig(alpha=0)
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "alpha" in str(e)

# ── 6. Dropout validation — must be in [0, 1) ──────────────────────

zero_dropout = LoRAConfig(dropout=0.0)
assert zero_dropout.dropout == 0.0, "Zero dropout is valid"

try:
    LoRAConfig(dropout=1.0)
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "dropout" in str(e)

try:
    LoRAConfig(dropout=-0.1)
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "dropout" in str(e)

# ── 7. NaN/Inf rejection on dropout ────────────────────────────────

try:
    LoRAConfig(dropout=float("nan"))
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: NaN rejected

try:
    LoRAConfig(dropout=float("inf"))
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: Inf rejected

# ── 8. target_modules must not be empty ─────────────────────────────

try:
    LoRAConfig(target_modules=())
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "target_modules" in str(e)

# ── 9. Bias validation — restricted to three values ─────────────────

for valid_bias in ("none", "all", "lora_only"):
    cfg = LoRAConfig(bias=valid_bias)
    assert cfg.bias == valid_bias

try:
    LoRAConfig(bias="invalid")
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "bias" in str(e)

# ── 10. Common LoRA patterns ────────────────────────────────────────
# Low-rank: fewer parameters, faster training, less capacity
low_rank = LoRAConfig(rank=4, alpha=8)
assert low_rank.rank == 4
assert low_rank.alpha == 8

# High-rank: more parameters, slower training, more capacity
high_rank = LoRAConfig(rank=128, alpha=256)
assert high_rank.rank == 128

# Typical convention: alpha = 2 * rank
assert default.alpha == default.rank * 2, "Convention: alpha = 2 * rank"

# ── 11. NaN/Inf rejection on rank and alpha ─────────────────────────

try:
    LoRAConfig(rank=int(float("inf")))
    assert False, "Should have raised OverflowError or ValueError"
except (ValueError, OverflowError):
    pass

print("PASS: 07-align/02_lora_config")
