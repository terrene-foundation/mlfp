# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Align / SFT Configuration
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Configure Supervised Fine-Tuning
# LEVEL: Intermediate
# PARITY: Full — Rust has equivalent SftConfig struct
# VALIDATES: SFTConfig (learning rate, epochs, batch size, dataset format)
#
# Run: uv run python textbook/python/07-align/03_sft_config.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from kailash_align import SFTConfig

# ── 1. Default SFTConfig ───────────────────────────────────────────

default = SFTConfig()

assert default.num_train_epochs == 3
assert default.per_device_train_batch_size == 4
assert default.gradient_accumulation_steps == 4
assert default.learning_rate == 2e-4
assert default.warmup_ratio == 0.03
assert default.max_seq_length == 2048
assert default.logging_steps == 10
assert default.save_steps == 100
assert default.gradient_checkpointing is True
assert default.bf16 is True
assert default.fp16 is False
assert default.dataset_text_field == "text"

# ── 2. Custom SFT configuration ───────────────────────────────────

custom = SFTConfig(
    num_train_epochs=5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_ratio=0.1,
    max_seq_length=4096,
    logging_steps=5,
    save_steps=200,
    gradient_checkpointing=False,
    bf16=False,
    fp16=True,
    dataset_text_field="content",
)

assert custom.num_train_epochs == 5
assert custom.per_device_train_batch_size == 8
assert custom.learning_rate == 1e-4
assert custom.max_seq_length == 4096
assert custom.fp16 is True
assert custom.bf16 is False
assert custom.dataset_text_field == "content"

# ── 3. Frozen dataclass — immutable after creation ──────────────────

try:
    default.learning_rate = 1e-3  # type: ignore[misc]
    assert False, "Should have raised FrozenInstanceError"
except AttributeError:
    pass  # Expected: frozen dataclass

# ── 4. Learning rate validation — must be positive and finite ───────

try:
    SFTConfig(learning_rate=0.0)
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: learning_rate must be positive

try:
    SFTConfig(learning_rate=-1e-4)
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: learning_rate must be positive

try:
    SFTConfig(learning_rate=float("nan"))
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: NaN rejected

try:
    SFTConfig(learning_rate=float("inf"))
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: Inf rejected

# ── 5. warmup_ratio validation — must be in [0, 1) ─────────────────

zero_warmup = SFTConfig(warmup_ratio=0.0)
assert zero_warmup.warmup_ratio == 0.0, "Zero warmup is valid"

try:
    SFTConfig(warmup_ratio=1.0)
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: warmup_ratio must be in [0, 1)

try:
    SFTConfig(warmup_ratio=-0.1)
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: warmup_ratio must be >= 0

try:
    SFTConfig(warmup_ratio=float("nan"))
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: NaN rejected

# ── 6. bf16 and fp16 mutual exclusion ──────────────────────────────

try:
    SFTConfig(bf16=True, fp16=True)
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "bf16" in str(e) and "fp16" in str(e)

# Both False is valid (uses float32)
both_off = SFTConfig(bf16=False, fp16=False)
assert both_off.bf16 is False
assert both_off.fp16 is False

# ── 7. Effective batch size calculation ─────────────────────────────
# effective_batch = per_device * gradient_accumulation * num_gpus
# This is a useful concept students should understand.

cfg = SFTConfig(per_device_train_batch_size=4, gradient_accumulation_steps=8)
effective_single_gpu = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
assert effective_single_gpu == 32, "4 * 8 = 32 effective batch size on one GPU"

# ── 8. Common SFT configurations ───────────────────────────────────

# Memory-constrained: small batch, high accumulation
constrained = SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    max_seq_length=1024,
)
assert constrained.per_device_train_batch_size == 1
assert constrained.gradient_checkpointing is True

# Speed-focused: large batch, less accumulation
fast = SFTConfig(
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
)
assert fast.per_device_train_batch_size == 16

print("PASS: 07-align/03_sft_config")
