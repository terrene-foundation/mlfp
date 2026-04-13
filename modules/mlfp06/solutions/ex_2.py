# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 2: LLM Fine-Tuning — LoRA, Adapters, and the
#                       Technique Landscape
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Implement LoRA from scratch as an nn.Module (understand the maths:
#     W_new = W + A @ B, rank r << d)
#   - Implement adapter layers from scratch (bottleneck: FC -> act -> FC)
#   - Compare LoRA vs adapters across 4 dimensions (param efficiency,
#     implementation complexity, flexibility, modularity)
#   - Survey all 10 fine-tuning techniques and select the right one
#   - Explain model merging techniques (TIES, DARE, SLERP, task arithmetic)
#   - Describe quantisation methods (GPTQ, AWQ, GGUF, QLoRA)
#   - Use kailash-align AlignmentPipeline for SFT with LoRA
#   - Register and version adapters in AdapterRegistry
#
# PREREQUISITES:
#   Exercise 1 (LLM fundamentals, transformer architecture from M5.4).
#   Linear algebra: matrix rank, SVD (M4.3) — LoRA IS low-rank
#   factorisation applied to weight updates.
#
# ESTIMATED TIME: ~180 min
#
# TASKS:
#    1. Load SFT dataset (IMDB instruction-response pairs)
#    2. Implement LoRA layer FROM SCRATCH (nn.Module)
#    3. Implement adapter layer FROM SCRATCH (bottleneck module)
#    4. Compare LoRA vs adapter: parameter count, architecture
#    5. Fine-tuning landscape survey (all 10 techniques)
#    6. Model merging: TIES, DARE, SLERP, task arithmetic
#    7. Quantisation overview: GPTQ, AWQ, GGUF, QLoRA
#    8. AlignmentPipeline SFT training with LoRA
#    9. Register adapter in AdapterRegistry
#   10. Parameter reduction analysis across ranks
#
# DATASET: IMDB sentiment (stanfordnlp/imdb on HuggingFace)
#   25,000 real movie reviews with binary positive/negative labels.
#   Reformatted as SFT instruction-response pairs.  Subsampled to 2,000
#   for fast training.  Split: 90% train / 10% eval.
#   Base model: from env variable SFT_BASE_MODEL.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path

import polars as pl
import torch
import torch.nn as nn

from kailash_align import AdapterRegistry, AlignmentConfig, AlignmentPipeline

from shared.kailash_helpers import get_device, setup_environment

setup_environment()

device = get_device()
print(f"Compute device: {device}")


# ══════════════════════════════════════════════════════════════════════════
# TASK 1: Load SFT Dataset (IMDB sentiment from HuggingFace)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 1: Load SFT Dataset")
print("=" * 70)

CACHE_DIR = Path("data/mlfp06/imdb")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "imdb_sft_2k.parquet"

if CACHE_FILE.exists():
    print(f"Loading cached IMDB SFT pairs from {CACHE_FILE}")
    sft_data = pl.read_parquet(CACHE_FILE)
else:
    print("Downloading stanfordnlp/imdb from HuggingFace (first run)...")
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/imdb", split="train")
    ds = ds.shuffle(seed=42).select(range(min(2000, len(ds))))

    label_names = {0: "negative", 1: "positive"}
    rows = []
    for row in ds:
        review = row["text"][:1500]
        sentiment = label_names[row["label"]]
        rows.append(
            {
                "instruction": (
                    "Classify the sentiment of the following movie review as "
                    "either 'positive' or 'negative', then briefly justify "
                    f"your answer.\n\nReview: {review}"
                ),
                "response": (
                    f"Sentiment: {sentiment}. The reviewer expresses a clearly "
                    f"{sentiment} reaction to the film."
                ),
                "text": review,
                "label": sentiment,
            }
        )
    sft_data = pl.DataFrame(rows)
    sft_data.write_parquet(CACHE_FILE)
    print(f"Cached {sft_data.height} SFT pairs to {CACHE_FILE}")

print(f"Shape: {sft_data.shape}")
print(f"Columns: {sft_data.columns}")
print(f"Sample instruction:\n{sft_data['instruction'][0][:300]}...")
print(f"Sample response:\n{sft_data['response'][0]}")

n_train = int(sft_data.height * 0.9)
train_data = sft_data[:n_train]
eval_data = sft_data[n_train:]
print(f"Train: {train_data.height}, Eval: {eval_data.height}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert sft_data.height > 0, "Task 1: SFT dataset should not be empty"
assert "instruction" in sft_data.columns, "Dataset needs 'instruction' column"
assert "response" in sft_data.columns, "Dataset needs 'response' column"
print(f"✓ Checkpoint 1 passed — {sft_data.height} SFT pairs loaded\n")

# INTERPRETATION: SFT data = (instruction, response) pairs.  The model
# learns to follow instructions by maximising P(response | instruction).
# Quality matters more than quantity: 500 high-quality domain-specific
# pairs can outperform 10,000 generic pairs.


# ══════════════════════════════════════════════════════════════════════════
# TASK 2: Implement LoRA Layer FROM SCRATCH (nn.Module)
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: LoRA Layer — From-Scratch Implementation")
print("=" * 70)

print(
    """
LoRA theory (Hu et al., 2021):
  Full fine-tuning: W_new = W + delta_W   (delta_W is d × d)
  LoRA:             W_new = W + A @ B     (A is d × r, B is r × d)
  rank r << d  =>  A @ B has at most r non-zero singular values
  Equivalent: only r "directions" in weight space are updated.
  Pre-trained W remains FROZEN; only A and B are trained.
  Connects to M4.3 SVD: LoRA IS low-rank factorisation of delta_W.
"""
)


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer — from-scratch implementation.

    Computes: output = pretrained_linear(x) + (x @ A @ B) * (alpha / r)

    Args:
        in_features:  Input dimension (d_model).
        out_features: Output dimension (d_model for self-attn projections).
        rank:         LoRA rank r.  Lower = fewer params, less expressive.
        alpha:        Scaling factor.  Typical: alpha = 2 * rank.
        dropout:      Dropout on LoRA path for regularisation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # A: d_in × r  (initialised with Kaiming uniform for stable gradients)
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        # B: r × d_out (initialised to zero so LoRA starts as identity)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        """Kaiming init for A; zero init for B.

        This ensures the LoRA contribution is zero at initialisation:
        A @ B_zero = 0  =>  W_new = W + 0 = W (no change at start).
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the LoRA delta: (x @ A @ B) * scaling."""
        dropped = self.dropout(x)
        lora_out = dropped @ self.lora_A @ self.lora_B
        return lora_out * self.scaling


class LoRALinear(nn.Module):
    """A frozen linear layer augmented with a trainable LoRA path.

    Computes: y = W_frozen @ x + bias + LoRALayer(x)
    """

    def __init__(
        self,
        pretrained_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = pretrained_linear
        # Freeze the pretrained weights
        for param in self.linear.parameters():
            param.requires_grad = False

        self.lora = LoRALayer(
            in_features=pretrained_linear.in_features,
            out_features=pretrained_linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)
        lora_out = self.lora(x)
        return base_out + lora_out


# Demonstrate with a synthetic linear layer
d_model = 512
lora_rank = 8
pretrained = nn.Linear(d_model, d_model).to(device)
lora_linear = LoRALinear(pretrained, rank=lora_rank, alpha=16.0).to(device)

# Count parameters
frozen_params = sum(p.numel() for p in lora_linear.linear.parameters())
trainable_params = sum(
    p.numel() for p in lora_linear.lora.parameters() if p.requires_grad
)
x_test = torch.randn(2, 10, d_model, device=device)
y_test = lora_linear(x_test)

print(f"LoRA layer created: d={d_model}, r={lora_rank}")
print(f"  Frozen params:    {frozen_params:,}")
print(f"  Trainable params: {trainable_params:,}")
print(f"  Ratio: {trainable_params / frozen_params:.4%} of original")
print(f"  Output shape:     {y_test.shape}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert isinstance(lora_linear, LoRALinear), "Task 2: LoRALinear should be created"
assert (
    trainable_params == d_model * lora_rank + lora_rank * d_model
), f"LoRA params should be 2*d*r = {2 * d_model * lora_rank}"
assert y_test.shape == x_test.shape, "Output shape should match input shape"
print("✓ Checkpoint 2 passed — LoRA from-scratch implementation verified\n")

# INTERPRETATION: LoRA trainable params = d*r + r*d = 2*d*r.
# For d=512, r=8: 8,192 vs 262,144 full params = 3.1%.
# The B=0 initialisation means LoRA starts as identity (no change to
# the pretrained weights), then gradually learns the domain-specific delta.


# ══════════════════════════════════════════════════════════════════════════
# TASK 3: Implement Adapter Layer FROM SCRATCH (bottleneck module)
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: Adapter Layer — From-Scratch Implementation")
print("=" * 70)

print(
    """
Adapter theory (Houlsby et al., 2019):
  Insert a bottleneck module between transformer layers:
    Input (d) -> Down-project (d -> b) -> Activation -> Up-project (b -> d)
  where b << d (bottleneck dimension).
  The original transformer weights remain FROZEN.
  Only the adapter weights (down + up projections) are trained.
  Residual connection: output = adapter(x) + x
"""
)


class AdapterLayer(nn.Module):
    """Bottleneck adapter module — from-scratch implementation.

    Architecture: x -> LayerNorm -> Down(d->b) -> GELU -> Up(b->d) -> + x

    Args:
        d_model:        Input/output dimension.
        bottleneck_dim: Bottleneck dimension b.  Smaller = fewer params.
        dropout:        Dropout after up-projection.
    """

    def __init__(
        self,
        d_model: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        # Zero-init up-projection so adapter starts as identity
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.layer_norm(x)
        h = self.down_proj(h)
        h = self.activation(h)
        h = self.up_proj(h)
        h = self.dropout(h)
        return h + residual  # residual connection


class AdapterTransformerBlock(nn.Module):
    """A transformer block augmented with an adapter after the FFN.

    This wraps an existing transformer block and inserts an adapter
    without modifying the original block's weights.
    """

    def __init__(
        self,
        original_block: nn.Module,
        d_model: int,
        bottleneck_dim: int = 64,
    ):
        super().__init__()
        self.block = original_block
        # Freeze original block
        for param in self.block.parameters():
            param.requires_grad = False
        self.adapter = AdapterLayer(d_model, bottleneck_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block(x)
        h = self.adapter(h)
        return h


# Demonstrate adapter
adapter = AdapterLayer(d_model=d_model, bottleneck_dim=64).to(device)
adapter_params = sum(p.numel() for p in adapter.parameters())
y_adapter = adapter(x_test)

print(f"Adapter layer: d={d_model}, bottleneck=64")
print(f"  Adapter params: {adapter_params:,}")
print(f"  Output shape:   {y_adapter.shape}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert isinstance(adapter, AdapterLayer), "Task 3: AdapterLayer should be created"
assert y_adapter.shape == x_test.shape, "Adapter output shape should match input"
expected_adapter_params = (
    d_model  # LayerNorm weight
    + d_model  # LayerNorm bias
    + d_model * 64
    + 64  # down_proj
    + 64 * d_model
    + d_model  # up_proj
)
assert (
    adapter_params == expected_adapter_params
), f"Adapter params mismatch: got {adapter_params}, expected {expected_adapter_params}"
print("✓ Checkpoint 3 passed — adapter from-scratch implementation verified\n")

# INTERPRETATION: Adapter params = 2*d*b + 2*b + 2*d (down, up, norms).
# For d=512, b=64: ~66K params.  More than LoRA r=8 (~8K) but the
# bottleneck gives a nonlinear transformation (GELU), which can model
# more complex adaptations per parameter.


# ══════════════════════════════════════════════════════════════════════════
# TASK 4: Compare LoRA vs Adapter
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 4: LoRA vs Adapter Comparison")
print("=" * 70)

# Systematic comparison across 4 dimensions
comparison_data = {
    "Dimension": [
        "Parameter Update Mechanism",
        "Parameter Efficiency",
        "Implementation Complexity",
        "Flexibility & Modularity",
    ],
    "LoRA": [
        "Low-rank matrices A,B added to weight matrices",
        f"2*d*r params per target module (d={d_model}, r={lora_rank}: {trainable_params:,})",
        "Simple: 2 matrices per target, no arch change",
        "Swap A,B at inference; merge into W for zero overhead",
    ],
    "Adapter": [
        "Bottleneck FC->act->FC inserted between layers",
        f"2*d*b + norms per adapter (d={d_model}, b=64: {adapter_params:,})",
        "Moderate: new layers, residual connections, init strategy",
        "Stack adapters for multi-task; cannot merge into W",
    ],
}

comparison_df = pl.DataFrame(comparison_data)
print(comparison_df)

# Parameter efficiency comparison across configurations
print("\n--- Parameter Efficiency Sweep ---")
configs = [
    ("LoRA r=4", 2 * d_model * 4),
    ("LoRA r=8", 2 * d_model * 8),
    ("LoRA r=16", 2 * d_model * 16),
    ("LoRA r=32", 2 * d_model * 32),
    ("Adapter b=32", 2 * d_model * 32 + 2 * 32 + 2 * d_model),
    ("Adapter b=64", 2 * d_model * 64 + 2 * 64 + 2 * d_model),
    ("Adapter b=128", 2 * d_model * 128 + 2 * 128 + 2 * d_model),
    ("Full fine-tune", d_model * d_model),
]
full_params = d_model * d_model
for name, params in configs:
    pct = params / full_params * 100
    print(f"  {name:<18} {params:>10,} params  ({pct:>6.2f}% of full)")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert comparison_df.height == 4, "Task 4: comparison should cover 4 dimensions"
print("\n✓ Checkpoint 4 passed — LoRA vs adapter comparison complete\n")

# INTERPRETATION: LoRA wins on parameter efficiency and inference speed
# (merge A,B into W for zero additional latency).  Adapters win on
# expressiveness (nonlinear bottleneck) and multi-task stacking.
# In practice, LoRA dominates for single-task adaptation; adapters
# are used when multiple tasks share a base model simultaneously.


# ══════════════════════════════════════════════════════════════════════════
# TASK 5: Fine-Tuning Landscape Survey (all 10 techniques)
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 5: Fine-Tuning Technique Landscape")
print("=" * 70)

techniques = [
    {
        "name": "LoRA",
        "mechanism": "Low-rank A,B matrices on attention projections",
        "params": "0.1-5% of full",
        "when": "Standard choice for most task adaptation",
    },
    {
        "name": "Adapter Layers",
        "mechanism": "Bottleneck FC->act->FC between transformer layers",
        "params": "1-10% of full",
        "when": "Multi-task serving with shared base model",
    },
    {
        "name": "Prefix Tuning",
        "mechanism": "Learnable vectors prepended to K,V in attention",
        "params": "<1% of full",
        "when": "Very parameter-efficient; limited expressiveness",
    },
    {
        "name": "Prompt Tuning",
        "mechanism": "Learnable soft tokens added to input embeddings",
        "params": "<0.1% of full",
        "when": "Minimal overhead; good for classification tasks",
    },
    {
        "name": "Full Fine-Tuning",
        "mechanism": "Backprop through all parameters with LR schedule",
        "params": "100% (all params)",
        "when": "Maximum quality; requires large GPU memory",
    },
    {
        "name": "LLRD (Layer-wise LR Decay)",
        "mechanism": "Lower learning rate for earlier layers, higher for later",
        "params": "100% (all params, different LRs)",
        "when": "Preserve general knowledge in early layers",
    },
    {
        "name": "Progressive Layer Freezing",
        "mechanism": "Start with all frozen; unfreeze top-down over epochs",
        "params": "Varies (10-100%)",
        "when": "Small datasets where overfitting is a risk",
    },
    {
        "name": "Knowledge Distillation",
        "mechanism": "Teacher-student: train small model on soft labels from large",
        "params": "100% of student model",
        "when": "Compress large model into deployable small model",
    },
    {
        "name": "Differential Privacy (DPSGD)",
        "mechanism": "Clip gradients, add noise; provable privacy guarantees",
        "params": "100% (with gradient noise)",
        "when": "Training on sensitive data (medical, financial)",
    },
    {
        "name": "Elastic Weight Consolidation",
        "mechanism": "Fisher Information Matrix penalises changing important weights",
        "params": "100% (with regulariser)",
        "when": "Sequential tasks; prevent catastrophic forgetting",
    },
]

techniques_df = pl.DataFrame(techniques)
print(techniques_df)

# ── Checkpoint 5 ─────────────────────────────────────────────────────────
assert techniques_df.height == 10, "Task 5: should cover all 10 techniques"
print("\n✓ Checkpoint 5 passed — 10 fine-tuning techniques surveyed\n")

# INTERPRETATION: The landscape ranges from <0.1% params (prompt tuning)
# to 100% (full fine-tuning + regularisation).  The right choice depends
# on: dataset size, compute budget, privacy requirements, number of tasks.
# LoRA is the pragmatic default; full FT when quality justifies cost.


# ══════════════════════════════════════════════════════════════════════════
# TASK 6: Model Merging Techniques
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 6: Model Merging (TIES, DARE, SLERP, Task Arithmetic)")
print("=" * 70)

print(
    """
Model merging combines multiple fine-tuned models (or LoRA adapters)
into a single model WITHOUT additional training.

TIES (Trim, Elect Sign, Merge — Yadav et al., 2023):
  1. TRIM:        zero out small-magnitude parameter changes (noise)
  2. ELECT SIGN:  for each parameter, majority-vote on the sign of delta
  3. MERGE:       average the deltas that agree on sign
  Produces cleaner merges than naive averaging.

DARE (Drop And REscale — Yu et al., 2023):
  1. Randomly DROP a fraction of parameter deltas (set to zero)
  2. RESCALE remaining deltas by 1/(1-drop_rate) to preserve magnitude
  Like dropout for model merging.  Reduces interference between tasks.

SLERP (Spherical Linear Interpolation):
  Interpolate on the hypersphere rather than linearly:
    merged = sin((1-t)*theta)/sin(theta) * W_A + sin(t*theta)/sin(theta) * W_B
  where theta = arccos(cos_sim(W_A, W_B)).
  Preserves weight norms better than linear interpolation.

Task Arithmetic:
  Simple: task_vector = W_finetuned - W_base
  Add:    W_merged = W_base + alpha * task_vector_A + beta * task_vector_B
  Subtract: W_merged = W_base - alpha * task_vector_A  (remove a capability)
  Alpha/beta control the strength of each task.
"""
)

# Demonstrate TIES conceptually with small tensors
torch.manual_seed(42)
delta_A = torch.randn(128, 128) * 0.1  # Task A fine-tuned delta
delta_B = torch.randn(128, 128) * 0.1  # Task B fine-tuned delta

# TIES step 1: Trim (zero small values)
trim_threshold = 0.05
delta_A_trimmed = delta_A.clone()
delta_A_trimmed[delta_A_trimmed.abs() < trim_threshold] = 0
delta_B_trimmed = delta_B.clone()
delta_B_trimmed[delta_B_trimmed.abs() < trim_threshold] = 0

# TIES step 2: Elect sign (majority vote)
sign_A = delta_A_trimmed.sign()
sign_B = delta_B_trimmed.sign()
elected_sign = (sign_A + sign_B).sign()  # majority wins

# TIES step 3: Merge (average deltas that agree on sign)
mask_A = (sign_A == elected_sign).float()
mask_B = (sign_B == elected_sign).float()
merged_delta = (delta_A_trimmed * mask_A + delta_B_trimmed * mask_B) / (
    mask_A + mask_B + 1e-8
)

print(f"TIES demonstration (128x128 deltas):")
print(f"  Original delta_A non-zero: {(delta_A != 0).sum().item():,}")
print(f"  After trim: {(delta_A_trimmed != 0).sum().item():,}")
print(f"  Sign agreement: {(sign_A == sign_B).float().mean():.1%}")
print(f"  Merged delta norm: {merged_delta.norm():.4f}")


# Demonstrate SLERP
def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation between two tensors."""
    v0_flat = v0.flatten().float()
    v1_flat = v1.flatten().float()
    cos_theta = torch.dot(v0_flat, v1_flat) / (v0_flat.norm() * v1_flat.norm() + 1e-8)
    cos_theta = cos_theta.clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    if theta.abs() < 1e-6:
        return (1 - t) * v0 + t * v1  # fallback to linear
    sin_theta = torch.sin(theta)
    w0 = torch.sin((1 - t) * theta) / sin_theta
    w1 = torch.sin(t * theta) / sin_theta
    return w0 * v0 + w1 * v1


W_base = torch.randn(128, 128)
W_task_A = W_base + delta_A
W_task_B = W_base + delta_B

slerp_merged = slerp(0.5, W_task_A, W_task_B)
linear_merged = 0.5 * W_task_A + 0.5 * W_task_B

print(f"\nSLERP vs linear interpolation:")
print(f"  W_task_A norm: {W_task_A.norm():.4f}")
print(f"  W_task_B norm: {W_task_B.norm():.4f}")
print(f"  SLERP merged norm: {slerp_merged.norm():.4f}")
print(f"  Linear merged norm: {linear_merged.norm():.4f}")
print(f"  SLERP preserves norms better (closer to originals)")

# ── Checkpoint 6 ─────────────────────────────────────────────────────────
assert merged_delta.shape == delta_A.shape, "Task 6: TIES merge should preserve shape"
assert slerp_merged.shape == W_task_A.shape, "SLERP should preserve shape"
print("\n✓ Checkpoint 6 passed — model merging techniques demonstrated\n")

# INTERPRETATION: Model merging is free (no training compute).  Use it
# to combine LoRA adapters from different tasks into a single model.
# TIES: best for reducing noise in multi-adapter merges.
# SLERP: best for two-model interpolation (preserves norms).
# Task arithmetic: simplest; good for adding/removing capabilities.


# ══════════════════════════════════════════════════════════════════════════
# TASK 7: Quantisation Overview
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 7: Quantisation Methods")
print("=" * 70)

print(
    """
Quantisation reduces model precision from FP16/BF16 to INT8/INT4,
shrinking memory footprint and enabling larger models on smaller GPUs.

GPTQ (Frantar et al., 2023):
  Post-training quantisation using second-order information (Hessian).
  Quantises weights to INT4 with minimal quality loss.
  Requires calibration data (small subset of training data).
  One-shot: quantise once, deploy forever.

AWQ (Activation-Aware Weight Quantisation, Lin et al., 2023):
  Identifies "salient" weights (those activated by important inputs).
  Protects salient weights at higher precision, quantises the rest aggressively.
  Better quality than GPTQ at INT4 for instruction-following tasks.

GGUF (formerly GGML):
  CPU-optimised quantisation format used by llama.cpp.
  Supports Q2_K through Q8_0 (2-bit to 8-bit).
  Key advantage: runs on CPU without GPU.  Ideal for edge deployment.

QLoRA (Dettmers et al., 2023):
  Quantise the BASE model to 4-bit (NF4 or FP4).
  Apply LoRA adapters in FP16/BF16 on top of the quantised base.
  Enables fine-tuning 65B models on a single 48GB GPU.
  The base model's 4-bit weights are frozen; only LoRA A,B are trained.
  Memory savings: ~4× vs FP16 base + LoRA.

Quantisation decision guide:
  Deployment on GPU:  GPTQ or AWQ (INT4, fast CUDA kernels)
  Deployment on CPU:  GGUF (Q4_K_M is good default)
  Fine-tuning large models on small GPU: QLoRA
  Maximum quality:    FP16/BF16 (no quantisation)
"""
)

quant_comparison = pl.DataFrame(
    {
        "Method": ["GPTQ", "AWQ", "GGUF (Q4_K_M)", "QLoRA", "FP16 (baseline)"],
        "Precision": ["INT4", "INT4", "4-bit mixed", "NF4 base + FP16 LoRA", "FP16"],
        "Memory (7B model)": ["~4 GB", "~4 GB", "~4.5 GB", "~5 GB train", "~14 GB"],
        "Speed vs FP16": [
            "~1.5-2×",
            "~1.5-2×",
            "CPU-only",
            "~0.8× (train)",
            "1× (baseline)",
        ],
        "Use Case": [
            "GPU inference",
            "GPU inference",
            "CPU/edge deploy",
            "Fine-tune large models",
            "Maximum quality",
        ],
    }
)
print(quant_comparison)

# ── Checkpoint 7 ─────────────────────────────────────────────────────────
assert quant_comparison.height == 5, "Task 7: quantisation table should have 5 methods"
print("\n✓ Checkpoint 7 passed — quantisation methods surveyed\n")

# INTERPRETATION: Quantisation is the practical enabler for deploying
# large models.  A 70B model in FP16 needs ~140GB VRAM — impossible on
# consumer hardware.  At INT4, it fits in ~35GB (one A100-40GB GPU).
# QLoRA makes fine-tuning accessible: adapt a 7B model on a 16GB GPU.


# ══════════════════════════════════════════════════════════════════════════
# TASK 8: AlignmentPipeline SFT Training with LoRA
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 8: AlignmentPipeline — SFT with LoRA")
print("=" * 70)

base_model = os.environ.get("SFT_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

config = AlignmentConfig(
    method="sft",
    base_model=base_model,
    dataset_format="instruction",
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    max_seq_length=512,
    gradient_accumulation_steps=4,
    output_dir="./sft_output",
)

print(f"AlignmentConfig:")
print(f"  Method: {config.method}")
print(f"  Base model: {config.base_model}")
print(f"  LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
print(f"  Targets: {config.target_modules}")
print(f"  Training: {config.num_epochs} epochs, lr={config.learning_rate}")


async def run_sft_training():
    pipeline = AlignmentPipeline(config)
    print("\nRunning SFT training...")
    result = await pipeline.train(train_data=train_data, eval_data=eval_data)
    print(f"  Final loss: {result.final_loss:.4f}")
    print(f"  Eval loss:  {result.eval_loss:.4f}")
    print(f"  Time:       {result.training_time_seconds:.0f}s")
    print(f"  Adapter:    {result.adapter_path}")
    return pipeline, result


pipeline, sft_result = asyncio.run(run_sft_training())

# ── Checkpoint 8 ─────────────────────────────────────────────────────────
assert (
    sft_result.final_loss is not None and sft_result.final_loss > 0
), "Task 8: SFT training should produce positive loss"
print(f"✓ Checkpoint 8 passed — SFT final_loss={sft_result.final_loss:.4f}\n")

# INTERPRETATION: eval_loss < train_loss → possible data leakage.
# eval_loss ≈ train_loss → good generalisation.
# eval_loss >> train_loss → overfitting (reduce epochs or increase dropout).


# ══════════════════════════════════════════════════════════════════════════
# TASK 9: Register Adapter in AdapterRegistry
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 9: Adapter Registry")
print("=" * 70)


async def register_adapter():
    registry = AdapterRegistry()
    adapter_id = await registry.register(
        name="imdb_sentiment_sft_v1",
        base_model=base_model,
        method="sft_lora",
        adapter_path=sft_result.adapter_path,
        metrics={
            "final_loss": sft_result.final_loss,
            "eval_loss": sft_result.eval_loss,
        },
        tags=["imdb", "sentiment", "lora-r16"],
    )
    print(f"Registered adapter: {adapter_id}")
    adapters = await registry.list_adapters()
    print(f"Total adapters: {len(adapters)}")
    for a in adapters:
        print(f"  {a['name']}: {a['method']} on {a['base_model']}")
    return registry, adapter_id


registry, adapter_id = asyncio.run(register_adapter())

# ── Checkpoint 9 ─────────────────────────────────────────────────────────
assert adapter_id is not None, "Task 9: adapter should be registered"
print(f"✓ Checkpoint 9 passed — adapter registered: {adapter_id}\n")

# INTERPRETATION: AdapterRegistry stores metadata (not full weights).
# You can store hundreds of LoRA adapters sharing the same base model.
# At inference, load the base once and swap adapters dynamically.


# ══════════════════════════════════════════════════════════════════════════
# TASK 10: Parameter Reduction Analysis Across Ranks
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 10: Parameter Reduction Analysis")
print("=" * 70)

# Model architecture constants (TinyLlama)
hidden_dim = 2048
num_layers = 22
num_target_modules = len(config.target_modules)

# Full fine-tuning baseline
full_params_per_module = hidden_dim * hidden_dim
full_params_total = full_params_per_module * num_target_modules * num_layers

print(f"Model: {base_model}")
print(f"Hidden dim: {hidden_dim}, Layers: {num_layers}")
print(f"Target modules: {config.target_modules}")
print(f"Full fine-tuning params: {full_params_total:,}")

# LoRA reduction across ranks
print("\n--- LoRA Rank Comparison ---")
ranks = [2, 4, 8, 16, 32, 64, 128]
rank_data = []
for r in ranks:
    lora_p = (hidden_dim * r + r * hidden_dim) * num_target_modules * num_layers
    pct = (lora_p / full_params_total) * 100
    reduction = full_params_total / lora_p
    rank_data.append(
        {"rank": r, "params": lora_p, "pct_of_full": pct, "reduction_factor": reduction}
    )
    print(
        f"  r={r:<4} params={lora_p:>12,}  ({pct:>6.2f}% of full)  {reduction:>6.1f}× reduction"
    )

rank_df = pl.DataFrame(rank_data)

# Adapter comparison
print("\n--- Adapter Bottleneck Comparison ---")
bottlenecks = [32, 64, 128, 256]
for b in bottlenecks:
    # Per adapter: down(d*b + b) + up(b*d + d) + layernorm(2*d)
    adapter_p = (
        hidden_dim * b + b + b * hidden_dim + hidden_dim + 2 * hidden_dim
    ) * num_layers
    pct = (adapter_p / full_params_total) * 100
    print(f"  b={b:<4} params={adapter_p:>12,}  ({pct:>6.2f}% of full)")

# Current config analysis
lora_r = config.lora_r
lora_params_total = (
    (hidden_dim * lora_r + lora_r * hidden_dim) * num_target_modules * num_layers
)
reduction_ratio = full_params_total / lora_params_total
percent_of_full = (lora_params_total / full_params_total) * 100

print(f"\nCurrent config (r={lora_r}):")
print(f"  LoRA params:   {lora_params_total:,}")
print(f"  Full params:   {full_params_total:,}")
print(f"  Reduction:     {reduction_ratio:.1f}×")
print(f"  Percentage:    {percent_of_full:.2f}%")

# ── Checkpoint 10 ────────────────────────────────────────────────────────
assert reduction_ratio > 1, "Task 10: LoRA should use fewer params than full"
assert percent_of_full < 10, f"LoRA should be <10% of full: {percent_of_full:.2f}%"
print(
    f"\n✓ Checkpoint 10 passed — LoRA uses {percent_of_full:.2f}% of full "
    f"({reduction_ratio:.1f}× reduction)\n"
)

# INTERPRETATION: LoRA rank selection guide:
#   r=4:    <1% — fastest training, minimal adaptation (classification)
#   r=8:    ~1% — standard default for most tasks
#   r=16:   ~2% — better for complex domains (legal, medical)
#   r=32+:  diminishing returns — consider full fine-tuning if budget allows
# The optimal rank depends on the task's intrinsic dimensionality —
# how many independent "directions" in weight space the task requires.


# ══════════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  ✓ LoRA from scratch: W_new = W + A @ B, B=0 init, scaling = alpha/r
    A is (d × r), B is (r × d) — only r directions updated
  ✓ Adapter from scratch: bottleneck FC -> GELU -> FC with residual
    Up-proj zero-init so adapter starts as identity
  ✓ LoRA vs adapter: 4-dimension comparison (params, complexity,
    flexibility, modularity)
  ✓ 10 fine-tuning techniques: LoRA, adapters, prefix tuning, prompt
    tuning, full FT, LLRD, progressive freezing, distillation, DPSGD, EWC
  ✓ Model merging: TIES (trim+sign+merge), DARE (drop+rescale),
    SLERP (spherical interpolation), task arithmetic (add/subtract)
  ✓ Quantisation: GPTQ, AWQ, GGUF, QLoRA — and when to use each
  ✓ AlignmentPipeline: SFT training with LoRA hyperparameters
  ✓ AdapterRegistry: versioned adapter storage and retrieval

  NEXT: Exercise 3 (DPO Alignment) moves beyond SFT.  Instead of
  learning "what response to give", DPO learns "which response is
  PREFERRED" — encoding human judgements about quality, safety, style.
"""
)
