# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 2: LoRA Fine-Tuning with AlignmentPipeline
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Explain LoRA mathematically (W + delta_W = W + A*B, rank r << d)
#   - Configure AlignmentConfig for SFT with LoRA hyperparameters
#   - Run AlignmentPipeline and interpret training/eval loss
#   - Register and version-control adapters in AdapterRegistry
#   - Compute LoRA parameter reduction vs full fine-tuning across rank values
#
# PREREQUISITES:
#   Exercise 1 (LLM fundamentals, transformer architecture from M5.4).
#   Linear algebra (matrix rank, SVD — LoRA IS low-rank factorisation).
#   Understanding that full fine-tuning updates all weights; LoRA only
#   updates two small matrices A and B per layer.
#
# ESTIMATED TIME: 45-75 minutes
#
# TASKS:
#   1. Load SFT dataset (instruction-response pairs)
#   2. Configure AlignmentConfig with LoRA parameters
#   3. Run AlignmentPipeline training
#   4. Register adapter in AdapterRegistry
#   5. Calculate and verify LoRA parameter reduction vs full fine-tuning
#
# DATASET: Singapore domain Q&A pairs (instruction + response format)
#   Instruction-following format for supervised fine-tuning (SFT)
#   Split: 90% train / 10% eval
#   Base model: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (via env variable)
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl

from kailash_align import AdapterRegistry, AlignmentConfig, AlignmentPipeline

from shared import MLFPDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Load SFT dataset
# ══════════════════════════════════════════════════════════════════════

loader = MLFPDataLoader()
sft_data = loader.load("mlfp06", "sg_domain_qa.parquet")

print("=== SFT Dataset ===")
print(f"Shape: {sft_data.shape}")
print(f"Columns: {sft_data.columns}")
print(f"\nSample:\n{sft_data.head(3)}")

n_train = int(sft_data.height * 0.9)
train_data = sft_data[:n_train]
eval_data = sft_data[n_train:]
print(f"\nTrain: {train_data.height}, Eval: {eval_data.height}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Configure AlignmentConfig with LoRA parameters
# ══════════════════════════════════════════════════════════════════════

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

print("\n=== AlignmentConfig ===")
print(f"Method: {config.method}")
print(f"Base model: {config.base_model}")
print(f"LoRA rank: {config.lora_r}")
print(f"LoRA alpha: {config.lora_alpha}")
print(f"Target modules: {config.target_modules}")
print(f"Training: {config.num_epochs} epochs, lr={config.learning_rate}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Run AlignmentPipeline training
# ══════════════════════════════════════════════════════════════════════


async def run_training():
    pipeline = AlignmentPipeline(config)

    print("\n=== Running SFT Training ===")
    result = await pipeline.train(
        train_data=train_data,
        eval_data=eval_data,
    )

    print("Training complete:")
    print(f"  Final loss: {result.final_loss:.4f}")
    print(f"  Eval loss: {result.eval_loss:.4f}")
    print(f"  Training time: {result.training_time_seconds:.0f}s")
    print(f"  Adapter path: {result.adapter_path}")

    return pipeline, result


pipeline, sft_result = asyncio.run(run_training())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Register adapter in AdapterRegistry
# ══════════════════════════════════════════════════════════════════════


async def register_adapter():
    registry = AdapterRegistry()

    adapter_id = await registry.register(
        name="sg_domain_sft_v1",
        base_model=base_model,
        method="sft_lora",
        adapter_path=sft_result.adapter_path,
        metrics={
            "final_loss": sft_result.final_loss,
            "eval_loss": sft_result.eval_loss,
        },
        tags=["singapore", "domain-qa", "lora-r16"],
    )

    print("\n=== Adapter Registered ===")
    print(f"ID: {adapter_id}")
    print(f"Name: sg_domain_sft_v1")

    adapters = await registry.list_adapters()
    print(f"\nRegistered adapters: {len(adapters)}")
    for a in adapters:
        print(f"  {a['name']}: {a['method']} on {a['base_model']}")

    return registry, adapter_id


registry, adapter_id = asyncio.run(register_adapter())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Calculate LoRA parameter reduction vs full fine-tuning
# ══════════════════════════════════════════════════════════════════════

# TinyLlama hidden_dim=2048, num_layers=22
hidden_dim = 2048
num_layers = 22
lora_r = config.lora_r
num_target_modules = len(config.target_modules)

# Full fine-tuning: each target module weight = hidden_dim x hidden_dim
full_params_per_module = hidden_dim * hidden_dim
full_params_total = full_params_per_module * num_target_modules * num_layers

# LoRA: each module gets B (hidden_dim x r) + A (r x hidden_dim)
lora_params_per_module = (hidden_dim * lora_r) + (lora_r * hidden_dim)
lora_params_total = lora_params_per_module * num_target_modules * num_layers

reduction_ratio = full_params_total / lora_params_total
percent_of_full = (lora_params_total / full_params_total) * 100

print("\n=== Parameter Reduction Analysis ===")
print(f"Model: {base_model}")
print(f"Hidden dim: {hidden_dim}, Layers: {num_layers}")
print(f"Target modules: {config.target_modules}")
print(f"\nFull fine-tuning parameters: {full_params_total:,}")
print(f"LoRA parameters (r={lora_r}): {lora_params_total:,}")
print(f"Reduction ratio: {reduction_ratio:.1f}x fewer parameters")
print(f"LoRA is {percent_of_full:.2f}% of full fine-tuning")

# Compare different ranks
print("\n--- LoRA rank comparison ---")
ranks = [4, 8, 16, 32, 64]
for r in ranks:
    lora_p = (hidden_dim * r + r * hidden_dim) * num_target_modules * num_layers
    pct = (lora_p / full_params_total) * 100
    print(f"  r={r:<3}  params={lora_p:>10,}  ({pct:.2f}% of full)")

print("=" * 60)
print("  MLFP06 Exercise 2: LoRA Fine-Tuning with AlignmentPipeline")
print("=" * 60)
print(f"\n  LoRA SFT complete. Adapter registered in AdapterRegistry.\n")

# ── Checkpoint 1: Dataset ──────────────────────────────────────────────
assert sft_data.height > 0, "SFT dataset should not be empty"
assert "instruction" in sft_data.columns or "text" in sft_data.columns, \
    "Dataset should have instruction or text column"
print(f"✓ Checkpoint 1 passed — SFT dataset: {sft_data.height} pairs "
      f"({train_data.height} train, {eval_data.height} eval)\n")

# INTERPRETATION: Supervised Fine-Tuning (SFT) data consists of
# (instruction, response) pairs. The model learns to follow instructions
# by maximising the probability of the response given the instruction.
# Data quality matters more than quantity for SFT: 500 high-quality
# Singapore-specific pairs can outperform 10,000 generic pairs.

# ── Checkpoint 2: LoRA configuration ─────────────────────────────────
assert config.lora_r == 16, "LoRA rank should be 16"
assert config.lora_alpha == 32, "LoRA alpha should be 32"
assert "q_proj" in config.target_modules, "Should target q_proj"
assert "v_proj" in config.target_modules, "Should target v_proj"
print(f"✓ Checkpoint 2 passed — AlignmentConfig: r={config.lora_r}, "
      f"alpha={config.lora_alpha}, targets={config.target_modules}\n")

# INTERPRETATION: LoRA hyperparameters:
# lora_r (rank): smaller = fewer params, less expressive. r=8-16 for most tasks.
# lora_alpha: scaling factor, typically 2x rank. Higher alpha = stronger adaptation.
# target_modules: which weight matrices to adapt. q_proj + v_proj is standard;
# adding k_proj, o_proj, gate_proj adds more params but may improve performance.
# lora_dropout: regularisation to prevent overfitting the adapter.

# ── Checkpoint 3: Training ────────────────────────────────────────────
assert sft_result.final_loss is not None, "Training should produce final loss"
assert sft_result.final_loss > 0, "Loss should be positive"
print(f"✓ Checkpoint 3 passed — SFT training: final_loss={sft_result.final_loss:.4f}, "
      f"eval_loss={sft_result.eval_loss:.4f}\n")

# INTERPRETATION: eval_loss < train_loss would indicate data leakage.
# eval_loss ≈ train_loss indicates good generalisation.
# eval_loss >> train_loss indicates overfitting — reduce epochs or increase dropout.
# Absolute loss values depend on vocabulary size and sequence length.
# Decreasing loss confirms the model is learning domain-specific patterns.

# ── Checkpoint 4: Adapter registry ───────────────────────────────────
assert adapter_id is not None, "Adapter should be registered with an ID"
print(f"✓ Checkpoint 4 passed — adapter registered: ID={adapter_id}\n")

# INTERPRETATION: AdapterRegistry stores adapter metadata without the full
# model weights — just the small delta matrices (A and B for each targeted
# layer). This means you can store hundreds of LoRA adapters for different
# tasks, all sharing the same base model weights. At inference time, you
# load the base model once and swap adapters dynamically.

# ── Checkpoint 5: Parameter reduction ────────────────────────────────
assert reduction_ratio > 1, "LoRA should use fewer parameters than full fine-tuning"
assert percent_of_full < 10, f"LoRA should be <10% of full params: {percent_of_full:.2f}%"
print(f"✓ Checkpoint 5 passed — LoRA uses {percent_of_full:.2f}% of full params "
      f"({reduction_ratio:.1f}x reduction)\n")

# INTERPRETATION: At r=16, LoRA trains (hidden_dim * r + r * hidden_dim) per module.
# For TinyLlama (d=2048, 22 layers, 2 target modules):
# LoRA params = 2048*16*2 * 2 * 22 = ~2.9M vs full fine-tuning ~185M per layer.
# This is why LoRA can be trained on a single GPU: the gradient computation
# for A and B is tiny compared to the full weight matrix.
# r=4: <1% of full (fastest but least expressive)
# r=64: ~4% of full (more expressive, used for complex task adaptation)


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 60)
print("  WHAT YOU'VE MASTERED")
print("═" * 60)
print("""
  ✓ LoRA mathematics: W_new = W + A*B, where A is (d x r) and B is (r x d)
    rank r << d means A*B has at most r non-zero singular values
    Equivalent to: only r "directions" in weight space are updated
  ✓ AlignmentConfig: method, base_model, lora_r, lora_alpha, target_modules
  ✓ AlignmentPipeline.train(): SFT training with train/eval data
  ✓ AdapterRegistry: versioned storage of adapter metadata + paths
  ✓ Parameter reduction: LoRA uses 1-5% of full fine-tuning parameters

  LoRA rank selection guide:
    r=4:   <1% of full params, suitable for simple task adaptation
    r=8:   ~1% of full params, standard for most tasks
    r=16:  ~2% of full params, better for complex domains
    r=32+: diminishing returns, consider full fine-tuning if needed

  NEXT: Exercise 3 (DPO Alignment) moves beyond SFT to preference alignment.
  Instead of learning "what response to give", DPO learns "what response
  is PREFERRED" — encoding human judgements about quality, safety, and style.
""")
