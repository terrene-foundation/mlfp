# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT10 — Exercise 1: LoRA Fine-Tuning with AlignmentPipeline
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Configure AlignmentConfig for LoRA-based SFT, train on
#   Singapore domain Q&A, register the adapter, and calculate parameter
#   savings vs full fine-tuning.
#
# TASKS:
#   1. Load SFT dataset (instruction-response pairs)
#   2. Configure AlignmentConfig with LoRA parameters
#   3. Run AlignmentPipeline training
#   4. Register adapter in AdapterRegistry
#   5. Calculate and verify LoRA parameter reduction vs full fine-tuning
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl

from kailash_align import AdapterRegistry, AlignmentConfig, AlignmentPipeline

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Load SFT dataset
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
sft_data = loader.load("ascent10", "sg_domain_qa.parquet")

print("=== SFT Dataset ===")
print(f"Shape: {sft_data.shape}")
print(f"Columns: {sft_data.columns}")
print(f"\nSample:\n{sft_data.head(3)}")

# TODO: Split into 90% train, 10% eval.
# Hint: Use sft_data.height for the row count.
n_train = ____
train_data = ____
eval_data = ____
print(f"\nTrain: {train_data.height}, Eval: {eval_data.height}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Configure AlignmentConfig with LoRA parameters
# ══════════════════════════════════════════════════════════════════════

base_model = os.environ.get("SFT_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# TODO: Create an AlignmentConfig for SFT with LoRA.
# Hint: AlignmentConfig(method=..., base_model=..., dataset_format=...,
#        lora_r=..., lora_alpha=..., lora_dropout=..., target_modules=...,
#        num_epochs=..., batch_size=..., learning_rate=..., ...)
config = ____

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
    # TODO: Create AlignmentPipeline and train on the dataset.
    # Hint: AlignmentPipeline(config), then await pipeline.train(...)
    pipeline = ____

    print("\n=== Running SFT Training ===")
    result = ____

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
    # TODO: Create AdapterRegistry and register the trained adapter.
    # Hint: AdapterRegistry(), then await registry.register(name=..., base_model=...,
    #        method=..., adapter_path=..., metrics=..., tags=...)
    registry = ____

    adapter_id = ____

    print("\n=== Adapter Registered ===")
    print(f"ID: {adapter_id}")
    print(f"Name: sg_domain_sft_v1")

    # TODO: List all registered adapters.
    # Hint: await registry.list_adapters()
    adapters = ____
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

# TODO: Calculate full fine-tuning parameters per module and total.
# Hint: Each target module weight matrix is hidden_dim x hidden_dim.
full_params_per_module = ____
full_params_total = ____

# TODO: Calculate LoRA parameters per module and total.
# Hint: LoRA decomposes into B (hidden_dim x r) + A (r x hidden_dim).
lora_params_per_module = ____
lora_params_total = ____

# TODO: Calculate reduction ratio and percentage.
reduction_ratio = ____
percent_of_full = ____

print("\n=== Parameter Reduction Analysis ===")
print(f"Model: {base_model}")
print(f"Hidden dim: {hidden_dim}, Layers: {num_layers}")
print(f"Target modules: {config.target_modules}")
print(f"\nFull fine-tuning parameters: {full_params_total:,}")
print(f"LoRA parameters (r={lora_r}): {lora_params_total:,}")
print(f"Reduction ratio: {reduction_ratio:.1f}x fewer parameters")
print(f"LoRA is {percent_of_full:.2f}% of full fine-tuning")

# TODO: Compare different LoRA ranks. For each rank, compute params and percentage.
# Hint: Loop over ranks [4, 8, 16, 32, 64], compute (hidden_dim * r + r * hidden_dim) * ...
print("\n--- LoRA rank comparison ---")
ranks = [4, 8, 16, 32, 64]
for r in ranks:
    lora_p = ____
    pct = ____
    print(f"  r={r:<3}  params={lora_p:>10,}  ({pct:.2f}% of full)")

print("\n✓ Exercise 1 complete — LoRA SFT with parameter reduction analysis")
