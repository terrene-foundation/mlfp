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

print("\n✓ Exercise 1 complete — LoRA SFT with parameter reduction analysis")
