# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT06 — Exercise 1: SFT Fine-Tuning with Kailash Align
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Configure AlignmentConfig and run AlignmentPipeline for
#   supervised fine-tuning on a small model. Track adapter in
#   AdapterRegistry. Dataset: Singapore domain Q&A pairs.
#
# TASKS:
#   1. Load SFT dataset (instruction-response pairs)
#   2. Configure AlignmentConfig for SFT
#   3. Run AlignmentPipeline
#   4. Register adapter in AdapterRegistry
#   5. Evaluate fine-tuned vs base model
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from kailash_align import AlignmentConfig, AlignmentPipeline, AdapterRegistry

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Load SFT dataset
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
sft_data = loader.load("ascent06", "sg_domain_qa.parquet")

print(f"=== SFT Dataset ===")
print(f"Shape: {sft_data.shape}")
print(f"Columns: {sft_data.columns}")
print(f"\nSample:")
print(sft_data.head(3))

# SFT expects instruction-response format
# Columns: instruction, response (and optionally: context, category)
n_train = int(sft_data.height * 0.9)
train_data = sft_data[:n_train]
eval_data = sft_data[n_train:]
print(f"\nTrain: {train_data.height}, Eval: {eval_data.height}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Configure AlignmentConfig
# ══════════════════════════════════════════════════════════════════════

base_model = os.environ.get("SFT_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

config = AlignmentConfig(
    method="sft",
    base_model=base_model,
    dataset_format="instruction",  # instruction-response pairs
    # LoRA parameters (efficient fine-tuning)
    lora_r=16,  # Rank — low-rank approximation dimension
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    # Training parameters
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    max_seq_length=512,
    gradient_accumulation_steps=4,
    # Output
    output_dir="./sft_output",
)

print(f"\n=== AlignmentConfig ===")
print(f"Method: {config.method}")
print(f"Base model: {config.base_model}")
print(f"LoRA rank: {config.lora_r} (W = W₀ + BA where B∈ℝ^{{d×r}}, A∈ℝ^{{r×k}})")
print(f"Target modules: {config.target_modules}")
print(f"Training: {config.num_epochs} epochs, lr={config.learning_rate}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Run AlignmentPipeline
# ══════════════════════════════════════════════════════════════════════


async def run_sft():
    pipeline = AlignmentPipeline(config)

    print(f"\n=== Running SFT ===")
    result = await pipeline.train(
        train_data=train_data,
        eval_data=eval_data,
    )

    print(f"Training complete:")
    print(f"  Final loss: {result.final_loss:.4f}")
    print(f"  Eval loss: {result.eval_loss:.4f}")
    print(f"  Training time: {result.training_time_seconds:.0f}s")
    print(f"  Adapter path: {result.adapter_path}")

    return pipeline, result


pipeline, sft_result = asyncio.run(run_sft())


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

    print(f"\n=== Adapter Registered ===")
    print(f"ID: {adapter_id}")
    print(f"Name: sg_domain_sft_v1")
    print(f"Base: {base_model}")

    # List all adapters
    adapters = await registry.list_adapters()
    print(f"\nRegistered adapters: {len(adapters)}")
    for a in adapters:
        print(
            f"  {a.get('name', '?')}: {a.get('method', '?')} on {a.get('base_model', '?')}"
        )

    return registry, adapter_id


registry, adapter_id = asyncio.run(register_adapter())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Evaluate fine-tuned vs base
# ══════════════════════════════════════════════════════════════════════


async def evaluate():
    print(f"\n=== Evaluation: Base vs Fine-tuned ===")

    eval_questions = [
        "What is the HDB resale flat procedure in Singapore?",
        "Explain Singapore's CPF contribution rates for employees.",
        "How does MAS regulate AI in financial services?",
    ]

    for q in eval_questions:
        # Base model response
        base_response = await pipeline.generate(q, use_adapter=False)
        # Fine-tuned response
        ft_response = await pipeline.generate(q, use_adapter=True)

        print(f"\nQ: {q}")
        print(f"Base:      {base_response[:150]}...")
        print(f"Fine-tuned: {ft_response[:150]}...")

    print(f"\nKey observations:")
    print(f"  - Fine-tuned model should show better Singapore domain knowledge")
    print(
        f"  - LoRA adds only {config.lora_r * 2 * len(config.target_modules)} parameters per layer"
    )
    print(
        f"  - Full fine-tuning would update ALL parameters (catastrophic forgetting risk)"
    )


asyncio.run(evaluate())

print("\n✓ Exercise 1 complete — SFT fine-tuning with LoRA + AdapterRegistry")
