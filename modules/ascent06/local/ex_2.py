# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT06 — Exercise 2: DPO / QLoRA Alignment
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Compare DPO (preference optimization) vs QLoRA (quantized
#   fine-tuning). Evaluate with LLM-as-judge and human rubric.
#
# TASKS:
#   1. Load preference pairs dataset
#   2. Configure and run DPO alignment
#   3. Configure and run QLoRA fine-tuning
#   4. Evaluate both with LLM-as-judge
#   5. Compare methods: quality, cost, speed
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

base_model = os.environ.get("SFT_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Load preference pairs
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
preferences = loader.load("ascent06", "preference_pairs.parquet")

print(f"=== Preference Pairs ===")
print(f"Shape: {preferences.shape}")
print(f"Columns: {preferences.columns}")
# Expected columns: prompt, chosen, rejected
print(preferences.head(2))


# ══════════════════════════════════════════════════════════════════════
# TASK 2: DPO alignment
# ══════════════════════════════════════════════════════════════════════
# DPO: Direct Preference Optimization
# Derives from Bradley-Terry preference model:
#   P(y_w > y_l | x) = σ(β⁻¹(r(x,y_w) - r(x,y_l)))
# Key insight: eliminates the reward model entirely
#   π*(y|x) ∝ π_ref(y|x) · exp(β⁻¹ · r*(x,y))
# Loss: L_DPO = -E[log σ(β(log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]

# TODO: Create AlignmentConfig for DPO.
# Hint: AlignmentConfig(
#   method="dpo",             # Direct Preference Optimization
#   base_model=base_model,
#   dataset_format="preference",  # expects: prompt, chosen, rejected columns
#   beta=0.1,                 # temperature: higher = stay closer to reference policy
#   lora_r=16, lora_alpha=32, lora_dropout=0.05,
#   target_modules=["q_proj", "v_proj"],
#   num_epochs=2, batch_size=2, learning_rate=5e-5,
#   max_seq_length=512, output_dir="./dpo_output"
# )
dpo_config = ____


async def run_dpo():
    # TODO: Create AlignmentPipeline from dpo_config and call pipeline.train().
    # Hint: Split preferences into 90/10 train/eval using preferences[:n_train]
    #   and preferences[n_train:]. Result has .final_loss, .eval_loss,
    #   .training_time_seconds attributes.
    pipeline = ____
    n_train = int(preferences.height * 0.9)

    print(f"\n=== Running DPO ===")
    print(f"β = {dpo_config.beta} (higher β = stay closer to reference)")

    result = await ____

    print(f"DPO complete:")
    print(f"  Final loss: {result.final_loss:.4f}")
    print(f"  Eval loss: {result.eval_loss:.4f}")
    print(f"  Time: {result.training_time_seconds:.0f}s")

    return pipeline, result


dpo_pipeline, dpo_result = asyncio.run(run_dpo())


# ══════════════════════════════════════════════════════════════════════
# TASK 3: QLoRA fine-tuning
# ══════════════════════════════════════════════════════════════════════
# QLoRA: Quantized LoRA
# 1. Quantize base model to NF4 (4-bit NormalFloat)
# 2. Apply LoRA adapters on top of quantized model
# 3. Train adapters in full precision while base stays 4-bit
# 4. Double quantization: quantize the quantization constants too

# TODO: Create AlignmentConfig for QLoRA.
# Hint: AlignmentConfig(
#   method="sft",             # SFT on the "chosen" responses from the preference pairs
#   base_model=base_model,
#   dataset_format="instruction",
#   quantization="nf4",       # 4-bit NormalFloat — the QLoRA quantization scheme
#   double_quantization=True, # quantize the quantization constants for extra memory savings
#   compute_dtype="bfloat16", # compute in bf16 while base stays 4-bit
#   lora_r=16, lora_alpha=32, lora_dropout=0.05,
#   target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # more modules for QLoRA
#   num_epochs=3, batch_size=4, learning_rate=2e-4,
#   max_seq_length=512, output_dir="./qlora_output"
# )
qlora_config = ____


async def run_qlora():
    # TODO: Convert preference pairs to instruction format (use the "chosen" column),
    # then create AlignmentPipeline from qlora_config and call pipeline.train().
    # Hint: Use polars .select() to rename columns:
    #   sft_from_prefs = preferences.select(
    #       pl.col("prompt").alias("instruction"),
    #       pl.col("chosen").alias("response"),
    #   )
    import polars as pl

    sft_from_prefs = ____

    pipeline = ____
    n_train = int(sft_from_prefs.height * 0.9)

    print(f"\n=== Running QLoRA ===")
    print(f"Quantization: NF4 + double quantization")
    print(f"Memory savings: ~75% vs full precision")

    result = await ____

    print(f"QLoRA complete:")
    print(f"  Final loss: {result.final_loss:.4f}")
    print(f"  Eval loss: {result.eval_loss:.4f}")
    print(f"  Time: {result.training_time_seconds:.0f}s")

    return pipeline, result


qlora_pipeline, qlora_result = asyncio.run(run_qlora())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Evaluate with LLM-as-judge
# ══════════════════════════════════════════════════════════════════════


async def llm_judge_evaluation():
    """Use an LLM to judge response quality."""
    # TODO: Create a Kaizen Delegate as the judge and evaluate both models.
    # Hint:
    #   from kaizen_agents import Delegate
    #   judge_model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
    #   judge = Delegate(model=judge_model, max_llm_cost_usd=2.0)
    #
    # For each eval prompt, get responses from both pipelines:
    #   dpo_response = await dpo_pipeline.generate(prompt, use_adapter=True)
    #   qlora_response = await qlora_pipeline.generate(prompt, use_adapter=True)
    #
    # Then ask the judge to compare them:
    #   async for event in judge.run(judge_prompt):
    #       if hasattr(event, "text"): judge_text += event.text
    from kaizen_agents import Delegate

    judge_model = os.environ.get(
        "DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL")
    )
    judge = ____

    eval_prompts = [
        "Explain Singapore's HDB BTO application process.",
        "What are the key differences between CPF OA, SA, and MA?",
        "How does Singapore's AI governance framework compare to the EU AI Act?",
    ]

    print(f"\n=== LLM-as-Judge Evaluation ===")
    for prompt in eval_prompts:
        dpo_response = await ____
        qlora_response = await ____

        judge_prompt = (
            f"Compare these two responses to: '{prompt}'\n\n"
            f"Response A: {dpo_response[:300]}\n\n"
            f"Response B: {qlora_response[:300]}\n\n"
            f"Which is better? Rate each 1-5 on: accuracy, completeness, clarity."
        )

        # TODO: Stream the judge's evaluation response.
        # Hint: async for event in judge.run(judge_prompt):
        #           if hasattr(event, "text"): judge_text += event.text
        judge_text = ""
        async for event in ____:
            if hasattr(event, "text"):
                judge_text += event.text

        print(f"\nPrompt: {prompt[:60]}...")
        print(f"Judge: {judge_text[:200]}...")

    # Known biases in LLM-as-judge
    print(f"\nLLM-as-Judge Biases:")
    print(f"  1. Position bias: prefers Response A (first position)")
    print(f"  2. Verbosity bias: prefers longer responses")
    print(f"  3. Self-enhancement: prefers responses similar to its own style")
    print(f"  Mitigation: swap positions, control length, use multiple judges")


asyncio.run(llm_judge_evaluation())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Method comparison
# ══════════════════════════════════════════════════════════════════════

# TODO: Print a comparison table of DPO vs QLoRA results.
# Hint: Use dpo_result.final_loss, dpo_result.training_time_seconds,
#   qlora_result.final_loss, qlora_result.training_time_seconds.
#   Format with f-string column alignment using :>N width specifiers.
print(f"\n=== DPO vs QLoRA Comparison ===")
print(f"{'Aspect':<25} {'DPO':>15} {'QLoRA':>15}")
print("─" * 58)
print(f"{'Data requirement':<25} {'preference pairs':>15} {'instruction pairs':>15}")
print(f"{'Reward model needed':<25} {'No (eliminated)':>15} {'No':>15}")
print(f"{'Memory efficiency':<25} {'LoRA + fp16':>15} {'LoRA + NF4':>15}")
print(f"{'Alignment quality':<25} {'Higher':>15} {'Good':>15}")
# TODO: Print training loss and time rows using dpo_result and qlora_result.
# Hint: f"{dpo_result.final_loss:>15.4f}" and f"{qlora_result.final_loss:>15.4f}"
print(f"{'Training loss':<25} {____:>15.4f} {____:>15.4f}")
print(f"{'Training time':<25} {____:>15.0f}s {____:>15.0f}s")
print(f"\nWhen to choose:")
print(f"  DPO: when you have preference data and want alignment")
print(f"  QLoRA: when you have instruction data and limited GPU memory")
print(f"  Both: can be combined (QLoRA for SFT, then DPO for alignment)")

print("\n✓ Exercise 2 complete — DPO vs QLoRA with LLM-as-judge evaluation")
