# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 2.6: SFT with kailash-align AlignmentPipeline
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build an AlignmentConfig for SFT + LoRA (no raw transformers.Trainer)
#   - Run AlignmentPipeline.train() on the IMDB instruction dataset
#   - Register the trained adapter in AdapterRegistry
#   - Visualise SFT training loss + throughput
#   - Apply adapter-registry discipline to a Singapore e-commerce scenario
#
# PREREQUISITES: Exercises 2.1-2.5
# ESTIMATED TIME: ~40 min on GPU/MPS (training dominates). Measured
# 2026-10-08: ~4 min on a 2×3090 host; on a CPU-only host the 3-epoch SFT
# runs ~1.7 hours in fp32 (bf16 is CUDA-only — the config now follows the
# device). On a CPU laptop, run the Colab notebook on a GPU runtime instead.
#
# FRAMEWORK-FIRST: kailash-align, NOT raw transformers.Trainer. The
# pipeline wraps TRL SFTTrainer but adds a single typed config, adapter
# serialisation, and registry integration — the same pattern you used
# in MLFP03 for TrainingPipeline.
#
# TASKS:
#   1. THEORY: why AlignmentPipeline beats raw SFTTrainer
#   2. BUILD: AlignmentConfig for SFT + LoRA r=16
#   3. TRAIN: pipeline.train() on IMDB instruction pairs
#   4. VISUALISE: SFT training loss + throughput
#   5. APPLY: Singapore e-commerce adapter registry governance
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import matplotlib.pyplot as plt
import polars as pl
from dotenv import load_dotenv

from shared.mlfp06.ex_2 import (
    OUTPUT_DIR,
    build_sft_config,
    get_base_model_name,
    load_imdb_sft,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why AlignmentPipeline (Framework-First)
# ════════════════════════════════════════════════════════════════════════
# Raw transformers.Trainer + peft.get_peft_model loses three guarantees:
#   1. One typed AlignmentConfig (LoRA/SFT/DPO sub-configs) — note it only
#      checks target_modules is non-empty, not that the names exist
#   2. Structured adapter artefact lifecycle (not a bag of .bin files)
#   3. AdapterRegistry hand-off between training and serving
# Framework-first rule: Engine layer first, primitives only when the
# engine cannot express the behaviour.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load the SFT dataset
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 1: Load IMDB SFT instruction pairs")
print("=" * 70)

sft_data, train_data, eval_data = load_imdb_sft()
print(f"Train: {train_data.height} pairs")
print(f"Eval:  {eval_data.height} pairs")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert train_data.height > 0, "Task 1: train split should not be empty"
assert eval_data.height > 0, "Task 1: eval split should not be empty"
print("✓ Checkpoint 1 passed — SFT data loaded\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: AlignmentConfig via the shared factory
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: Build AlignmentConfig for SFT + LoRA r=16")
print("=" * 70)

# kailash-align 0.7.3 resumes unconditionally from any checkpoint found in
# the experiment dir — a crashed earlier run leaves a completed-looking
# checkpoint and every re-run "trains" zero steps. Retrain deterministically:
import shutil

shutil.rmtree(OUTPUT_DIR / "sft_output", ignore_errors=True)

# TODO: Read the base model name from the environment via get_base_model_name()
base_model = ____
# TODO: Build an AlignmentConfig via build_sft_config(base_model=base_model,
# lora_r=16, lora_alpha=32, num_epochs=3, output_subdir="sft_output")
config = ____

print(f"  Method:        {config.method}")
print(f"  Base model:    {config.base_model_id}")
print(f"  LoRA:          r={config.lora.rank}, alpha={config.lora.alpha}")
print(f"  Target modules: {config.lora.target_modules}")
print(f"  Epochs:        {config.sft.num_train_epochs}")
print(f"  Batch size:    {config.sft.per_device_train_batch_size}")
print(f"  Learning rate: {config.sft.learning_rate}")
print(f"  Output dir:    {config.experiment_dir}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert config.method == "sft", "Task 2: method should be 'sft'"
assert config.lora.rank == 16, "Task 2: LoRA rank should be 16"
assert "q_proj" in config.lora.target_modules, "Task 2: q_proj must be a target module"
print("✓ Checkpoint 2 passed — AlignmentConfig built\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: AlignmentPipeline + register adapter
# ════════════════════════════════════════════════════════════════════════
# Training needs the base model from HuggingFace and is slow on CPU. There
# is deliberately NO "skip and pretend" mode: MLFP_SKIP_SFT_TRAIN=1 stops
# the exercise with an explanation instead of reporting a made-up loss.

print("=" * 70)
print("TASK 3: AlignmentPipeline.train() -> AdapterRegistry")
print("=" * 70)


async def run_sft_and_register() -> dict:
    """Train the SFT adapter and register it. Returns metrics dict."""
    import os

    if os.environ.get("MLFP_SKIP_SFT_TRAIN") == "1":
        raise RuntimeError(
            "MLFP_SKIP_SFT_TRAIN=1 is set, so SFT training was skipped and "
            "there is no training loss to report. Unset it to run Task 3."
        )
    from kailash_align import (
        AdapterRegistry,
        AdapterSignature,
        AlignmentPipeline,
    )

    # TODO: Instantiate AlignmentPipeline(config)
    pipeline = ____
    print("  Running SFT training (this may take several minutes)...")
    # TODO: SFT trains on the `text` column, but `text` in train_data is the
    # RAW review. Build a one-column polars frame whose `text` is
    # instruction + "\n\n" + response (pl.col(...) + ... .alias("text")).
    sft_frame = ____
    # TODO: kailash-align checks `dataset.column_names`, which a polars
    # DataFrame does not have. Convert: Dataset.from_dict(
    #   sft_frame.to_dict(as_series=False)) with `from datasets import Dataset`.
    from datasets import Dataset

    train_data_hf = ____
    # TODO: await pipeline.train(train_data_hf, adapter_name="imdb_sentiment_sft_v1").
    # Note (kailash-align 0.7.3): `dataset` is positional, `adapter_name`
    # is REQUIRED, and there is NO eval_data parameter anymore.
    result = ____

    # result.training_metrics is the raw TRL TrainOutput.metrics dict:
    # train_loss + train_runtime + train_samples_per_second. There is NO
    # eval_loss (no eval dataset is configured under the new API).
    train_metrics = result.training_metrics
    metrics = {
        "train_loss": train_metrics.get("train_loss"),
        "train_runtime": train_metrics.get("train_runtime", 0),
        "train_samples_per_second": train_metrics.get(
            "train_samples_per_second", 0
        ),
        "adapter_path": result.adapter_path,
    }
    print(f"  Train loss:    {metrics['train_loss']:.4f}")
    print(f"  Train runtime: {metrics['train_runtime']:.0f}s")
    print(f"  Throughput:    {metrics['train_samples_per_second']:.1f} samples/s")

    # TODO: Instantiate registry = AdapterRegistry()
    registry = ____
    # TODO: Build an AdapterSignature with base_model_id=config.base_model_id,
    #   adapter_type="lora", training_method="sft"
    signature = ____
    # TODO: await registry.register_adapter(name=..., adapter_path=...,
    #   signature=..., training_metrics={"train_loss": ...}, tags=[...])
    # register_adapter returns an AdapterVersion (not a string).
    version = ____
    # TODO: Build a stable adapter_id string of the form
    #   f"{version.adapter_name}:v{version.version}"
    adapter_id = ____
    metrics["adapter_id"] = adapter_id
    print(f"  Registered as: {adapter_id}")

    return metrics


metrics = asyncio.run(run_sft_and_register())

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert metrics["train_loss"] is not None, "Task 3: training should produce a loss"
assert metrics["train_loss"] > 0, "Task 3: train loss should be positive"
print("✓ Checkpoint 3 passed — SFT training + registration complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: SFT training loss + throughput
# ════════════════════════════════════════════════════════════════════════
# The new train() API returns the TRL headline metrics (no eval split), so
# we plot the honest signal it provides: the final train loss alongside
# training throughput. We deliberately do NOT fabricate an eval curve —
# generalisation is measured by a separate held-out eval pass (Ex 3.4).

print("=" * 70)
print("TASK 4: Visualise SFT training loss + throughput")
print("=" * 70)

fig, (ax_loss, ax_tput) = plt.subplots(1, 2, figsize=(11, 5))

# Left: final train loss (single headline bar — train() returns the run
# mean, not a per-step curve; the per-step curve lives in the TRL logs).
ax_loss.bar(
    ["Train loss (run mean)"],
    [metrics["train_loss"]],
    color="steelblue",
    edgecolor="black",
)
ax_loss.annotate(
    f"{metrics['train_loss']:.3f}",
    xy=(0, metrics["train_loss"]),
    xytext=(0, 3),
    textcoords="offset points",
    ha="center",
)
ax_loss.set_ylabel("Cross-entropy loss")
ax_loss.set_title("SFT LoRA r=16 — train loss", fontweight="bold")
ax_loss.grid(True, axis="y", alpha=0.3)

# Right: throughput — the other honest signal the pipeline reports.
ax_tput.bar(
    ["samples/s"],
    [metrics["train_samples_per_second"]],
    color="seagreen",
    edgecolor="black",
)
ax_tput.annotate(
    f"{metrics['train_samples_per_second']:.1f}",
    xy=(0, metrics["train_samples_per_second"]),
    xytext=(0, 3),
    textcoords="offset points",
    ha="center",
)
ax_tput.set_ylabel("Samples / second")
ax_tput.set_title(
    f"Training throughput (runtime {metrics['train_runtime']:.0f}s)",
    fontweight="bold",
)
ax_tput.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
fname = OUTPUT_DIR / "ex2_sft_train_loss.png"
fname.unlink(missing_ok=True)  # so the checkpoint cannot pass on a stale file
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {fname}")

print(f"\n  Final train loss: {metrics['train_loss']:.3f}")
print("  Healthy SFT runs show train_loss falling across epochs; measure")
print("  generalisation with a separate held-out eval pass (Exercise 3.4).")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert fname.exists(), "Task 4: loss plot should exist"
print("✓ Checkpoint 4 passed — SFT train loss visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore e-commerce adapter registry governance
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore e-commerce platform runs an LLM-powered
# customer service assistant.  Over 18 months, the team has trained
# 37 different LoRA adapters on top of the same 7B base: returns,
# refunds, promotions, shipping, loyalty tiers, billing, KYC, and so
# on.  Each adapter was trained by a different on-call engineer with
# no shared record of "which adapter is live in production right now".
#
# PROBLEM: when a shopper complains that "the chatbot said shipping
# is free above S$50 but it charged me", the team cannot trace which
# adapter produced the answer, which dataset it was trained on, or
# which eval metrics it passed at training time.  Customer trust
# erodes; the promotion team cannot roll back a bad adapter because
# they don't know which version is deployed.
#
# GOVERNANCE FIX: every trained adapter MUST go through AdapterRegistry
# at training time with:
#   - name: human-readable identifier (e.g. "shipping_v3")
#   - base_model: exact base checkpoint SHA
#   - method: sft_lora / dpo_lora / adapter / ...
#   - metrics: train loss, eval loss, business-level eval (CSAT proxy)
#   - tags: domain (shipping), audience (retail), legal-reviewed (yes)
#
# Downstream serving pulls by registry name + metric threshold, never
# by raw .bin filename.  Deployment becomes auditable: every live
# adapter has a registry row that links back to the training run.
#
# BUSINESS IMPACT (illustrative figures):
#   - Audit: the compliance team can trace any customer-facing LLM
#     response to a registered adapter + training run + eval report —
#     the evidence a regulator's technology-risk review asks for.
#   - Rollback: a bad adapter can be rolled back in minutes by
#     pointing the serving layer at the previous registered version.
#     Previous rollback took ~4 hours and required redeploying the
#     serving image.
#   - Retraining cost avoided: the registry prevents duplicate runs
#     of the same fine-tune.  Engineers used to retrain ~4 adapters
#     per quarter because they could not find the previous adapter
#     files.  At ~S$600/training run + ~1 day engineering time, that
#     is ~S$2,400 + 4 engineer-days per quarter avoided.
#
# ANNUAL BENEFIT: computed below from these assumptions (~S$21K in direct
# cost) plus audit traceability and faster incident response.
#
# THE RULE: no adapter ships to production without a registry entry.
# Treat AdapterRegistry the way you treat ModelRegistry in MLFP03 —
# the single source of truth for which artefact runs where.

print("Singapore e-commerce adapter governance:")
annual_retraining_saving = 4 * 4 * 600  # 4 quarters * 4 runs/quarter * S$600
annual_engineer_hours_saved = 4 * 4 * 8  # 4 * 4 * 1 day
engineer_hourly_sgd = 90
annual_engineer_saving = annual_engineer_hours_saved * engineer_hourly_sgd
total_annual = annual_retraining_saving + annual_engineer_saving
print(f"  Retraining runs avoided / year:      {4 * 4}")
print(f"  Annual retraining cost avoided:      S${annual_retraining_saving:,}")
print(f"  Engineer hours saved / year:         {annual_engineer_hours_saved}")
print(f"  Annual engineer time saving:         S${annual_engineer_saving:,}")
print(f"  Total direct annual saving:          S${total_annual:,}")
print("  Plus: audit traceability + rollback SLA (not quantified here)")
print("  Recommended: AdapterRegistry mandatory for all SFT runs")

# ── Checkpoint 5 ─────────────────────────────────────────────────────────
assert total_annual > 0, "Task 5: governance should deliver positive ROI"
print("✓ Checkpoint 5 passed — e-commerce governance analysed\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built an AlignmentConfig for SFT + LoRA via kailash-align
      (framework-first: no raw transformers.Trainer)
  [x] Ran AlignmentPipeline.train() end-to-end on IMDB SFT data
  [x] Registered the trained adapter in AdapterRegistry with metrics
  [x] Visualised SFT train loss + throughput (the honest signals the
      0.7.3 train() API reports — no fabricated eval curve)
  [x] Applied adapter-registry governance to a Singapore e-commerce
      scenario (illustrative direct saving + audit traceability)

  KEY INSIGHT: SFT is the first rung of the alignment ladder.
  AlignmentPipeline + AdapterRegistry give you the versioning and
  audit trail you need before you layer DPO, GRPO, or RLHF on top.

  Next exercise (Exercise 3) moves from "learn the right response"
  (SFT) to "learn which response is PREFERRED" (DPO). Preference
  data replaces instruction pairs as the training signal.
"""
)
