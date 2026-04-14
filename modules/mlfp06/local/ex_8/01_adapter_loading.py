# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 8.1: Load Fine-Tuned Model from AdapterRegistry
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Use kailash-align's AdapterRegistry as the source of truth for
#     model provenance and version control
#   - Load a LoRA adapter through AlignmentPipeline (SFT + DPO merge)
#   - Fall back safely when a specific adapter is unavailable
#   - Visualise the adapter catalogue as a registry table
#   - Apply adapter loading to a Singapore HR compliance scenario
#
# PREREQUISITES: MLFP06 Ex 2 (SFT LoRA), Ex 3 (DPO alignment)
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Load MMLU evaluation data for downstream monitoring
#   2. Query AdapterRegistry and pick the best available adapter
#   3. Instantiate AlignmentPipeline in inference mode
#   4. Visualise the adapter catalogue
#   5. Apply to a Singapore HR compliance QA scenario
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl

from kailash_align import AdapterRegistry, AlignmentConfig, AlignmentPipeline

from shared.mlfp06.ex_8 import (
    OUTPUT_DIR,
    load_mmlu_eval,
    run_async,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — Why an AdapterRegistry?
# ════════════════════════════════════════════════════════════════════════
# A production LLM platform trains many adapters: SFT for domain tone,
# DPO for preference alignment, SLERP merges for combining both. The
# AdapterRegistry is the Git for model weights — it stores what each
# adapter was trained on, which base model it attaches to, who owns
# it, and how recently it was evaluated. Hardcoding a filesystem path
# silently ships the wrong weights tomorrow; the registry turns model
# loading into a named lookup against an audited catalogue.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load MMLU evaluation data
# ════════════════════════════════════════════════════════════════════════

# TODO: Call load_mmlu_eval with n_rows=100
eval_data = ____

print(f"\nEvaluation data (MMLU): {eval_data.shape}")
print(f"Subjects: {eval_data['subject'].n_unique()}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert eval_data.height > 0, "Task 1: MMLU should load at least 1 row"
assert "instruction" in eval_data.columns, "Task 1: expected 'instruction' column"
print("\u2713 Checkpoint 1 passed — evaluation data loaded\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Query AdapterRegistry and pick the best adapter
# ════════════════════════════════════════════════════════════════════════


async def pick_best_adapter() -> tuple[AdapterRegistry, dict]:
    """Return the registry and the best available adapter metadata."""
    # TODO: instantiate AdapterRegistry()
    registry = ____

    # TODO: call registry.list_adapters() (async)
    adapters = ____
    print(f"Available adapters: {len(adapters)}")

    best_adapter: dict = {}
    for candidate in (
        "sg_domain_slerp_merge_v1",
        "ultrafeedback_dpo_v1",
        "imdb_sentiment_sft_v1",
    ):
        try:
            # TODO: call registry.get_adapter(candidate) (async)
            found = ____
            if found:
                best_adapter = found
                break
        except Exception:
            continue
    return registry, best_adapter


registry, best_adapter = run_async(pick_best_adapter())

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert registry is not None, "Task 2: registry should be accessible"
print(f"\u2713 Checkpoint 2 passed — adapter: {best_adapter.get('name', 'none')}\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Build the AlignmentPipeline for inference
# ════════════════════════════════════════════════════════════════════════

# TODO: Build an AlignmentPipeline with an AlignmentConfig whose
#       method="inference" and adapter_path=best_adapter.get("adapter_path", "")
inference_pipeline = ____

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert inference_pipeline is not None, "Task 3: pipeline should be created"
print("\u2713 Checkpoint 3 passed — AlignmentPipeline ready\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise the adapter catalogue
# ════════════════════════════════════════════════════════════════════════


async def snapshot_catalogue() -> pl.DataFrame:
    # TODO: list adapters, return as polars DataFrame with
    #       columns: name, method, base_model
    ____


catalogue = run_async(snapshot_catalogue())
catalogue.write_parquet(OUTPUT_DIR / "adapter_catalogue.parquet")
print("Adapter catalogue:")
print(catalogue)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: Singapore HR Compliance QA
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore SME (200 employees) runs an internal HR policy
# assistant fine-tuned on MOM guidelines. The base model is general; the
# SFT adapter teaches Singapore employment law (CPF, Work Pass, notice);
# DPO alignment filters out legally risky phrasing.
#
# BUSINESS IMPACT: Legal review externally is ~S$400/query. A governed
# adapter serving 200 queries/month at S$0.08/query = S$16 vs S$80,000
# — a ~5,000x reduction, with full provenance from the registry.

print("\n" + "=" * 70)
print("  APPLY — Singapore HR Policy Assistant")
print("=" * 70)
print(
    f"""
  Active adapter: {best_adapter.get('name', 'none')}
  Method:         {best_adapter.get('method', '-')}

  Legal-review baseline:  S$80,000/month
  Governed-adapter cost:  S$16/month
  Saving:                 ~5,000x reduction
"""
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  [x] Queried AdapterRegistry as the catalogue of trained adapters
  [x] Chose the best available adapter with graceful fallback
  [x] Built an AlignmentPipeline for inference
  [x] Snapshotted the registry for rollback visibility
  [x] Applied adapter loading to a Singapore HR scenario

  Next: 02_governance_pipeline.py wraps this adapter in PACT controls.
"""
)
