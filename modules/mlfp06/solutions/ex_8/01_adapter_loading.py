# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 8.1: Load a Fine-Tuned Adapter via AdapterRegistry
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Use kailash-align's AdapterRegistry as the source of truth for
#     model provenance (base model, method, LoRA config, version)
#   - Register the adapters Ex 2.6 (SFT) and Ex 3.3 (DPO) trained, and
#     select one by a named preference — never by a hardcoded path
#   - Load the selected adapter for inference (AdapterMerger + the HF
#     generation backend) and score it on a few MMLU questions
#   - Visualise real adapter sizes against the base model
#   - Apply adapter loading to a Singapore HR compliance scenario
#
# PREREQUISITES: MLFP06 Ex 2.6 (SFT adapter), Ex 3.3 (DPO adapter) — this
#   file loads what those runs saved; it stops with instructions if none
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Load MMLU evaluation data
#   2. Register the trained adapters and pick the best available one
#   3. Merge the adapter into its base model and generate answers
#   4. Visualise the adapter catalogue and parameter counts
#   5. Apply to a Singapore HR compliance QA scenario
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import polars as pl
from kailash_align import AdapterMerger, AdapterRegistry, AdapterSignature
from kailash_align.exceptions import AdapterNotFoundError
from kailash_align.vllm_backend import HFGenerationBackend

from shared.mlfp06.ex_8 import (
    ADAPTER_SEARCH_ROOTS,
    OUTPUT_DIR,
    count_safetensors_params,
    discover_trained_adapters,
    load_mmlu_eval,
    run_async,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — Why an AdapterRegistry?
# ════════════════════════════════════════════════════════════════════════
# A production LLM platform trains many adapters: SFT for domain tone,
# DPO for preference alignment, merges that combine both. The
# AdapterRegistry is version control for those weights — it records
# which base model each adapter attaches to, how it was trained (LoRA
# rank, alpha, target modules), its metrics, and its version.
#
# Hardcoding a path like "./models/imdb_sft_v1" looks fine today and
# silently ships the wrong weights tomorrow. The registry turns model
# loading into a named lookup against a catalogue.
#
# One honest caveat: `AdapterRegistry()` built without a backing model
# registry lives in memory, so a new process starts empty. The adapters
# Ex 2.6 / 3.3 trained are still on disk (AlignmentPipeline saves them to
# <experiment_dir>/<adapter_name>/<method>/adapter/), so this file finds
# them there and registers them before looking anything up.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load MMLU evaluation data
# ════════════════════════════════════════════════════════════════════════

eval_data = load_mmlu_eval(n_rows=100)

print(f"\nEvaluation data (MMLU): {eval_data.shape}")
print(f"Subjects: {eval_data['subject'].n_unique()}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert eval_data.height > 0, "Task 1: MMLU should load at least 1 row"
assert "instruction" in eval_data.columns, "Task 1: expected 'instruction' column"
print("✓ Checkpoint 1 passed — evaluation data loaded\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Register the trained adapters, pick the best available one
# ════════════════════════════════════════════════════════════════════════

on_disk = discover_trained_adapters()
if not on_disk:
    raise FileNotFoundError(
        "No trained adapters found under "
        f"{[str(r) for r in ADAPTER_SEARCH_ROOTS]}. Run "
        "modules/mlfp06/solutions/ex_2/06_sft_alignment_pipeline.py and "
        "ex_3/03_dpo_training.py first (from the repo root)."
    )

# Preferred order: the DPO-aligned adapter, then the SFT adapter.
PREFERENCE = ("ultrafeedback_dpo_v1", "imdb_sentiment_sft_v1")


async def register_and_pick() -> tuple[AdapterRegistry, object]:
    """Register every on-disk adapter, then select by named preference."""
    registry = AdapterRegistry()
    for found in on_disk:
        await registry.register_adapter(
            name=found["adapter_name"],
            adapter_path=found["adapter_path"],
            signature=AdapterSignature(
                base_model_id=found["base_model_id"],
                rank=found["rank"],
                alpha=found["alpha"],
                target_modules=found["target_modules"],
                training_method=found["method"],
            ),
            tags=[found["method"]],
        )

    versions = await registry.list_adapters()
    print(f"Registered adapters: {len(versions)}")
    for av in versions:
        print(
            f"  {av.adapter_name:28s} v{av.version}  base={av.base_model_id}  "
            f"stage={av.stage}"
        )

    for name in PREFERENCE + tuple(av.adapter_name for av in versions):
        try:
            return registry, await registry.get_adapter(name)
        except AdapterNotFoundError:
            print(f"  (no adapter named {name!r} — trying the next preference)")
    raise AdapterNotFoundError("registry is empty after registration")


registry, best = run_async(register_and_pick())

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert best.adapter_name, "Task 2: an adapter must be selected"
assert best.base_model_id, "Task 2: the adapter must name its base model"
print(
    f"✓ Checkpoint 2 passed — selected {best.adapter_name} v{best.version} "
    f"(base {best.base_model_id})\n"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Load the adapter for inference and generate answers
# ════════════════════════════════════════════════════════════════════════
# AdapterMerger loads the base model, applies the LoRA adapter and
# merges the weights (merge_and_unload), then records merge_status and
# merged_model_path back in the registry. HFGenerationBackend then
# generates from the merged model on CPU/MPS/CUDA. (For an Ollama
# deployment, AlignmentServing.deploy_ollama goes one step further and
# exports a GGUF; that needs llama.cpp tooling and is not run here.)

merged_path = run_async(AdapterMerger(adapter_registry=registry).merge(best.adapter_name))
backend = HFGenerationBackend(model_id=str(merged_path))

N_EVAL = 5
sample = eval_data.head(N_EVAL)
prompts = [
    f"{q}\n\nAnswer with a single letter (A, B, C or D).\nAnswer:"
    for q in sample["instruction"].to_list()
]
completions = backend.batch_generate(prompts, max_new_tokens=4, temperature=0.0)
predicted = []
for c in completions:
    m = re.search(r"\b([ABCD])\b", c[0])
    predicted.append(m.group(1) if m else "?")
mmlu_acc = sum(p == g for p, g in zip(predicted, sample["response"].to_list())) / N_EVAL
backend.shutdown()

print(f"Merged model: {merged_path}")
for subj, p, g in zip(sample["subject"].to_list(), predicted, sample["response"].to_list()):
    print(f"  {subj:<32s} predicted={p}  gold={g}")
print(f"MMLU accuracy on {N_EVAL} questions: {mmlu_acc:.0%} (chance = 25%)")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert len(completions) == N_EVAL, "Task 3: one completion per prompt"
assert merged_path.exists(), "Task 3: the merged model should be on disk"
print("✓ Checkpoint 3 passed — adapter loaded and generating\n")
# INTERPRETATION: 5 questions is a smoke test, not a benchmark — one
# right or wrong answer moves accuracy by 20 points. The point is that
# the weights you serve are the registered ones, provably.


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise the adapter catalogue
# ════════════════════════════════════════════════════════════════════════

catalogue = pl.DataFrame(
    {
        "adapter_name": [f["adapter_name"] for f in on_disk],
        "method": [f["method"] for f in on_disk],
        "base_model_id": [f["base_model_id"] for f in on_disk],
        "lora_rank": [f["rank"] for f in on_disk],
        "target_modules": [",".join(f["target_modules"]) for f in on_disk],
        "trainable_params": [f["trainable_params"] for f in on_disk],
    }
)
catalogue.write_parquet(OUTPUT_DIR / "adapter_catalogue.parquet")
print("Adapter catalogue:")
print(catalogue)

# INTERPRETATION: The catalogue is the single pane of glass operations
# teams use for rollback. If a new adapter ships broken, they pick the
# previous row and redeploy by name.


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Adapter parameter count vs the base model
# ════════════════════════════════════════════════════════════════════════
# Both numbers are counted from the safetensors files on disk: the
# adapter's own weights, and the merged model (= base model size).

base_params = count_safetensors_params(merged_path)
names = catalogue["adapter_name"].to_list()
param_counts = catalogue["trainable_params"].to_list()

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(names, param_counts, color=["#3498db", "#2ecc71", "#e67e22"][: len(names)])
ax.axhline(
    base_params,
    color="#e74c3c",
    linestyle="--",
    linewidth=2,
    label=f"Base model ({best.base_model_id}): {base_params / 1e6:,.0f}M params",
)
ax.set_ylabel("Parameters")
ax.set_title("Adapter Parameters vs Base Model (counted from disk)", fontweight="bold")
ax.set_yscale("log")
for bar, count in zip(bars, param_counts):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        max(count, 1) * 1.5,
        f"{count / 1e6:.2f}M\n({count / base_params:.2%})",
        ha="center",
        fontsize=9,
    )
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3, which="both")
plt.tight_layout()
fname = OUTPUT_DIR / "ex8_adapter_params.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {fname}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: Singapore HR Compliance QA
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore SME (200 employees) runs an internal HR policy
# assistant tuned on Ministry of Manpower guidance (CPF, Work Pass
# categories, retrenchment notice). DPO alignment then discourages
# speculative or legally risky phrasing.
#
# BUSINESS IMPACT (illustrative figures): if external legal review of an
# HR question costs ~S$400, 200 questions a month cost S$80,000. A
# governed adapter answering first-line questions at a few cents each,
# with lawyers reviewing only escalations, cuts most of that — and the
# registry lets the HR team show which adapter version answered which
# question.

ILLUSTRATIVE_REVIEW_COST_SGD = 400
ILLUSTRATIVE_QUERIES_PER_MONTH = 200
print("\n" + "=" * 70)
print("  APPLY — Singapore HR Policy Assistant")
print("=" * 70)
print(
    f"""
  Base model:     {best.base_model_id}
  Active adapter: {best.adapter_name} v{best.version}
  LoRA config:    r={best.lora_config.get('r')}, alpha={best.lora_config.get('alpha')}, targets={best.lora_config.get('target_modules')}
  Smoke-test MMLU accuracy: {mmlu_acc:.0%} on {N_EVAL} questions

  Illustrative legal-review baseline: S${ILLUSTRATIVE_REVIEW_COST_SGD * ILLUSTRATIVE_QUERIES_PER_MONTH:,}/month
  ({ILLUSTRATIVE_QUERIES_PER_MONTH} questions at S${ILLUSTRATIVE_REVIEW_COST_SGD} each)

  Audit trail: log adapter name + version with every answer, so an
  inspection can reconstruct which model answered any policy question.
"""
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Registered trained adapters in AdapterRegistry with their signatures
  [x] Selected an adapter by named preference, not by a hardcoded path
  [x] Merged the adapter into its base model and generated real answers
  [x] Counted adapter vs base parameters from the files on disk
  [x] Applied adapter provenance to a Singapore HR compliance scenario

  KEY INSIGHT: A model you cannot name, version and trace to its
  training run is a model you cannot roll back or defend in an audit.

  Next: 02_governance_pipeline.py wraps the served model in PACT
  governance tiers.
"""
)
