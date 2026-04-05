# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT6 — Exercise 4: Advanced Alignment — Model Merging and Evaluation
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Merge the SFT adapter (Exercise 1) and DPO adapter (Exercise 2)
#   using TIES and DARE merging strategies. Evaluate all three variants
#   (SFT, DPO, merged) using LLM-as-judge, RAGAS-style metrics, and a
#   structured rubric. Determine which adapter is best for production.
#
# TASKS:
#   1. Load registered adapters from AdapterRegistry (Ex1 SFT, Ex2 DPO)
#   2. Merge adapters with TIES merging strategy
#   3. Merge adapters with DARE merging strategy
#   4. Evaluate all variants: SFT, DPO, TIES-merged, DARE-merged
#   5. Compare methods: quality, alignment, Singapore-domain accuracy
#   6. Select the best adapter and register it as production-ready
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl

from kailash_align import (
    AlignmentConfig,
    AlignmentPipeline,
    AdapterRegistry,
    merge,
    evaluator,
)
from kailash_align.merge import TIESConfig, DAREConfig, MergeResult
from kailash_align.evaluator import AlignmentEvaluator, EvalConfig, EvalResult

from kaizen_agents import Delegate

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

base_model = os.environ.get("SFT_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
judge_model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))


# ── Evaluation Dataset ────────────────────────────────────────────────

loader = ASCENTDataLoader()

# Load the same evaluation prompts used in Ex1 and Ex2
eval_prompts = [
    {
        "prompt": "What is the HDB resale flat application procedure in Singapore?",
        "category": "housing",
        "reference": (
            "The HDB resale process involves: (1) register intent to buy/sell on HDB portal, "
            "(2) grant of option (14 days), (3) exercise option (21 days), (4) submit resale "
            "application (both parties), (5) HDB approval (8 weeks), (6) completion appointment. "
            "Buyers must secure an HLE or bank loan before exercising the option."
        ),
    },
    {
        "prompt": "Explain CPF contribution rates for Singapore employees aged 35-45.",
        "category": "cpf",
        "reference": (
            "For employees aged 35-45, CPF contributions are: employer 16%, employee 20%, "
            "total 36%. Allocation: OA 23%, SA 6%, MA 8% (approximate). Contributions apply "
            "to ordinary wages up to $6,000/month and additional wages up to the annual limit."
        ),
    },
    {
        "prompt": "How does MAS regulate AI systems used in financial services?",
        "category": "regulation",
        "reference": (
            "MAS regulates AI in financial services through FEAT principles (Fairness, Ethics, "
            "Accountability, Transparency). Key requirements: model risk management framework, "
            "ongoing monitoring, explainability for customer-facing decisions (especially credit), "
            "board oversight of AI governance, and regular model validation by independent parties."
        ),
    },
    {
        "prompt": "What is Singapore's SingPass MyInfo system and how does it work?",
        "category": "digital_gov",
        "reference": (
            "MyInfo is a government-managed personal data platform. Citizens store verified "
            "personal data once; services retrieve it with consent. Data includes NRIC details, "
            "income records, CPF balances, and property ownership. Authentication via SingPass "
            "with 2FA. Pre-fills forms for banking, insurance, and government applications."
        ),
    },
    {
        "prompt": "Explain the key differences between HDB BTO, SBF, and resale flats.",
        "category": "housing",
        "reference": (
            "BTO (Build-To-Order): new flats built to demand, 3-5 year wait, subsidised price. "
            "SBF (Sale of Balance Flats): unsold BTO flats, shorter wait, similar subsidy. "
            "Resale: existing flats on open market, immediate occupancy, market price + CPF grant "
            "if eligible. BTO/SBF require citizenship and income ceiling; resale has fewer restrictions."
        ),
    },
]

print(f"=== Model Merging Exercise ===")
print(f"Base model: {base_model}")
print(f"Judge model: {judge_model}")
print(f"Evaluation prompts: {len(eval_prompts)}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Load registered adapters from AdapterRegistry
# ══════════════════════════════════════════════════════════════════════


async def load_adapters():
    """Load SFT and DPO adapters registered in Ex1 and Ex2."""
    registry = AdapterRegistry()

    all_adapters = await registry.list_adapters()
    print(f"\n=== AdapterRegistry ===")
    print(f"Registered adapters: {len(all_adapters)}")
    for a in all_adapters:
        print(
            f"  {a.get('name')}: {a.get('method')} "
            f"loss={a.get('metrics', {}).get('eval_loss', '?'):.4f}"
        )

    # Load by name (registered in Ex1 and Ex2)
    sft_adapter = await registry.get_adapter("sg_domain_sft_v1")
    dpo_adapter = await registry.get_adapter("sg_domain_dpo_v1")

    print(f"\nLoaded adapters:")
    print(
        f"  SFT:  {sft_adapter['adapter_path']}  (LoRA-r16, eval_loss={sft_adapter['metrics'].get('eval_loss', '?'):.4f})"
    )
    print(
        f"  DPO:  {dpo_adapter['adapter_path']}  (LoRA-r16, β=0.1, eval_loss={dpo_adapter['metrics'].get('eval_loss', '?'):.4f})"
    )

    return registry, sft_adapter, dpo_adapter


registry, sft_adapter, dpo_adapter = asyncio.run(load_adapters())


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Merge with TIES (Task-Interference Elimination Strategy)
# ══════════════════════════════════════════════════════════════════════
#
# TIES merging algorithm:
#   1. Trim: zero out redundant (low-magnitude) delta weights per adapter
#   2. Elect sign: resolve sign conflicts via majority vote across adapters
#   3. Disjoint merge: merge only parameters where the adapters agree on sign
#
# Key parameter: density (fraction of weights kept after trimming)
#   density=1.0 → no trimming (same as linear averaging)
#   density=0.2 → keep top 20% of delta weights by magnitude


async def merge_ties():
    """Merge SFT + DPO adapters using TIES strategy."""
    print(f"\n=== TIES Merging ===")
    print(f"Strategy: Trim → Elect sign → Disjoint merge")

    ties_config = TIESConfig(
        base_model=base_model,
        adapters=[
            {"path": sft_adapter["adapter_path"], "weight": 0.6, "name": "sft"},
            {"path": dpo_adapter["adapter_path"], "weight": 0.4, "name": "dpo"},
        ],
        density=0.3,  # Keep top 30% of delta weights
        merge_coefficient=1.0,  # Scaling applied to merged delta
    )

    result: MergeResult = await merge.ties(ties_config)

    print(f"TIES merge complete:")
    print(f"  Merged adapter path: {result.adapter_path}")
    print(f"  Parameters merged:   {result.parameters_merged:,}")
    print(f"  Conflicts resolved:  {result.conflicts_resolved:,}")
    print(f"  Sparsity achieved:   {result.sparsity:.1%}")
    print(f"  Merge time:          {result.merge_time_seconds:.1f}s")

    return result


ties_result = asyncio.run(merge_ties())


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Merge with DARE (Drop And REscale)
# ══════════════════════════════════════════════════════════════════════
#
# DARE merging algorithm:
#   1. Drop: randomly zero out delta weights with probability p (drop_rate)
#   2. Rescale: multiply remaining weights by 1/(1-p) to preserve expectation
#
# This is simpler than TIES but often competitive. The stochastic dropping
# acts as a regulariser that reduces interference between adapters.
# Combines well with high-density TIES for an ensemble effect.


async def merge_dare():
    """Merge SFT + DPO adapters using DARE strategy."""
    print(f"\n=== DARE Merging ===")
    print(f"Strategy: Drop (random zeros) → Rescale (1/(1-p))")

    dare_config = DAREConfig(
        base_model=base_model,
        adapters=[
            {"path": sft_adapter["adapter_path"], "weight": 0.5, "name": "sft"},
            {"path": dpo_adapter["adapter_path"], "weight": 0.5, "name": "dpo"},
        ],
        drop_rate=0.1,  # Drop 10% of delta weights randomly
        rescale=True,  # Rescale by 1/(1-drop_rate) to preserve E[δ]
        seed=42,
    )

    result: MergeResult = await merge.dare(dare_config)

    print(f"DARE merge complete:")
    print(f"  Merged adapter path: {result.adapter_path}")
    print(f"  Parameters merged:   {result.parameters_merged:,}")
    print(f"  Weights dropped:     {result.weights_dropped:,} ({result.sparsity:.1%})")
    print(f"  Merge time:          {result.merge_time_seconds:.1f}s")

    return result


dare_result = asyncio.run(merge_dare())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Evaluate all four variants
# ══════════════════════════════════════════════════════════════════════
#
# Evaluation dimensions:
#   1. Faithfulness  — does the answer contain only grounded claims?
#   2. Relevance     — does the answer address the question?
#   3. Singapore accuracy — does the answer reflect SG-specific facts?
#   4. LLM-as-judge  — pairwise comparison judged by a stronger model
#   5. Self-rated confidence — internal model confidence (if available)


async def evaluate_all_variants():
    """Run structured evaluation across all four adapter variants."""

    eval_config = EvalConfig(
        base_model=base_model,
        judge_model=judge_model,
        judge_cost_budget=3.0,
        metrics=["faithfulness", "relevance", "domain_accuracy"],
        pairwise_comparison=True,  # Judge ranks all variants head-to-head
        n_eval_prompts=len(eval_prompts),
    )

    evaluator_instance = AlignmentEvaluator(eval_config)

    variants = {
        "base_model": None,  # No adapter (baseline)
        "sft_v1": sft_adapter["adapter_path"],
        "dpo_v1": dpo_adapter["adapter_path"],
        "ties_merged": ties_result.adapter_path,
        "dare_merged": dare_result.adapter_path,
    }

    print(f"\n=== Running Evaluation (5 variants × {len(eval_prompts)} prompts) ===")

    results: dict[str, EvalResult] = {}
    for variant_name, adapter_path in variants.items():
        print(f"  Evaluating: {variant_name}...")
        result = await evaluator_instance.evaluate(
            adapter_path=adapter_path,
            eval_prompts=eval_prompts,
        )
        results[variant_name] = result
        print(
            f"    faithfulness={result.faithfulness:.3f}  "
            f"relevance={result.relevance:.3f}  "
            f"domain_accuracy={result.domain_accuracy:.3f}  "
            f"judge_score={result.judge_score:.3f}"
        )

    return results


eval_results = asyncio.run(evaluate_all_variants())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Comparison table and analysis
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Evaluation Results ===")
print(f"{'Variant':<16} {'Faithful':>9} {'Relevant':>9} {'Domain':>9} {'Judge':>9}")
print("─" * 56)
for name, r in eval_results.items():
    print(
        f"{name:<16} {r.faithfulness:>9.3f} {r.relevance:>9.3f} "
        f"{r.domain_accuracy:>9.3f} {r.judge_score:>9.3f}"
    )

# Find the best variant
best_name = max(
    eval_results,
    key=lambda n: eval_results[n].judge_score,
)
best = eval_results[best_name]

print(f"\nBest variant: {best_name}")
print(f"  Faithfulness:    {best.faithfulness:.3f}")
print(f"  Relevance:       {best.relevance:.3f}")
print(f"  Domain accuracy: {best.domain_accuracy:.3f}")
print(f"  Judge score:     {best.judge_score:.3f}")

print(f"\nMethod comparison:")
print(
    """
SFT adapter:
  + Good task format adherence (instruction-following)
  - No explicit preference signal; may still produce suboptimal responses
  + Fast to train, stable loss curve

DPO adapter:
  + Preference signal drives output closer to human-preferred style
  - Requires preference pairs (harder to collect than instruction data)
  + Better alignment on contentious/ambiguous queries

TIES-merged:
  + Combines task format (SFT) + preference alignment (DPO)
  + Trimming reduces inter-adapter interference
  - Requires careful density tuning; too low loses SFT signal
  ✓ Best choice when both instruction quality AND alignment matter

DARE-merged:
  + Simplest merging strategy, strong regularisation effect
  + Stochastic dropping creates implicit ensemble
  - Less principled conflict resolution than TIES
  ✓ Good fallback when TIES is unstable
"""
)


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Register the best adapter as production-ready
# ══════════════════════════════════════════════════════════════════════


async def register_production_adapter():
    """Register winning adapter with production tag."""
    best_result = eval_results[best_name]
    adapter_path = (
        ties_result.adapter_path
        if best_name == "ties_merged"
        else (
            dare_result.adapter_path
            if best_name == "dare_merged"
            else (
                sft_adapter["adapter_path"]
                if best_name == "sft_v1"
                else dpo_adapter["adapter_path"]
            )
        )
    )

    prod_id = await registry.register(
        name="sg_domain_production_v1",
        base_model=base_model,
        method=f"merged_{best_name}",
        adapter_path=adapter_path,
        metrics={
            "faithfulness": best_result.faithfulness,
            "relevance": best_result.relevance,
            "domain_accuracy": best_result.domain_accuracy,
            "judge_score": best_result.judge_score,
        },
        tags=["singapore", "domain-qa", "production", best_name],
        stage="production",
    )

    print(f"\n=== Production Adapter Registered ===")
    print(f"ID:     {prod_id}")
    print(f"Name:   sg_domain_production_v1")
    print(f"Method: merged_{best_name}")
    print(f"Stage:  production")

    # Final registry listing
    all_adapters = await registry.list_adapters()
    print(f"\nAll registered adapters ({len(all_adapters)}):")
    for a in all_adapters:
        stage = " [PRODUCTION]" if a.get("stage") == "production" else ""
        print(f"  {a.get('name')}: {a.get('method')}{stage}")

    return prod_id


production_id = asyncio.run(register_production_adapter())

print(f"\n=== Key Takeaways ===")
print(
    """
1. Merging is NOT fine-tuning again:
   → No new training data needed
   → No GPU time required
   → Combine capabilities from separate training runs

2. TIES vs DARE:
   → TIES: deterministic, sign-conflict resolution, better for diverse adapters
   → DARE: stochastic, regularisation, better for similar adapters

3. Evaluation trumps intuition:
   → "Merged should be better" is not always true
   → Run structured evaluation before promoting to production
   → LLM-as-judge captures quality the training loss cannot

4. AdapterRegistry as source of truth:
   → Every adapter tagged with provenance (method, base model, metrics)
   → Production adapter promoted explicitly — no silent overwrites
   → Enables rollback: previous adapter is never deleted
"""
)

print(
    "\n✓ Exercise 4 complete — model merging (TIES + DARE) with structured evaluation"
)
