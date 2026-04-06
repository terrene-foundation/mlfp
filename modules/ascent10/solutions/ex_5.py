# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT10 — Exercise 5: Model Merging and Export
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Merge multiple LoRA adapters (SFT + DPO) using different
#   strategies, export to ONNX, and compare merged model quality.
#
# TASKS:
#   1. Load SFT and DPO adapters from AdapterRegistry
#   2. Merge with linear interpolation (weighted average)
#   3. Merge with SLERP strategy
#   4. Compare merged models on evaluation set
#   5. Export best merged model to ONNX
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import math

import polars as pl

from kailash_align import AdapterRegistry, AlignmentConfig, AlignmentPipeline
from kailash_ml import ModelVisualizer, OnnxBridge

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Load adapters from AdapterRegistry
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
eval_data = loader.load("ascent10", "sg_domain_qa.parquet")

print(f"=== Evaluation Dataset ===")
print(f"Shape: {eval_data.shape}")


async def load_adapters():
    registry = AdapterRegistry()

    # List available adapters
    adapters = await registry.list_adapters()
    print(f"\n=== Registered Adapters ===")
    for a in adapters:
        print(
            f"  {a.get('name', '?')}: method={a.get('method', '?')}, "
            f"base={a.get('base_model', '?')}"
        )

    # Load SFT adapter (from Exercise 1)
    sft_adapter = await registry.get_adapter("sg_domain_sft_v1")
    # Load DPO adapter (from Exercise 2)
    dpo_adapter = await registry.get_adapter("sg_domain_dpo_v1")

    print(f"\nSFT adapter: {sft_adapter.get('name', 'N/A')}")
    print(f"  Loss: {sft_adapter.get('metrics', {}).get('eval_loss', 'N/A')}")
    print(f"DPO adapter: {dpo_adapter.get('name', 'N/A')}")
    print(f"  Loss: {dpo_adapter.get('metrics', {}).get('eval_loss', 'N/A')}")

    return registry, sft_adapter, dpo_adapter


registry, sft_adapter, dpo_adapter = asyncio.run(load_adapters())


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Merge with linear interpolation
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Linear Merge (Weighted Average) ===")
print(f"W_merged = alpha * W_sft + (1 - alpha) * W_dpo")
print(f"Simple but effective for combining complementary adapters.")


async def linear_merge():
    pipeline = AlignmentPipeline(
        AlignmentConfig(
            method="merge",
            merge_strategy="linear",
            merge_weights=[0.6, 0.4],  # 60% SFT, 40% DPO
            adapters=[
                sft_adapter.get("adapter_path", ""),
                dpo_adapter.get("adapter_path", ""),
            ],
            output_dir="./linear_merged",
        )
    )

    result = await pipeline.merge()

    print(f"Linear merge complete:")
    print(f"  Weights: SFT=0.6, DPO=0.4")
    print(f"  Output: {result.merged_path}")
    return result


linear_result = asyncio.run(linear_merge())


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Merge with SLERP strategy
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== SLERP Merge (Spherical Linear Interpolation) ===")
print(f"Unlike linear interpolation, SLERP interpolates along the")
print(f"hypersphere surface, preserving the magnitude of weight vectors.")
print(f"Better for merging adapters with different training dynamics.")


def slerp(v0: list[float], v1: list[float], t: float) -> list[float]:
    """Spherical linear interpolation between two vectors."""
    # Normalize
    norm0 = math.sqrt(sum(x * x for x in v0))
    norm1 = math.sqrt(sum(x * x for x in v1))
    v0_n = [x / (norm0 + 1e-10) for x in v0]
    v1_n = [x / (norm1 + 1e-10) for x in v1]

    # Angle between vectors
    dot = sum(a * b for a, b in zip(v0_n, v1_n))
    dot = max(-1.0, min(1.0, dot))
    omega = math.acos(dot)

    if abs(omega) < 1e-10:
        # Vectors are parallel — fall back to linear
        return [a * (1 - t) + b * t for a, b in zip(v0, v1)]

    sin_omega = math.sin(omega)
    s0 = math.sin((1 - t) * omega) / sin_omega
    s1 = math.sin(t * omega) / sin_omega

    # Interpolate with original magnitudes
    target_norm = norm0 * (1 - t) + norm1 * t
    result = [s0 * a + s1 * b for a, b in zip(v0_n, v1_n)]
    result_norm = math.sqrt(sum(x * x for x in result))
    return [x * target_norm / (result_norm + 1e-10) for x in result]


# Demonstrate SLERP on small example
v0 = [1.0, 0.0, 0.0]
v1 = [0.0, 1.0, 0.0]
for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    interp = slerp(v0, v1, t)
    magnitude = math.sqrt(sum(x * x for x in interp))
    print(f"  t={t:.2f}: {[f'{x:.3f}' for x in interp]} (|v|={magnitude:.3f})")

print(f"\nNotice: SLERP maintains unit magnitude throughout the interpolation,")
print(f"unlike linear interpolation which shrinks to |v|=0.707 at t=0.5.")


async def slerp_merge():
    pipeline = AlignmentPipeline(
        AlignmentConfig(
            method="merge",
            merge_strategy="slerp",
            merge_t=0.5,  # Interpolation factor
            adapters=[
                sft_adapter.get("adapter_path", ""),
                dpo_adapter.get("adapter_path", ""),
            ],
            output_dir="./slerp_merged",
        )
    )

    result = await pipeline.merge()
    print(f"\nSLERP merge complete: {result.merged_path}")
    return result


slerp_result = asyncio.run(slerp_merge())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Compare merged models on evaluation set
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Model Comparison ===")


async def evaluate_models():
    eval_questions = eval_data["instruction"].to_list()[:10]
    eval_references = eval_data["response"].to_list()[:10]

    results = {
        "linear_merge": {"path": linear_result.merged_path, "scores": []},
        "slerp_merge": {"path": slerp_result.merged_path, "scores": []},
    }

    for name, info in results.items():
        pipeline = AlignmentPipeline(
            AlignmentConfig(
                method="inference",
                adapter_path=info["path"],
            )
        )

        print(f"\n--- {name} ---")
        for i, (q, ref) in enumerate(zip(eval_questions[:3], eval_references[:3])):
            response = await pipeline.generate(q)
            # Simple similarity score (word overlap)
            resp_words = set(response.lower().split())
            ref_words = set(ref.lower().split())
            overlap = len(resp_words & ref_words) / max(len(ref_words), 1)
            info["scores"].append(overlap)
            print(f"  Q{i+1}: overlap={overlap:.3f}")

        avg_score = sum(info["scores"]) / len(info["scores"]) if info["scores"] else 0
        print(f"  Average overlap: {avg_score:.3f}")

    return results


model_results = asyncio.run(evaluate_models())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Export best merged model to ONNX
# ══════════════════════════════════════════════════════════════════════


async def export_best():
    # Pick the better merge strategy
    linear_avg = sum(model_results["linear_merge"]["scores"]) / max(
        len(model_results["linear_merge"]["scores"]), 1
    )
    slerp_avg = sum(model_results["slerp_merge"]["scores"]) / max(
        len(model_results["slerp_merge"]["scores"]), 1
    )

    best_name = "slerp_merge" if slerp_avg >= linear_avg else "linear_merge"
    best_path = model_results[best_name]["path"]

    print(f"\n=== Best Model: {best_name} ===")
    print(f"Linear avg score: {linear_avg:.3f}")
    print(f"SLERP avg score: {slerp_avg:.3f}")

    bridge = OnnxBridge()
    onnx_path = bridge.export(
        model=best_path,
        input_shape=(1, 512),  # sequence length
        output_path="merged_model.onnx",
    )

    print(f"\nExported to ONNX: {onnx_path}")
    print(f"ONNX model can be served via InferenceServer or deployed to edge devices.")

    # Register merged adapter
    adapter_id = await registry.register(
        name=f"sg_domain_{best_name}_v1",
        base_model=sft_adapter.get("base_model", "unknown"),
        method=f"merged_{best_name}",
        adapter_path=best_path,
        metrics={"avg_overlap": max(linear_avg, slerp_avg)},
        tags=["merged", "production"],
    )
    print(f"Registered merged adapter: {adapter_id}")

    return onnx_path


onnx_path = asyncio.run(export_best())

print(f"\n=== Model Merging Summary ===")
print(f"Linear merge: simple weighted average, good baseline")
print(f"SLERP merge: preserves magnitude, better for diverse adapters")
print(f"Other strategies: TIES (trim+elect+merge), DARE (drop+rescale)")
print(f"Pipeline: train adapters → merge → evaluate → export ONNX → deploy")

print("\n✓ Exercise 5 complete — model merging (linear + SLERP) with ONNX export")
