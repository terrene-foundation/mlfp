# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Align / Adapter Registry
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Version and manage LoRA adapters
# LEVEL: Advanced
# PARITY: Full — Rust has equivalent AdapterRegistry
# VALIDATES: AdapterRegistry, AdapterSignature, register, list, load,
#            promote, delete, capacity bounds
#
# Run: uv run python textbook/python/07-align/06_adapter_registry.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

from kailash_align import AdapterRegistry, AdapterSignature
from kailash_align.exceptions import AdapterNotFoundError, AlignmentError
from kailash_align.registry import AdapterVersion

model_id = os.environ.get("DEFAULT_LLM_MODEL", "meta-llama/Llama-3.1-8B")


async def main() -> None:
    # ── 1. Create an AdapterRegistry ────────────────────────────────

    registry = AdapterRegistry()

    # Registry accepts optional model_registry and capacity bounds
    bounded_registry = AdapterRegistry(
        max_adapters=100,
        max_versions_per_adapter=10,
    )

    # ── 2. AdapterSignature describes an adapter's characteristics ──

    signature = AdapterSignature(
        base_model_id=model_id,
        adapter_type="lora",
        rank=16,
        alpha=32,
        target_modules=("q_proj", "v_proj"),
        task_type="CAUSAL_LM",
        training_method="sft",
    )

    assert signature.base_model_id == model_id
    assert signature.rank == 16
    assert signature.alpha == 32
    assert signature.adapter_type == "lora"

    # base_model_id is required
    try:
        AdapterSignature(base_model_id="")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected: base_model_id is required

    # adapter_type restricted to lora or qlora
    try:
        AdapterSignature(base_model_id=model_id, adapter_type="invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # rank must be >= 1
    try:
        AdapterSignature(base_model_id=model_id, rank=0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # target_modules must not be empty
    try:
        AdapterSignature(base_model_id=model_id, target_modules=())
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # training_method is validated against METHOD_REGISTRY
    try:
        AdapterSignature(base_model_id=model_id, training_method="nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # sft_then_dpo is accepted as a valid training method
    combo_sig = AdapterSignature(base_model_id=model_id, training_method="sft_then_dpo")
    assert combo_sig.training_method == "sft_then_dpo"

    # ── 3. Register an adapter version ──────────────────────────────

    version1 = await registry.register_adapter(
        name="my-sft-adapter",
        adapter_path="/tmp/adapters/my-sft-adapter/v1",
        signature=signature,
        training_metrics={"train_loss": 0.45, "eval_loss": 0.52},
        tags=["sft", "experiment-1"],
    )

    assert isinstance(version1, AdapterVersion)
    assert version1.adapter_name == "my-sft-adapter"
    assert version1.version == "1"
    assert version1.stage == "staging"
    assert version1.merge_status == "separate"
    assert version1.training_metrics["train_loss"] == 0.45
    assert version1.lora_config["r"] == 16
    assert version1.lora_config["alpha"] == 32

    # ── 4. Register a second version — auto-increments ──────────────

    version2 = await registry.register_adapter(
        name="my-sft-adapter",
        adapter_path="/tmp/adapters/my-sft-adapter/v2",
        signature=signature,
        training_metrics={"train_loss": 0.38, "eval_loss": 0.44},
    )

    assert version2.version == "2", "Second version auto-incremented"
    assert version2.training_metrics["train_loss"] == 0.38

    # ── 5. Get adapter — latest version by default ──────────────────

    latest = await registry.get_adapter("my-sft-adapter")
    assert latest.version == "2", "Default: returns latest version"

    specific = await registry.get_adapter("my-sft-adapter", version="1")
    assert specific.version == "1"

    # Non-existent adapter raises AdapterNotFoundError
    try:
        await registry.get_adapter("nonexistent")
        assert False, "Should have raised AdapterNotFoundError"
    except AdapterNotFoundError:
        pass

    # Non-existent version raises AdapterNotFoundError
    try:
        await registry.get_adapter("my-sft-adapter", version="999")
        assert False, "Should have raised AdapterNotFoundError"
    except AdapterNotFoundError:
        pass

    # ── 6. List adapters — with optional filters ────────────────────

    all_adapters = await registry.list_adapters()
    assert len(all_adapters) == 1, "One adapter registered"

    # Filter by base_model_id
    by_model = await registry.list_adapters(base_model_id=model_id)
    assert len(by_model) == 1

    no_match = await registry.list_adapters(base_model_id="nonexistent/model")
    assert len(no_match) == 0

    # Filter by tags
    by_tags = await registry.list_adapters(tags=["sft"])
    assert len(by_tags) == 1

    no_tag_match = await registry.list_adapters(tags=["nonexistent-tag"])
    assert len(no_tag_match) == 0

    # ── 7. Promote through stages (monotonic) ──────────────────────
    # Stage order: staging -> shadow -> production -> archived

    promoted = await registry.promote("my-sft-adapter", "1", "shadow")
    assert promoted.stage == "shadow"

    promoted = await registry.promote("my-sft-adapter", "1", "production")
    assert promoted.stage == "production"

    # Cannot go backward
    try:
        await registry.promote("my-sft-adapter", "1", "staging")
        assert False, "Should have raised AlignmentError"
    except AlignmentError:
        pass  # Expected: backward transition not allowed

    # Cannot promote to same stage
    try:
        await registry.promote("my-sft-adapter", "1", "production")
        assert False, "Should have raised AlignmentError"
    except AlignmentError:
        pass  # Expected: same stage not allowed

    # Invalid stage name
    try:
        await registry.promote("my-sft-adapter", "1", "invalid_stage")
        assert False, "Should have raised AlignmentError"
    except AlignmentError:
        pass

    # ── 8. Filter by stage ──────────────────────────────────────────

    by_stage = await registry.list_adapters(stage="staging")
    # version 2 is still in staging
    assert len(by_stage) == 1

    production = await registry.list_adapters(stage="production")
    # version 1 is in production but list_adapters returns latest per adapter
    # in the given stage — so this should find the production version
    assert len(production) == 1
    assert production[0].stage == "production"

    # ── 9. Update merge status ──────────────────────────────────────

    merged = await registry.update_merge_status(
        "my-sft-adapter", "1", "merged", "/tmp/merged/my-sft-adapter"
    )
    assert merged.merge_status == "merged"
    assert merged.merged_model_path == "/tmp/merged/my-sft-adapter"

    # Invalid merge_status
    try:
        await registry.update_merge_status("my-sft-adapter", "1", "invalid")
        assert False, "Should have raised AlignmentError"
    except AlignmentError:
        pass

    # ── 10. Update GGUF path ────────────────────────────────────────

    exported = await registry.update_gguf_path(
        "my-sft-adapter",
        "1",
        "/tmp/gguf/my-sft-adapter.gguf",
        quantization_config={"method": "q4_k_m"},
    )
    assert exported.gguf_path == "/tmp/gguf/my-sft-adapter.gguf"
    assert exported.merge_status == "exported"
    assert exported.quantization_config == {"method": "q4_k_m"}

    # ── 11. Update eval results ─────────────────────────────────────

    evaled = await registry.update_eval_results(
        "my-sft-adapter",
        "2",
        {"arc_easy": 0.72, "hellaswag": 0.65},
    )
    assert evaled.eval_results == {"arc_easy": 0.72, "hellaswag": 0.65}

    # ── 12. Delete a specific version ───────────────────────────────

    await registry.delete_adapter("my-sft-adapter", version="2")

    try:
        await registry.get_adapter("my-sft-adapter", version="2")
        assert False, "Should have raised AdapterNotFoundError"
    except AdapterNotFoundError:
        pass  # Expected: deleted

    # Version 1 still exists
    v1 = await registry.get_adapter("my-sft-adapter", version="1")
    assert v1.version == "1"

    # ── 13. Delete entire adapter ───────────────────────────────────

    await registry.delete_adapter("my-sft-adapter")

    try:
        await registry.get_adapter("my-sft-adapter")
        assert False, "Should have raised AdapterNotFoundError"
    except AdapterNotFoundError:
        pass  # Expected: entire adapter deleted

    # ── 14. Capacity bounds — prevents OOM ──────────────────────────

    small_registry = AdapterRegistry(max_adapters=2, max_versions_per_adapter=2)

    sig = AdapterSignature(base_model_id=model_id)

    await small_registry.register_adapter("a1", "/tmp/a1", sig)
    await small_registry.register_adapter("a2", "/tmp/a2", sig)

    # Third adapter exceeds max_adapters
    try:
        await small_registry.register_adapter("a3", "/tmp/a3", sig)
        assert False, "Should have raised AlignmentError"
    except AlignmentError as e:
        assert "maximum" in str(e).lower()

    # Two versions of a1
    await small_registry.register_adapter("a1", "/tmp/a1-v2", sig)

    # Third version exceeds max_versions_per_adapter
    try:
        await small_registry.register_adapter("a1", "/tmp/a1-v3", sig)
        assert False, "Should have raised AlignmentError"
    except AlignmentError as e:
        assert "maximum" in str(e).lower()


asyncio.run(main())
print("PASS: 07-align/06_adapter_registry")
