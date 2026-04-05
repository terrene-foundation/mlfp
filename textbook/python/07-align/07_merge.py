# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Align / Adapter Merge
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Merge multiple LoRA adapters (merge_and_unload strategy)
# LEVEL: Advanced
# PARITY: Python-only
# VALIDATES: AdapterMerger, merge lifecycle, registry integration
#
# Run: uv run python textbook/python/07-align/07_merge.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

from kailash_align import AdapterMerger, AdapterRegistry, AdapterSignature
from kailash_align.exceptions import MergeError
from kailash_align.registry import AdapterVersion

model_id = os.environ.get("DEFAULT_LLM_MODEL", "meta-llama/Llama-3.1-8B")


async def main() -> None:
    # ── 1. AdapterMerger creation ───────────────────────────────────
    # AdapterMerger requires an AdapterRegistry to look up adapters
    # and update their merge status.

    registry = AdapterRegistry()
    merger = AdapterMerger(adapter_registry=registry)

    assert merger._registry is registry

    # ── 2. Merger without registry raises MergeError ────────────────

    no_registry_merger = AdapterMerger()

    try:
        await no_registry_merger.merge("any-adapter")
        assert False, "Should have raised MergeError"
    except MergeError as e:
        assert "AdapterRegistry" in str(e)

    # ── 3. Merge lifecycle: separate -> merged -> exported ──────────
    # The merge status tracks where an adapter is in its lifecycle.
    # This section validates the status tracking without loading models.

    sig = AdapterSignature(base_model_id=model_id)
    adapter = await registry.register_adapter(
        name="merge-test",
        adapter_path="/tmp/adapters/merge-test/v1",
        signature=sig,
        training_metrics={"train_loss": 0.3},
    )

    assert adapter.merge_status == "separate", "New adapters start as 'separate'"
    assert adapter.merged_model_path is None
    assert adapter.gguf_path is None

    # Simulate merge completion via registry update
    merged = await registry.update_merge_status(
        "merge-test", "1", "merged", "/tmp/merged/merge-test"
    )
    assert merged.merge_status == "merged"
    assert merged.merged_model_path == "/tmp/merged/merge-test"

    # Simulate GGUF export via registry update
    exported = await registry.update_gguf_path(
        "merge-test",
        "1",
        "/tmp/gguf/merge-test.gguf",
        quantization_config={"method": "q4_k_m"},
    )
    assert exported.merge_status == "exported"
    assert exported.gguf_path == "/tmp/gguf/merge-test.gguf"
    assert exported.quantization_config == {"method": "q4_k_m"}

    # ── 4. Valid merge_status values ────────────────────────────────

    for valid_status in ("separate", "merged", "exported"):
        # Reset to separate first
        adapter2 = await registry.register_adapter(
            name=f"status-test-{valid_status}",
            adapter_path=f"/tmp/adapters/status-test-{valid_status}",
            signature=sig,
        )
        assert adapter2.merge_status == "separate"
        result = await registry.update_merge_status(
            f"status-test-{valid_status}", "1", valid_status
        )
        assert result.merge_status == valid_status

    # Invalid merge_status
    try:
        await registry.update_merge_status("merge-test", "1", "invalid")
        assert False, "Should have raised AlignmentError"
    except Exception as e:
        assert "invalid" in str(e).lower()

    # ── 5. Merge function — convenience wrapper ─────────────────────
    # merge_adapter() is a standalone convenience function.

    from kailash_align.merge import merge_adapter

    # Without registry, raises MergeError
    try:
        await merge_adapter("any-adapter", adapter_registry=None)
        assert False, "Should have raised MergeError"
    except MergeError:
        pass

    # ── 6. Why merge is required ────────────────────────────────────
    # After merge, the model is a standard HuggingFace model:
    # - GGUF export (ALN-301): conversion tools expect a full model
    # - vLLM serving: vLLM loads HF models directly
    # - Distribution: merged models are simpler to share
    #
    # The merge pipeline: LoRA adapter + base model -> merged model
    # Uses PEFT's merge_and_unload() internally.

    # Verify the adapter lifecycle tracking works end-to-end
    lifecycle_adapter = await registry.register_adapter(
        name="lifecycle-demo",
        adapter_path="/tmp/adapters/lifecycle",
        signature=sig,
    )
    assert lifecycle_adapter.merge_status == "separate"

    # Stage 1: merge
    await registry.update_merge_status(
        "lifecycle-demo", "1", "merged", "/tmp/merged/lifecycle"
    )
    merged_adapter = await registry.get_adapter("lifecycle-demo")
    assert merged_adapter.merge_status == "merged"

    # Stage 2: export to GGUF
    await registry.update_gguf_path("lifecycle-demo", "1", "/tmp/gguf/lifecycle.gguf")
    exported_adapter = await registry.get_adapter("lifecycle-demo")
    assert exported_adapter.merge_status == "exported"
    assert exported_adapter.gguf_path == "/tmp/gguf/lifecycle.gguf"

    # Stage 3: promote to production
    await registry.promote("lifecycle-demo", "1", "shadow")
    await registry.promote("lifecycle-demo", "1", "production")
    prod = await registry.get_adapter("lifecycle-demo", version="1")
    assert prod.stage == "production"
    assert prod.merge_status == "exported"


asyncio.run(main())
print("PASS: 07-align/07_merge")
