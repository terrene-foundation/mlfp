# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Align / Serving Configuration
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Serve aligned models with GGUF export, Ollama, and vLLM
# LEVEL: Advanced
# PARITY: Partial (Rust has kailash-align-serving with GGUF)
# VALIDATES: AlignmentServing, ServingConfig, VLLMBackend, VLLMConfig,
#            GenerationBackend, HFGenerationBackend
#
# Run: uv run python textbook/python/07-align/09_serving.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os

from kailash_align import (
    AdapterRegistry,
    AlignmentServing,
    GenerationBackend,
    HFGenerationBackend,
    ServingConfig,
    VLLMBackend,
    VLLMConfig,
)
from kailash_align.exceptions import ServingError
from kailash_align.serving import QUANTIZATION_TYPES, SUPPORTED_ARCHITECTURES

model_id = os.environ.get("DEFAULT_LLM_MODEL", "meta-llama/Llama-3.1-8B")

# ── 1. ServingConfig defaults ───────────────────────────────────────

default_serving = ServingConfig()

assert default_serving.target == "ollama"
assert default_serving.quantization == "q4_k_m"
assert default_serving.system_prompt is None
assert default_serving.ollama_host == "http://localhost:11434"
assert default_serving.validate_gguf is True
assert default_serving.validation_timeout == 120

# ── 2. ServingConfig is frozen ──────────────────────────────────────

try:
    default_serving.target = "vllm"  # type: ignore[misc]
    assert False, "Should have raised FrozenInstanceError"
except AttributeError:
    pass  # Expected: frozen dataclass

# ── 3. Target validation — ollama or vllm only ─────────────────────

for valid_target in ("ollama", "vllm"):
    cfg = ServingConfig(target=valid_target)
    assert cfg.target == valid_target

try:
    ServingConfig(target="invalid")
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "target" in str(e)

# ── 4. Quantization validation ──────────────────────────────────────

for valid_quant in ("f16", "q4_k_m", "q8_0"):
    cfg = ServingConfig(quantization=valid_quant)
    assert cfg.quantization == valid_quant

try:
    ServingConfig(quantization="q2_k")
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "quantization" in str(e)

# ── 5. validation_timeout must be positive ──────────────────────────

try:
    ServingConfig(validation_timeout=0)
    assert False, "Should have raised ValueError"
except ValueError:
    pass

try:
    ServingConfig(validation_timeout=-1)
    assert False, "Should have raised ValueError"
except ValueError:
    pass

# ── 6. Custom ServingConfig ─────────────────────────────────────────

vllm_serving = ServingConfig(
    target="vllm",
    quantization="f16",
    system_prompt="You are a helpful assistant.",
    validate_gguf=False,
)

assert vllm_serving.target == "vllm"
assert vllm_serving.quantization == "f16"
assert vllm_serving.system_prompt == "You are a helpful assistant."
assert vllm_serving.validate_gguf is False

ollama_serving = ServingConfig(
    target="ollama",
    quantization="q8_0",
    ollama_host="http://my-server:11434",
    validation_timeout=300,
)

assert ollama_serving.ollama_host == "http://my-server:11434"
assert ollama_serving.validation_timeout == 300

# ── 7. Supported architectures for GGUF ────────────────────────────

assert "LlamaForCausalLM" in SUPPORTED_ARCHITECTURES
assert "MistralForCausalLM" in SUPPORTED_ARCHITECTURES
assert "Phi3ForCausalLM" in SUPPORTED_ARCHITECTURES
assert "Qwen2ForCausalLM" in SUPPORTED_ARCHITECTURES

assert SUPPORTED_ARCHITECTURES["LlamaForCausalLM"] == "fully_supported"
assert SUPPORTED_ARCHITECTURES["MistralForCausalLM"] == "fully_supported"

# ── 8. Quantization types ──────────────────────────────────────────

assert "f16" in QUANTIZATION_TYPES
assert "q4_k_m" in QUANTIZATION_TYPES
assert "q8_0" in QUANTIZATION_TYPES
assert QUANTIZATION_TYPES["f16"] is None, "f16 means no quantization step"

# ── 9. AlignmentServing creation ────────────────────────────────────

serving = AlignmentServing()
assert serving._registry is None
assert isinstance(serving._config, ServingConfig)

registry = AdapterRegistry()
serving_with_registry = AlignmentServing(
    adapter_registry=registry,
    config=ServingConfig(target="ollama", quantization="q4_k_m"),
)
assert serving_with_registry._registry is registry
assert serving_with_registry._config.target == "ollama"

# ── 10. AlignmentServing requires registry for operations ───────────
# Serving operations need AdapterRegistry to look up adapter metadata.

import asyncio


async def test_serving_requires_registry() -> None:
    # Use vllm target to avoid optional [serve] dependency check
    # (ollama path hits _check_serve_deps before the registry check)
    no_registry = AlignmentServing(config=ServingConfig(target="vllm"))

    try:
        await no_registry.deploy("any-adapter")
        assert False, "Should have raised ServingError"
    except ServingError as e:
        assert "AdapterRegistry" in str(e)


asyncio.run(test_serving_requires_registry())

# ── 11. VLLMConfig defaults ────────────────────────────────────────

vllm_config = VLLMConfig()

assert vllm_config.tensor_parallel_size == 1
assert vllm_config.gpu_memory_utilization == 0.9
assert vllm_config.max_model_len is None
assert vllm_config.dtype == "auto"
assert vllm_config.seed == 42

# ── 12. VLLMConfig is frozen ───────────────────────────────────────

try:
    vllm_config.seed = 123  # type: ignore[misc]
    assert False, "Should have raised FrozenInstanceError"
except AttributeError:
    pass

# ── 13. VLLMConfig validation ──────────────────────────────────────

# gpu_memory_utilization must be in (0, 1]
try:
    VLLMConfig(gpu_memory_utilization=0.0)
    assert False, "Should have raised ValueError"
except ValueError:
    pass

try:
    VLLMConfig(gpu_memory_utilization=1.5)
    assert False, "Should have raised ValueError"
except ValueError:
    pass

full_util = VLLMConfig(gpu_memory_utilization=1.0)
assert full_util.gpu_memory_utilization == 1.0, "1.0 is valid (use all GPU memory)"

# tensor_parallel_size must be >= 1
try:
    VLLMConfig(tensor_parallel_size=0)
    assert False, "Should have raised ValueError"
except ValueError:
    pass

multi_gpu = VLLMConfig(tensor_parallel_size=4)
assert multi_gpu.tensor_parallel_size == 4

# ── 14. Custom VLLMConfig ──────────────────────────────────────────

custom_vllm = VLLMConfig(
    tensor_parallel_size=2,
    gpu_memory_utilization=0.85,
    max_model_len=8192,
    dtype="bfloat16",
    seed=0,
)

assert custom_vllm.tensor_parallel_size == 2
assert custom_vllm.gpu_memory_utilization == 0.85
assert custom_vllm.max_model_len == 8192
assert custom_vllm.dtype == "bfloat16"
assert custom_vllm.seed == 0

# ── 15. GenerationBackend ABC ───────────────────────────────────────
# GenerationBackend is the abstract base for generation backends.
# VLLMBackend and HFGenerationBackend both implement it.

assert issubclass(VLLMBackend, GenerationBackend)
assert issubclass(HFGenerationBackend, GenerationBackend)

# VLLMBackend can be instantiated (lazy-loads model)
vllm_backend = VLLMBackend(model_id=model_id)
assert vllm_backend._model_id == model_id
assert vllm_backend._llm is None, "Model not loaded yet (lazy)"

# HFGenerationBackend can be instantiated (lazy-loads model)
hf_backend = HFGenerationBackend(model_id=model_id)
assert hf_backend._model_id == model_id
assert hf_backend._model is None, "Model not loaded yet (lazy)"

# VLLMBackend with custom config
custom_backend = VLLMBackend(
    model_id=model_id,
    config=VLLMConfig(tensor_parallel_size=2, gpu_memory_utilization=0.8),
)
assert custom_backend._config.tensor_parallel_size == 2
assert custom_backend._config.gpu_memory_utilization == 0.8

# HFGenerationBackend with device override
hf_cpu = HFGenerationBackend(model_id=model_id, device="cpu")
assert hf_cpu._device == "cpu"

# ── 16. shutdown() is safe to call on unloaded backends ─────────────

vllm_backend.shutdown()  # No-op: model not loaded
assert vllm_backend._llm is None

hf_backend.shutdown()  # No-op: model not loaded
assert hf_backend._model is None

print("PASS: 07-align/09_serving")
