// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Align / Model Serving
//!
//! OBJECTIVE: Serve fine-tuned models — GGUF loading, inference requests, sampling
//!            parameters, streaming, and adapter hot-swapping.
//! LEVEL: Intermediate
//! PARITY: Full -- kailash-align-serving provides InferenceEngine, ServingBackend,
//!         SamplingParams, InferenceRequest/Response, streaming, and hot-swap.
//! VALIDATES: SamplingParams, InferenceRequest, InferenceResponse, FinishReason,
//!            ModelParams, ModelInfo, EngineConfig, InferenceTiming, StreamToken
//!
//! Run: cargo run -p tutorial-align --bin 09_serving

use std::time::Duration;

use kailash_align_serving::{
    EngineConfig, FinishReason, InferenceTiming,
    InferenceRequest, InferenceResponse, SamplingParams, StreamToken,
};
use kailash_align_serving::model::{AdapterId, ModelInfo, ModelParams};

fn main() {
    // ── 1. SamplingParams ──
    // Controls how the model generates text.

    let params = SamplingParams::default();
    assert!((params.temperature - 0.7).abs() < f32::EPSILON);
    assert!((params.top_p - 0.9).abs() < f32::EPSILON);
    assert_eq!(params.top_k, 40);
    assert_eq!(params.max_tokens, 256);

    // Custom sampling for creative writing (higher temperature)
    let creative = SamplingParams {
        temperature: 1.2,
        top_p: 0.95,
        top_k: 50,
        max_tokens: 1024,
        repetition_penalty: 1.2,
        ..Default::default()
    };

    assert!(creative.temperature > params.temperature);
    assert!(creative.validate().is_ok());

    // Deterministic output (low temperature, no randomness)
    let deterministic = SamplingParams {
        temperature: 0.01,
        top_p: 1.0,
        top_k: 1, // Always pick the top token
        max_tokens: 512,
        seed: 42,
        ..Default::default()
    };

    assert!(deterministic.validate().is_ok());

    // ── 2. Parameter validation ──
    // Invalid parameters are caught at validation time.

    let invalid = SamplingParams {
        temperature: 0.0, // Must be > 0
        ..Default::default()
    };
    assert!(invalid.validate().is_err());

    let invalid_top_p = SamplingParams {
        top_p: 1.5, // Must be in [0.0, 1.0]
        ..Default::default()
    };
    assert!(invalid_top_p.validate().is_err());

    let invalid_tokens = SamplingParams {
        max_tokens: 0, // Must be > 0
        ..Default::default()
    };
    assert!(invalid_tokens.validate().is_err());

    // ── 3. InferenceRequest ──
    // A request combines a prompt with sampling parameters and optional adapters.

    let request = InferenceRequest::new("Explain the concept of LoRA fine-tuning.")
        .with_sampling(SamplingParams {
            temperature: 0.7,
            max_tokens: 256,
            ..Default::default()
        });

    assert_eq!(request.prompt, "Explain the concept of LoRA fine-tuning.");
    assert!(request.adapter_ids.is_empty());

    // Request with adapter
    let adapter_id = AdapterId::new();
    let request_with_adapter = InferenceRequest::new("Analyze the earnings report.")
        .with_adapter(adapter_id.clone())
        .with_sampling(SamplingParams::default());

    assert_eq!(request_with_adapter.adapter_ids.len(), 1);
    assert_eq!(request_with_adapter.adapter_ids[0], adapter_id);

    // Display format shows prompt preview and adapter count
    let display = format!("{}", request);
    assert!(display.contains("InferenceRequest"));
    assert!(display.contains("adapters=0"));

    // ── 4. ModelParams ──
    // Controls how the model is loaded into memory.

    let params = ModelParams::default();
    assert!(params.use_mmap, "Memory mapping is enabled by default");
    assert!(!params.use_mlock, "Memory locking is disabled by default");
    assert_eq!(params.batch_size, 512);
    assert_eq!(params.gpu_layers, 0); // CPU only by default

    // GPU-accelerated configuration
    let gpu_params = ModelParams {
        context_length: 8192,
        gpu_layers: 40, // Offload 40 layers to GPU
        threads: 8,
        batch_size: 1024,
        use_mmap: true,
        use_mlock: true,
        seed: 42,
    };

    assert_eq!(gpu_params.gpu_layers, 40);
    let display = format!("{}", gpu_params);
    assert!(display.contains("gpu_layers=40"));

    // ── 5. FinishReason ──
    // Why text generation stopped.

    assert_eq!(FinishReason::MaxTokens.to_string(), "max_tokens");
    assert_eq!(FinishReason::StopSequence.to_string(), "stop_sequence");
    assert_eq!(FinishReason::EndOfSequence.to_string(), "end_of_sequence");
    assert_eq!(FinishReason::Cancelled.to_string(), "cancelled");

    // ── 6. InferenceResponse ──

    let response = InferenceResponse {
        request_id: request.id,
        text: "LoRA (Low-Rank Adaptation) is a parameter-efficient \
               fine-tuning method that freezes the original model weights \
               and trains small adapter matrices.".into(),
        prompt_tokens: 12,
        completion_tokens: 28,
        finish_reason: FinishReason::EndOfSequence,
        timing: InferenceTiming {
            prompt_eval: Duration::from_millis(45),
            generation: Duration::from_millis(350),
            total: Duration::from_millis(395),
            tokens_per_second: 80.0,
        },
        ..Default::default()
    };

    assert!(!response.text.is_empty());
    assert_eq!(response.prompt_tokens, 12);
    assert_eq!(response.completion_tokens, 28);
    assert_eq!(response.finish_reason, FinishReason::EndOfSequence);
    assert!(response.timing.tokens_per_second > 0.0);

    let display = format!("{}", response);
    assert!(display.contains("12/28"));
    assert!(display.contains("end_of_sequence"));

    // ── 7. StreamToken ──
    // During streaming, tokens arrive one at a time.

    let token = StreamToken {
        index: 0,
        text: "LoRA".into(),
        token_id: 1234,
        logprob: Some(-0.5),
        is_final: false,
        finish_reason: None,
    };

    assert!(!token.is_final);
    assert_eq!(token.text, "LoRA");
    assert!(token.logprob.is_some());

    let final_token = StreamToken {
        index: 27,
        text: String::new(),
        token_id: 0,
        logprob: None,
        is_final: true,
        finish_reason: Some(FinishReason::EndOfSequence),
    };

    assert!(final_token.is_final);
    assert_eq!(final_token.finish_reason, Some(FinishReason::EndOfSequence));

    // ── 8. EngineConfig ──
    // Configures the InferenceEngine that composes backend + adapter management.

    let config = EngineConfig::new()
        .with_drain_timeout(Duration::from_secs(60))
        .with_max_concurrent_requests(16);

    assert_eq!(config.drain_timeout, Duration::from_secs(60));
    assert_eq!(config.max_concurrent_requests, 16);

    let default_config = EngineConfig::default();
    assert_eq!(default_config.drain_timeout, Duration::from_secs(30));
    assert_eq!(default_config.max_concurrent_requests, 8);

    // ── 9. ModelInfo ──
    // Metadata about a loaded model.

    let model_info = ModelInfo {
        name: "llama-3-8b-instruct".into(),
        format: "gguf".into(),
        parameter_count: 8_000_000_000,
        context_length: 8192,
        ..Default::default()
    };

    assert_eq!(model_info.format, "gguf");
    let display = format!("{}", model_info);
    assert!(display.contains("llama-3-8b-instruct"));
    assert!(display.contains("gguf"));

    // ── 10. Serving architecture ──
    // The serving stack:
    //
    //   InferenceEngine
    //     ├── ServingBackend (trait: load_model, generate, generate_stream)
    //     │     ├── LlamaCppBackend (GGUF via llama.cpp)
    //     │     └── CandleBackend  (future: Hugging Face Candle)
    //     └── DefaultAdapterManager (concurrent adapter registry)
    //
    // Hot-swap workflow:
    //   1. Set draining flag (new requests to old adapter rejected)
    //   2. Wait for in-flight requests to complete (exponential backoff)
    //   3. Acquire write lock on backend
    //   4. Remove old adapter, load new adapter
    //   5. Clear draining flag (new requests accepted)
    //
    // Zero downtime: read lock for inference (parallel),
    // write lock only during adapter swap (brief exclusive).

    // ── 11. Key concepts ──
    // - SamplingParams: temperature, top_p, top_k, max_tokens, stop_sequences
    // - Validation catches invalid params before inference starts
    // - InferenceRequest: prompt + sampling + optional adapter IDs
    // - InferenceResponse: generated text + token counts + timing breakdown
    // - StreamToken: incremental token delivery during streaming
    // - FinishReason: why generation stopped (max_tokens, stop, EOS, cancelled)
    // - ModelParams: GPU layers, context length, threads, memory mapping
    // - EngineConfig: drain timeout and concurrency limits
    // - Hot-swap: adapter replacement with zero-downtime drain protocol

    println!("PASS: 07-align/09_serving");
}
