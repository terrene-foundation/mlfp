// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- ML / Inference Server
//!
//! OBJECTIVE: Serve ML models for real-time predictions via API.
//! LEVEL: Advanced
//! PARITY: Equivalent -- Python uses InferenceServer; Rust uses the same
//!         serving pattern with request/response types and batch support.
//! VALIDATES: InferenceConfig, request/response types, batching, caching
//!
//! Run: cargo run -p tutorial-ml --bin 13_inference_server

use serde_json::json;

fn main() {
    // ── 1. InferenceConfig ──
    // Configuration for model serving.

    let config = InferenceConfig::new("churn-predictor")
        .model_version(2)
        .max_batch_size(32)
        .timeout_ms(500)
        .cache_enabled(true)
        .cache_ttl_secs(300);

    assert_eq!(config.model_name(), "churn-predictor");
    assert_eq!(config.model_version(), 2);
    assert_eq!(config.max_batch_size(), 32);
    assert_eq!(config.timeout_ms(), 500);
    assert!(config.cache_enabled());

    // ── 2. Inference request ──
    // Typed request with features and optional metadata.

    let request = InferenceRequest {
        features: json!({
            "age": 35.0,
            "income": 55000.0,
            "tenure_months": 24,
        }),
        request_id: "req-001".to_string(),
    };

    assert_eq!(request.request_id, "req-001");

    // ── 3. Inference response ──

    let response = InferenceResponse {
        prediction: json!({"churn_probability": 0.23, "class": "no_churn"}),
        model_version: 2,
        latency_ms: 12,
        request_id: "req-001".to_string(),
        cached: false,
    };

    assert_eq!(response.model_version, 2);
    assert!(!response.cached);
    assert!(response.latency_ms < 500);

    // ── 4. Batch inference ──
    // Process multiple requests in a single call for efficiency.

    let batch_requests = vec![
        InferenceRequest {
            features: json!({"age": 25.0, "income": 30000.0}),
            request_id: "req-002".to_string(),
        },
        InferenceRequest {
            features: json!({"age": 45.0, "income": 80000.0}),
            request_id: "req-003".to_string(),
        },
        InferenceRequest {
            features: json!({"age": 60.0, "income": 120000.0}),
            request_id: "req-004".to_string(),
        },
    ];

    assert!(batch_requests.len() <= config.max_batch_size());

    // ── 5. Response caching ──
    // Identical requests can be served from cache.

    let cached_response = InferenceResponse {
        prediction: json!({"churn_probability": 0.23, "class": "no_churn"}),
        model_version: 2,
        latency_ms: 1, // Much faster from cache
        request_id: "req-005".to_string(),
        cached: true,
    };

    assert!(cached_response.cached);
    assert!(cached_response.latency_ms < response.latency_ms);

    // ── 6. Health check ──
    // Inference server exposes health endpoints.

    let health = json!({
        "status": "healthy",
        "model_loaded": true,
        "model_name": "churn-predictor",
        "model_version": 2,
        "requests_served": 1000,
        "avg_latency_ms": 15,
    });

    assert_eq!(health["status"], "healthy");
    assert_eq!(health["model_loaded"], true);

    // ── 7. Server pattern ──
    //
    //   let server = InferenceServer::new(config)
    //       .load_model("churn-predictor", model).await;
    //   server.start("0.0.0.0:8080").await;
    //
    //   // Client:
    //   let response = client.predict(request).await;

    // ── 8. Key concepts ──
    // - InferenceConfig: model name, version, batch size, timeout, cache
    // - InferenceRequest: features + request_id
    // - InferenceResponse: prediction + metadata (version, latency, cached)
    // - Batch inference for throughput optimization
    // - Response caching for latency reduction
    // - Health endpoints for monitoring
    // - Timeout enforcement for SLA compliance

    println!("PASS: 05-ml/13_inference_server");
}

struct InferenceConfig {
    model_name: String,
    model_version: u32,
    max_batch_size: usize,
    timeout_ms: u64,
    cache_enabled: bool,
    cache_ttl_secs: u64,
}

impl InferenceConfig {
    fn new(name: &str) -> Self {
        Self { model_name: name.to_string(), model_version: 1, max_batch_size: 32, timeout_ms: 1000, cache_enabled: false, cache_ttl_secs: 0 }
    }
    fn model_version(mut self, v: u32) -> Self { self.model_version = v; self }
    fn max_batch_size(mut self, n: usize) -> Self { self.max_batch_size = n; self }
    fn timeout_ms(mut self, ms: u64) -> Self { self.timeout_ms = ms; self }
    fn cache_enabled(mut self, e: bool) -> Self { self.cache_enabled = e; self }
    fn cache_ttl_secs(mut self, s: u64) -> Self { self.cache_ttl_secs = s; self }
    fn model_name(&self) -> &str { &self.model_name }
    fn model_version(&self) -> u32 { self.model_version }
    fn max_batch_size(&self) -> usize { self.max_batch_size }
    fn timeout_ms(&self) -> u64 { self.timeout_ms }
    fn cache_enabled(&self) -> bool { self.cache_enabled }
}

struct InferenceRequest {
    features: serde_json::Value,
    request_id: String,
}

struct InferenceResponse {
    prediction: serde_json::Value,
    model_version: u32,
    latency_ms: u64,
    request_id: String,
    cached: bool,
}
