// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Kaizen / LLM Providers
//!
//! OBJECTIVE: Configure LLM providers via environment variables and model detection.
//! LEVEL: Intermediate
//! PARITY: Full -- Both SDKs detect providers from model name prefixes
//!         and read API keys from environment variables.
//! VALIDATES: Provider detection, cost estimation, API key patterns
//!
//! Run: cargo run -p tutorial-kaizen --bin 04_llm_providers

use kailash_kaizen::llm::{detect_provider, Provider};
use kailash_kaizen::cost::estimate_cost;

fn main() {
    // ── 1. Model configuration via environment ──
    // Kaizen NEVER hardcodes model names. They come from env vars:
    //   DEFAULT_LLM_MODEL    -- primary model for all agents
    //   OPENAI_PROD_MODEL    -- OpenAI-specific override
    //   ANTHROPIC_API_KEY    -- for Claude models
    //   OPENAI_API_KEY       -- for GPT models

    let model = std::env::var("DEFAULT_LLM_MODEL")
        .unwrap_or_else(|_| "claude-sonnet-4-20250514".to_string());
    assert!(!model.is_empty(), "Model name must be non-empty");

    // ── 2. Provider detection from model name ──
    // The SDK detects the provider from model name prefixes:
    //   claude-*         -> Anthropic
    //   gpt-*, o1-*, o3-*, o4-* -> OpenAI
    //   gemini-*         -> Google

    assert_eq!(detect_provider("claude-sonnet-4-20250514"), Provider::Anthropic);
    assert_eq!(detect_provider("gpt-4o"), Provider::OpenAI);
    assert_eq!(detect_provider("o3-mini"), Provider::OpenAI);
    assert_eq!(detect_provider("gemini-pro"), Provider::Google);
    assert_eq!(detect_provider("unknown-model"), Provider::Unknown);

    // Provider enum is exhaustive
    assert_eq!(Provider::Anthropic.as_str(), "anthropic");
    assert_eq!(Provider::OpenAI.as_str(), "openai");
    assert_eq!(Provider::Google.as_str(), "google");

    // ── 3. Cost estimation by provider ──
    // Delegate tracks cost estimates per 1M tokens:
    //   claude-  : $3 input, $15 output
    //   gpt-4o   : $2.5 input, $10 output
    //   gpt-4    : $30 input, $60 output
    //   gemini-  : $1.25 input, $5 output

    let claude_cost = estimate_cost("claude-sonnet-4-20250514", 1000, 500);
    // input: 1000/1M * $3 = $0.003, output: 500/1M * $15 = $0.0075
    assert!(claude_cost > 0.01 && claude_cost < 0.02);

    let gpt4o_cost = estimate_cost("gpt-4o", 1000, 500);
    // input: 1000/1M * $2.5 = $0.0025, output: 500/1M * $10 = $0.005
    assert!(gpt4o_cost > 0.007 && gpt4o_cost < 0.008);

    let gemini_cost = estimate_cost("gemini-pro", 1000, 500);
    // input: 1000/1M * $1.25 = $0.00125, output: 500/1M * $5 = $0.0025
    assert!(gemini_cost > 0.003 && gemini_cost < 0.004);

    // ── 4. API key patterns ──
    // Keys are always from environment, never hardcoded:
    //   ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY
    //
    // The SDK reads these at runtime:
    //   let key = std::env::var("ANTHROPIC_API_KEY")
    //       .expect("ANTHROPIC_API_KEY must be set");
    //
    // .env files loaded by dotenv crate or shell configuration.

    // ── 5. Provider-specific key lookup ──
    // Each provider has a standard environment variable:

    let key_vars = [
        (Provider::Anthropic, "ANTHROPIC_API_KEY"),
        (Provider::OpenAI, "OPENAI_API_KEY"),
        (Provider::Google, "GOOGLE_API_KEY"),
    ];

    for (provider, var_name) in &key_vars {
        assert!(
            !var_name.is_empty(),
            "{:?} must have a key variable",
            provider
        );
    }

    // ── 6. Key concepts ──
    // - Model names always from env vars (DEFAULT_LLM_MODEL, etc.)
    // - detect_provider() maps model prefix to Provider enum
    // - estimate_cost() calculates approximate cost per turn
    // - API keys from ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY
    // - .env files are the single source of truth for secrets
    // - NEVER hardcode model names or API keys in source code

    println!("PASS: 03-kaizen/04_llm_providers");
}
