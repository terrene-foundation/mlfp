// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Align / Alignment Configuration
//!
//! OBJECTIVE: Understand alignment configuration, training methods, and adapter metadata.
//! LEVEL: Basic
//! PARITY: Partial -- kailash-align-serving provides AdapterMetadata and TrainingMethod;
//!         full AlignmentConfig (Python's method selection + hyperparameters) is pending.
//! VALIDATES: TrainingMethod, AdapterMetadata, configuration patterns
//!
//! Run: cargo run -p tutorial-align --bin 01_alignment_config

use kailash_align_serving::adapter::{AdapterMetadata, TrainingMethod};

fn main() {
    // ── 1. Training methods ──
    // kailash-align supports multiple fine-tuning methods. Each trades off
    // parameter efficiency against expressiveness.

    let lora = TrainingMethod::Lora;
    let qlora = TrainingMethod::Qlora;
    let full = TrainingMethod::FullFineTune;

    assert_eq!(lora.to_string(), "LoRA");
    assert_eq!(qlora.to_string(), "QLoRA");
    assert_eq!(full.to_string(), "Full Fine-Tune");

    // ── 2. Default training method ──
    // LoRA is the default -- best balance of quality and efficiency.

    let default_method = TrainingMethod::default();
    assert_eq!(default_method, TrainingMethod::Lora);

    // ── 3. All supported methods ──
    // The TrainingMethod enum covers the major PEFT approaches:
    //   LoRA       - Low-Rank Adaptation (default, most popular)
    //   QLoRA      - Quantized LoRA (4-bit base model, same quality)
    //   PromptTuning  - Virtual tokens prepended to input
    //   PrefixTuning  - Virtual tokens at every layer
    //   FullFineTune  - All parameters updated (highest quality, most memory)
    //   IA3        - Inhibiting and Amplifying Inner Activations
    //   AdaLoRA    - Adaptive Budget Allocation LoRA

    let methods = vec![
        TrainingMethod::Lora,
        TrainingMethod::Qlora,
        TrainingMethod::PromptTuning,
        TrainingMethod::PrefixTuning,
        TrainingMethod::FullFineTune,
        TrainingMethod::Ia3,
        TrainingMethod::AdaLora,
        TrainingMethod::Other,
    ];
    assert_eq!(methods.len(), 8);

    // ── 4. AdapterMetadata ──
    // Metadata describes an adapter's provenance: how it was trained,
    // which base model, what rank, target modules, etc.

    let meta = AdapterMetadata {
        name: "finance-analyst-v1".into(),
        description: "Fine-tuned for financial report analysis".into(),
        method: TrainingMethod::Lora,
        rank: 16,
        alpha: 32.0,
        target_modules: vec!["q_proj".into(), "v_proj".into()],
        base_model: "llama-3-8b".into(),
        version: "1.0".into(),
        tags: vec!["finance".into(), "production".into()],
        ..Default::default()
    };

    assert_eq!(meta.name, "finance-analyst-v1");
    assert_eq!(meta.method, TrainingMethod::Lora);
    assert_eq!(meta.rank, 16);
    assert!((meta.alpha - 32.0).abs() < f32::EPSILON);
    assert_eq!(meta.target_modules.len(), 2);

    // ── 5. Default metadata values ──
    // Sensible defaults: LoRA with rank=16, alpha=32.0.

    let defaults = AdapterMetadata::default();
    assert_eq!(defaults.method, TrainingMethod::Lora);
    assert_eq!(defaults.rank, 16);
    assert!((defaults.alpha - 32.0).abs() < f32::EPSILON);

    // ── 6. Metadata display ──
    // Display format shows name, method, rank, alpha, and base model.

    let display = meta.to_string();
    assert!(display.contains("finance-analyst-v1"));
    assert!(display.contains("LoRA"));
    assert!(display.contains("rank=16"));
    assert!(display.contains("llama-3-8b"));

    // ── 7. Serialization ──
    // AdapterMetadata serializes to JSON for storage and transport.

    let json = serde_json::to_string(&meta).expect("serialize");
    let restored: AdapterMetadata = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(meta, restored);

    // TrainingMethod uses snake_case in JSON
    let method_json = serde_json::to_string(&TrainingMethod::FullFineTune).expect("serialize");
    assert_eq!(method_json, "\"full_fine_tune\"");

    // ── 8. Configuration patterns ──
    // In Python, AlignmentConfig combines method + hyperparameters:
    //
    //   config = AlignmentConfig(
    //       method="lora",
    //       rank=16,
    //       alpha=32.0,
    //       target_modules=["q_proj", "v_proj"],
    //       learning_rate=2e-4,
    //       num_epochs=3,
    //   )
    //
    // In Rust, AdapterMetadata captures the method/rank/alpha/targets.
    // Training hyperparameters (lr, epochs, batch_size) will be added
    // when kailash-align training lands.

    let config = AlignmentConfig {
        method: TrainingMethod::Lora,
        rank: 16,
        alpha: 32.0,
        target_modules: vec!["q_proj".into(), "v_proj".into()],
        learning_rate: 2e-4,
        num_epochs: 3,
        batch_size: 4,
        warmup_ratio: 0.03,
        max_seq_length: 2048,
    };

    assert_eq!(config.method, TrainingMethod::Lora);
    assert_eq!(config.rank, 16);
    assert!((config.learning_rate - 2e-4).abs() < 1e-10);

    // ── 9. Key concepts ──
    // - TrainingMethod: enum of fine-tuning approaches (LoRA default)
    // - AdapterMetadata: provenance record (method, rank, alpha, targets)
    // - AlignmentConfig: full training configuration (pending in Rust)
    // - LoRA is the default method -- efficient, minimal memory overhead
    // - rank controls adapter capacity (higher = more parameters)
    // - alpha scales the adapter contribution (typically 2x rank)
    // - target_modules specifies which layers get adapted

    println!("PASS: 07-align/01_alignment_config");
}

/// Training configuration combining method selection with hyperparameters.
/// Pending full implementation in kailash-align.
struct AlignmentConfig {
    method: TrainingMethod,
    rank: u32,
    alpha: f32,
    target_modules: Vec<String>,
    learning_rate: f64,
    num_epochs: u32,
    batch_size: u32,
    warmup_ratio: f64,
    max_seq_length: u32,
}
