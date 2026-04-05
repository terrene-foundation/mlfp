// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Align / LoRA Adapters
//!
//! OBJECTIVE: Understand LoRA (Low-Rank Adaptation) — the W = W0 + BA decomposition,
//!            rank selection, and parameter efficiency.
//! LEVEL: Intermediate
//! PARITY: Partial -- kailash-align-serving provides AdapterMetadata with rank/alpha;
//!         full LoRA training is pending kailash-align.
//! VALIDATES: LoRA math, rank/alpha tradeoffs, parameter counting, AdapterMetadata fields
//!
//! Run: cargo run -p tutorial-align --bin 02_lora

use kailash_align_serving::adapter::{AdapterMetadata, TrainingMethod};

fn main() {
    // ── 1. LoRA core idea ──
    // Instead of updating all parameters W, LoRA freezes W₀ and learns
    // a low-rank update: W = W₀ + BA
    //
    //   W₀: [d × d] original weight matrix (frozen)
    //   B:  [d × r] down-projection (learned)
    //   A:  [r × d] up-projection (learned)
    //   r:  rank (much smaller than d)
    //
    // For a 4096×4096 weight matrix with rank 16:
    //   Original params: 4096 × 4096 = 16,777,216
    //   LoRA params:     4096 × 16 + 16 × 4096 = 131,072
    //   Ratio:           0.78% of original parameters

    let d = 4096_u64;
    let r = 16_u64;
    let original_params = d * d;
    let lora_params = d * r + r * d;
    let ratio = lora_params as f64 / original_params as f64;

    assert_eq!(original_params, 16_777_216);
    assert_eq!(lora_params, 131_072);
    assert!(ratio < 0.01); // Less than 1% of parameters

    // ── 2. Rank selection ──
    // Rank controls the capacity of the adapter:
    //   r=4:   Minimal, good for narrow tasks
    //   r=8:   Light adaptation
    //   r=16:  Default, good balance (most common)
    //   r=32:  Richer adaptation
    //   r=64:  High capacity, approaching full fine-tune quality
    //   r=128: Very high, diminishing returns vs full fine-tune

    let ranks = vec![
        RankConfig { rank: 4,   params: 2 * d * 4,   use_case: "narrow domain" },
        RankConfig { rank: 8,   params: 2 * d * 8,   use_case: "light adaptation" },
        RankConfig { rank: 16,  params: 2 * d * 16,  use_case: "general purpose" },
        RankConfig { rank: 32,  params: 2 * d * 32,  use_case: "complex tasks" },
        RankConfig { rank: 64,  params: 2 * d * 64,  use_case: "high capacity" },
        RankConfig { rank: 128, params: 2 * d * 128, use_case: "maximum quality" },
    ];

    // Each doubling of rank doubles the parameter count
    assert_eq!(ranks[2].params, 2 * ranks[1].params);
    assert_eq!(ranks[3].params, 2 * ranks[2].params);

    // ── 3. Alpha scaling ──
    // Alpha controls how much the adapter contributes:
    //   effective_update = (alpha / rank) * BA
    //
    // Convention: alpha = 2 * rank (so scaling factor = 2.0)
    // Higher alpha = stronger adapter influence.

    let rank = 16_u32;
    let alpha = 32.0_f32;
    let scaling_factor = alpha / rank as f32;
    assert!((scaling_factor - 2.0).abs() < f32::EPSILON);

    // With rank=64, alpha=128 keeps the same scaling
    let rank2 = 64_u32;
    let alpha2 = 128.0_f32;
    let scaling2 = alpha2 / rank2 as f32;
    assert!((scaling2 - 2.0).abs() < f32::EPSILON);

    // ── 4. Target modules ──
    // LoRA can be applied to specific attention matrices:
    //   q_proj - query projection (most common target)
    //   k_proj - key projection
    //   v_proj - value projection (most common target)
    //   o_proj - output projection
    //   gate_proj, up_proj, down_proj - MLP layers
    //
    // More targets = more parameters but better quality.

    let minimal_targets = vec!["q_proj".to_string(), "v_proj".to_string()];
    let full_targets = vec![
        "q_proj".to_string(), "k_proj".to_string(),
        "v_proj".to_string(), "o_proj".to_string(),
    ];

    // Parameter count scales with number of target modules
    let layers = 32_u64; // typical for 7B model
    let minimal_total = minimal_targets.len() as u64 * layers * lora_params;
    let full_total = full_targets.len() as u64 * layers * lora_params;
    assert_eq!(full_total, 2 * minimal_total);

    // ── 5. AdapterMetadata captures LoRA config ──

    let adapter = AdapterMetadata {
        name: "code-assistant-v2".into(),
        description: "LoRA adapter for code generation".into(),
        method: TrainingMethod::Lora,
        rank: 16,
        alpha: 32.0,
        target_modules: minimal_targets.clone(),
        base_model: "llama-3-8b".into(),
        version: "2.0".into(),
        ..Default::default()
    };

    assert_eq!(adapter.rank, 16);
    assert!((adapter.alpha - 32.0).abs() < f32::EPSILON);
    assert_eq!(adapter.target_modules, minimal_targets);

    // ── 6. QLoRA variant ──
    // QLoRA quantizes the base model to 4-bit, then applies LoRA.
    // Same adapter quality, ~75% less GPU memory for the base model.

    let qlora_adapter = AdapterMetadata {
        name: "finance-qlora".into(),
        method: TrainingMethod::Qlora,
        rank: 16,
        alpha: 32.0,
        target_modules: vec!["q_proj".into(), "v_proj".into()],
        base_model: "llama-3-8b".into(),
        ..Default::default()
    };

    assert_eq!(qlora_adapter.method, TrainingMethod::Qlora);
    // LoRA adapter size is the same regardless of base quantization
    assert_eq!(qlora_adapter.rank, adapter.rank);

    // ── 7. Parameter efficiency comparison ──
    // For a 7B parameter model with 32 layers, 4096 hidden dim:

    let model_params = 7_000_000_000_u64;
    // Per layer per target: d*r + r*d = 2*d*r = 2*4096*16 = 131,072
    // Total: 2 targets * 32 layers * 131,072 = 8,388,608 (~0.12%)
    let lora_total = 2 * (2 * 4096 * 16) * 32_u64; // 2 targets, 32 layers
    let efficiency = lora_total as f64 / model_params as f64;
    assert!(efficiency < 0.002); // Less than 0.2% of model params

    // ── 8. Key concepts ──
    // - LoRA: W = W₀ + BA, freeze W₀, train only B and A
    // - Rank (r): controls adapter capacity, 16 is default
    // - Alpha: scaling factor, convention is alpha = 2 * rank
    // - Target modules: which layers get LoRA (q_proj, v_proj most common)
    // - QLoRA: 4-bit base + LoRA, same quality with less memory
    // - Parameter efficiency: <1% of model params for strong adaptation
    // - More targets × higher rank = more params but better quality

    println!("PASS: 07-align/02_lora");
}

struct RankConfig {
    rank: u64,
    params: u64,
    use_case: &'static str,
}
