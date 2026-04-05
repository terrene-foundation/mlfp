// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Align / Model Merging
//!
//! OBJECTIVE: Understand model merging strategies — TIES, DARE, and linear interpolation
//!            for combining multiple fine-tuned models without additional training.
//! LEVEL: Advanced
//! PARITY: Pending kailash-rs implementation — demonstrates merging math with Rust idioms.
//!         Python kailash-align supports TIES, DARE, and linear merge methods.
//! VALIDATES: Linear interpolation, TIES-Merging, DARE, weight arithmetic, task vectors
//!
//! Run: cargo run -p tutorial-align --bin 07_merge

fn main() {
    // ── 1. Linear interpolation ──
    // The simplest merge: weighted average of model parameters.
    //   W_merged = alpha * W_A + (1 - alpha) * W_B
    //
    // alpha = 0.5 gives equal weight to both models.

    let weights_a = vec![1.0_f64, 2.0, 3.0, 4.0];
    let weights_b = vec![2.0, 0.0, 4.0, 2.0];
    let alpha = 0.5;

    let merged = linear_interpolate(&weights_a, &weights_b, alpha);
    assert_eq!(merged, vec![1.5, 1.0, 3.5, 3.0]);

    // Asymmetric blend: favor model A
    let biased = linear_interpolate(&weights_a, &weights_b, 0.8);
    assert!((biased[0] - 1.2).abs() < 1e-10); // 0.8*1.0 + 0.2*2.0 = 1.2

    // ── 2. Task vectors ──
    // A task vector is the difference between a fine-tuned model and its base:
    //   tau_A = W_A - W_base
    //
    // Task vectors represent what a model "learned" during fine-tuning.

    let base_weights = vec![0.5, 1.0, 1.5, 2.0];
    let task_vector_a: Vec<f64> = weights_a.iter().zip(&base_weights)
        .map(|(a, b)| a - b).collect();
    let task_vector_b: Vec<f64> = weights_b.iter().zip(&base_weights)
        .map(|(a, b)| a - b).collect();

    assert_eq!(task_vector_a, vec![0.5, 1.0, 1.5, 2.0]);
    assert_eq!(task_vector_b, vec![1.5, -1.0, 2.5, 0.0]);

    // Merge via task vectors: W_merged = W_base + lambda_A * tau_A + lambda_B * tau_B
    let lambda_a = 0.7;
    let lambda_b = 0.3;
    let merged_tv: Vec<f64> = base_weights.iter().enumerate()
        .map(|(i, &w)| w + lambda_a * task_vector_a[i] + lambda_b * task_vector_b[i])
        .collect();

    // Verify: base + weighted task vectors
    assert!((merged_tv[0] - (0.5 + 0.7 * 0.5 + 0.3 * 1.5)).abs() < 1e-10);

    // ── 3. TIES-Merging ──
    // Trim, Elect Sign, and Merge:
    //   1. Trim: zero out small-magnitude values (below threshold)
    //   2. Elect sign: for each parameter, majority-vote on the sign
    //   3. Merge: average only the values with the elected sign
    //
    // This resolves sign conflicts between task vectors.

    // Step 1: Trim (zero out small values)
    let threshold = 0.5;
    let trimmed_a = trim(&task_vector_a, threshold);
    let trimmed_b = trim(&task_vector_b, threshold);

    // Values below threshold become zero
    assert_eq!(trimmed_a, vec![0.5, 1.0, 1.5, 2.0]); // All >= 0.5
    assert_eq!(trimmed_b, vec![1.5, -1.0, 2.5, 0.0]); // 0.0 is already zero

    // Step 2: Elect sign (majority vote)
    let signs = elect_sign(&[&trimmed_a, &trimmed_b]);

    // Position 0: both positive -> positive
    assert_eq!(signs[0], 1);
    // Position 1: +1.0 and -1.0 -> tie, resolve to positive (convention)
    assert_eq!(signs[1], 1);

    // Step 3: Merge with elected signs
    let ties_merged = ties_merge(&[&trimmed_a, &trimmed_b], &signs, &base_weights);
    assert_eq!(ties_merged.len(), base_weights.len());

    // ── 4. DARE (Drop And REscale) ──
    // Randomly drop parameters from task vectors, then rescale:
    //   1. For each parameter, drop with probability p
    //   2. Rescale remaining by 1/(1-p) to preserve expected magnitude
    //   3. Average the rescaled task vectors
    //
    // The sparsification reduces interference between models.

    let drop_rate = 0.3;
    let rescale_factor = 1.0 / (1.0 - drop_rate);
    assert!((rescale_factor - 1.0_f64 / 0.7).abs() < 1e-10);

    // Simulate DARE on task_vector_a with a fixed drop mask
    let drop_mask = vec![false, true, false, false]; // Drop position 1
    let dare_a: Vec<f64> = task_vector_a.iter().zip(&drop_mask)
        .map(|(&v, &dropped)| if dropped { 0.0 } else { v * rescale_factor })
        .collect();

    // Dropped position is zero
    assert_eq!(dare_a[1], 0.0);
    // Non-dropped positions are rescaled
    assert!((dare_a[0] - 0.5 * rescale_factor).abs() < 1e-10);

    // ── 5. Multi-model merge ──
    // Merge more than two models using weighted combination.
    //
    // Common use case: combine a code model, a math model, and a writing
    // model into a general-purpose assistant.

    let model_configs = vec![
        MergeSource { name: "code-expert".into(), weight: 0.4 },
        MergeSource { name: "math-expert".into(), weight: 0.3 },
        MergeSource { name: "writing-expert".into(), weight: 0.3 },
    ];

    let total_weight: f64 = model_configs.iter().map(|m| m.weight).sum();
    assert!((total_weight - 1.0).abs() < 1e-10, "Weights must sum to 1.0");

    // ── 6. Merge configuration ──

    let merge_config = MergeConfig {
        method: "ties".into(),
        models: model_configs,
        trim_threshold: 0.1,
        dare_drop_rate: None,       // Not used with TIES
        normalize_weights: true,
    };

    assert_eq!(merge_config.method, "ties");
    assert_eq!(merge_config.models.len(), 3);

    let dare_config = MergeConfig {
        method: "dare".into(),
        models: vec![
            MergeSource { name: "model-a".into(), weight: 0.5 },
            MergeSource { name: "model-b".into(), weight: 0.5 },
        ],
        trim_threshold: 0.0,
        dare_drop_rate: Some(0.3),
        normalize_weights: true,
    };

    assert_eq!(dare_config.dare_drop_rate, Some(0.3));

    // ── 7. When to merge vs when to train ──
    // Merge when:
    //   - Combining domain expertise from separate fine-tunes
    //   - Quick experimentation without GPU training
    //   - Creating balanced multi-skill models
    //
    // Train when:
    //   - Need precise control over behavior
    //   - Have enough data for the combined domain
    //   - Quality requirements are very high

    // ── 8. Key concepts ──
    // - Linear interpolation: W = alpha * W_A + (1-alpha) * W_B
    // - Task vectors: tau = W_fine_tuned - W_base (what the model learned)
    // - TIES: Trim + Elect Sign + Merge (resolves sign conflicts)
    // - DARE: Drop + Rescale (reduces parameter interference)
    // - Multi-model: combine N experts with weighted task vectors
    // - Weights must sum to 1.0 for normalized merging
    // - No training needed — merge is pure weight arithmetic

    println!("PASS: 07-align/07_merge");
}

fn linear_interpolate(a: &[f64], b: &[f64], alpha: f64) -> Vec<f64> {
    a.iter().zip(b.iter())
        .map(|(&va, &vb)| alpha * va + (1.0 - alpha) * vb)
        .collect()
}

fn trim(values: &[f64], threshold: f64) -> Vec<f64> {
    values.iter()
        .map(|&v| if v.abs() >= threshold { v } else { 0.0 })
        .collect()
}

/// Elect sign by majority vote across task vectors.
/// Returns +1 or -1 for each position.
fn elect_sign(vectors: &[&[f64]]) -> Vec<i32> {
    let len = vectors[0].len();
    (0..len)
        .map(|i| {
            let positive_count = vectors.iter()
                .filter(|v| v[i] > 0.0)
                .count();
            let negative_count = vectors.iter()
                .filter(|v| v[i] < 0.0)
                .count();
            if positive_count >= negative_count { 1 } else { -1 }
        })
        .collect()
}

/// Merge trimmed task vectors using elected signs, then add back to base.
fn ties_merge(vectors: &[&[f64]], signs: &[i32], base: &[f64]) -> Vec<f64> {
    let len = base.len();
    (0..len)
        .map(|i| {
            let agreeing: Vec<f64> = vectors.iter()
                .map(|v| v[i])
                .filter(|&v| {
                    (signs[i] > 0 && v > 0.0) || (signs[i] < 0 && v < 0.0)
                })
                .collect();
            let avg = if agreeing.is_empty() {
                0.0
            } else {
                agreeing.iter().sum::<f64>() / agreeing.len() as f64
            };
            base[i] + avg
        })
        .collect()
}

struct MergeSource {
    name: String,
    weight: f64,
}

struct MergeConfig {
    method: String,
    models: Vec<MergeSource>,
    trim_threshold: f64,
    dare_drop_rate: Option<f64>,
    normalize_weights: bool,
}
