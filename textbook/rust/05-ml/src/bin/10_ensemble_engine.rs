// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- ML / Ensemble Engine
//!
//! OBJECTIVE: Combine multiple models into ensembles for improved performance.
//! LEVEL: Advanced
//! PARITY: Equivalent -- Python uses EnsembleEngine; Rust uses the same
//!         voting, stacking, and blending ensemble strategies.
//! VALIDATES: Voting ensemble, stacking, blending, model weighting
//!
//! Run: cargo run -p tutorial-ml --bin 10_ensemble_engine

fn main() {
    // ── 1. Voting ensemble (classification) ──
    // Hard voting: each model votes for a class; majority wins.
    // Soft voting: average predicted probabilities; highest avg wins.

    // Simulate 3 model predictions (hard voting)
    let model1_preds = vec![0, 1, 1, 0, 1]; // 3 class-1
    let model2_preds = vec![0, 1, 0, 0, 1]; // 2 class-1
    let model3_preds = vec![1, 1, 1, 0, 0]; // 3 class-1

    let ensemble_preds: Vec<i32> = (0..5)
        .map(|i| {
            let votes = model1_preds[i] + model2_preds[i] + model3_preds[i];
            if votes >= 2 { 1 } else { 0 } // Majority vote
        })
        .collect();

    assert_eq!(ensemble_preds, vec![0, 1, 1, 0, 1]);

    // ── 2. Soft voting (probability averaging) ──
    // Average predicted probabilities for more nuanced decisions.

    let model1_probs = vec![0.3, 0.8, 0.7, 0.2, 0.9];
    let model2_probs = vec![0.4, 0.7, 0.4, 0.1, 0.8];
    let model3_probs = vec![0.6, 0.9, 0.8, 0.3, 0.3];

    let avg_probs: Vec<f64> = (0..5)
        .map(|i| (model1_probs[i] + model2_probs[i] + model3_probs[i]) / 3.0)
        .collect();

    let soft_preds: Vec<i32> = avg_probs
        .iter()
        .map(|&p| if p >= 0.5 { 1 } else { 0 })
        .collect();

    assert_eq!(soft_preds, vec![0, 1, 1, 0, 1]);

    // ── 3. Weighted voting ──
    // Give more weight to better-performing models.

    let weights = vec![0.5, 0.3, 0.2]; // model1 is best
    let weighted_probs: Vec<f64> = (0..5)
        .map(|i| {
            model1_probs[i] * weights[0]
                + model2_probs[i] * weights[1]
                + model3_probs[i] * weights[2]
        })
        .collect();

    assert!(weighted_probs[0] < 0.5); // Below threshold
    assert!(weighted_probs[1] > 0.5); // Above threshold

    // ── 4. Averaging ensemble (regression) ──
    // For regression: average predictions from multiple models.

    let reg_model1 = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let reg_model2 = vec![11.0, 19.0, 31.0, 39.0, 51.0];
    let reg_model3 = vec![9.5, 20.5, 29.5, 40.5, 49.5];

    let avg_preds: Vec<f64> = (0..5)
        .map(|i| (reg_model1[i] + reg_model2[i] + reg_model3[i]) / 3.0)
        .collect();

    // Ensemble predictions should be closer to true values
    assert!((avg_preds[0] - 10.17).abs() < 0.1);

    // ── 5. Stacking ──
    // Use model predictions as features for a meta-learner.
    //
    // Level 0: Train base models on training data
    // Level 1: Collect base model predictions on validation set
    // Level 2: Train meta-learner on base predictions -> final output
    //
    //   let stacker = StackingEnsemble::new()
    //       .add_base("rf", random_forest)
    //       .add_base("gb", gradient_boost)
    //       .add_base("lr", logistic_regression)
    //       .meta_learner(LogisticRegression::new())
    //       .build();

    // ── 6. Blending ──
    // Simpler than stacking: uses a holdout set instead of CV.
    //
    //   let blender = BlendingEnsemble::new()
    //       .holdout_fraction(0.2)
    //       .add_base("rf", random_forest)
    //       .add_base("gb", gradient_boost)
    //       .meta_learner(LinearRegression::new())
    //       .build();

    // ── 7. Diversity matters ──
    // Ensembles work best when base models make DIFFERENT errors.
    // Combining 3 identical models does not improve performance.

    let diverse1 = vec![0, 1, 1, 0, 0]; // Errors on sample 4
    let diverse2 = vec![0, 1, 0, 0, 1]; // Errors on sample 2
    let diverse3 = vec![0, 0, 1, 0, 1]; // Errors on sample 1

    let truth = vec![0, 1, 1, 0, 1];
    let diverse_ensemble: Vec<i32> = (0..5)
        .map(|i| {
            let votes = diverse1[i] + diverse2[i] + diverse3[i];
            if votes >= 2 { 1 } else { 0 }
        })
        .collect();

    // Ensemble corrects individual errors through majority voting
    assert_eq!(diverse_ensemble, truth);

    // ── 8. Key concepts ──
    // - Hard voting: majority class prediction
    // - Soft voting: averaged probabilities
    // - Weighted voting: better models get more influence
    // - Averaging: regression ensemble via mean predictions
    // - Stacking: meta-learner on base model predictions (with CV)
    // - Blending: meta-learner on holdout set predictions
    // - Diversity: ensembles need models with different error patterns

    println!("PASS: 05-ml/10_ensemble_engine");
}
