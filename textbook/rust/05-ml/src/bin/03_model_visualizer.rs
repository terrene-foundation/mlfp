// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- ML / Model Visualizer
//!
//! OBJECTIVE: Visualize model performance and data distributions.
//! LEVEL: Basic
//! PARITY: Equivalent -- Python uses ModelVisualizer with plotly;
//!         Rust generates visualization data structures for rendering.
//! VALIDATES: Confusion matrix, ROC data, feature importance, residual plots
//!
//! Run: cargo run -p tutorial-ml --bin 03_model_visualizer

use kailash_ml_core::validation::{confusion_matrix, ClassificationMetrics};
use ndarray::array;

fn main() {
    // ── 1. Confusion matrix ──
    // The confusion matrix shows predicted vs actual class labels.
    // Essential for understanding classification model performance.

    let y_true = array![0, 0, 0, 1, 1, 1, 2, 2, 2];
    let y_pred = array![0, 0, 1, 1, 1, 0, 2, 2, 1];

    let cm = confusion_matrix(&y_true, &y_pred, 3);
    assert_eq!(cm.shape(), &[3, 3]);

    // Class 0: 2 correct, 0 as class 1, 0 as class 2
    assert_eq!(cm[[0, 0]], 2); // True positives for class 0
    assert_eq!(cm[[0, 1]], 1); // Class 0 predicted as class 1

    // Class 1: 2 correct, 1 missed
    assert_eq!(cm[[1, 1]], 2); // True positives for class 1

    // ── 2. Classification metrics ──
    // Precision, recall, F1 score per class.

    let metrics = ClassificationMetrics::from_confusion_matrix(&cm);

    assert!(metrics.accuracy() > 0.6);
    assert!(metrics.precision(0) > 0.5);
    assert!(metrics.recall(0) > 0.5);
    assert!(metrics.f1_score(0) > 0.0);

    // ── 3. ROC curve data ──
    // For binary classification, ROC plots TPR vs FPR at various thresholds.

    let y_true_binary = array![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0];
    let y_scores = array![0.1, 0.3, 0.7, 0.9, 0.8, 0.4, 0.6, 0.2];

    // ROC computation: sort by score descending, compute TPR/FPR at each threshold
    let mut pairs: Vec<(f64, f64)> = y_scores
        .iter()
        .zip(y_true_binary.iter())
        .map(|(&s, &t)| (s, t))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let total_pos = y_true_binary.iter().filter(|&&v| v == 1.0).count() as f64;
    let total_neg = y_true_binary.iter().filter(|&&v| v == 0.0).count() as f64;

    let mut tpr_values = Vec::new();
    let mut fpr_values = Vec::new();
    let mut tp = 0.0;
    let mut fp = 0.0;

    for &(_, label) in &pairs {
        if label == 1.0 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        tpr_values.push(tp / total_pos);
        fpr_values.push(fp / total_neg);
    }

    // ROC should go from (0,0) to (1,1)
    assert!(*fpr_values.last().unwrap() <= 1.0);
    assert!(*tpr_values.last().unwrap() <= 1.0);

    // ── 4. Feature importance ──
    // Ranked list of features by their importance score.

    let feature_names = vec!["age", "income", "score", "department"];
    let importances = vec![0.35, 0.45, 0.15, 0.05];

    // Sort by importance descending
    let mut ranked: Vec<(&str, f64)> = feature_names
        .iter()
        .zip(importances.iter())
        .map(|(&n, &i)| (n, i))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    assert_eq!(ranked[0].0, "income");
    assert_eq!(ranked[1].0, "age");

    // ── 5. Residual analysis ──
    // For regression: residuals = actual - predicted.

    let y_actual = array![10.0, 20.0, 30.0, 40.0, 50.0];
    let y_predicted = array![12.0, 18.0, 31.0, 38.0, 52.0];
    let residuals = &y_actual - &y_predicted;

    assert!((residuals[0] - (-2.0)).abs() < 0.001);
    assert!((residuals[1] - 2.0).abs() < 0.001);

    // Mean residual should be close to 0 for an unbiased model
    let mean_residual: f64 = residuals.iter().sum::<f64>() / residuals.len() as f64;
    assert!(mean_residual.abs() < 1.0);

    // ── 6. Key concepts ──
    // - Confusion matrix: predicted vs actual class counts
    // - ClassificationMetrics: accuracy, precision, recall, F1
    // - ROC curve: TPR vs FPR at varying thresholds
    // - Feature importance: ranked contribution of each feature
    // - Residual analysis: actual - predicted for regression
    // - Python uses plotly for interactive charts; Rust generates data structures
    // - Visualization data can be serialized to JSON for any frontend

    println!("PASS: 05-ml/03_model_visualizer");
}
