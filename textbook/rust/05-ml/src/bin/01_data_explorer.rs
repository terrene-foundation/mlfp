// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- ML / DataExplorer
//!
//! OBJECTIVE: Profile datasets using DataExplorer for statistical analysis and alerts.
//! LEVEL: Basic
//! PARITY: Equivalent -- Python uses polars DataFrames; Rust uses ndarray Dataset
//!         with the same profiling, correlation, and alert APIs.
//! VALIDATES: DataExplorer, DataProfile, ColumnProfile, AlertConfig, profiling
//!
//! Run: cargo run -p tutorial-ml --bin 01_data_explorer

use kailash_ml::explorer::{AlertConfig, DataExplorer, DataProfile};
use ndarray::Array2;

fn main() {
    // ── 1. Create sample data ──
    // DataExplorer works with ndarray Array2 in Rust (polars in Python).

    let data = Array2::from_shape_vec(
        (10, 3),
        vec![
            25.0, 30000.0, 88.0,
            30.0, 45000.0, f64::NAN,
            35.0, 55000.0, 72.0,
            40.0, 60000.0, 95.0,
            45.0, 70000.0, f64::NAN,
            50.0, 80000.0, 63.0,
            55.0, 90000.0, 77.0,
            60.0, 100000.0, 81.0,
            65.0, 110000.0, f64::NAN,
            70.0, 120000.0, 90.0,
        ],
    )
    .expect("valid shape");

    let columns = vec!["age".to_string(), "income".to_string(), "score".to_string()];

    // ── 2. Create a DataExplorer ──
    // DataExplorer is the entry point for dataset profiling.

    let explorer = DataExplorer::new(data.clone(), columns.clone());

    assert_eq!(explorer.num_rows(), 10);
    assert_eq!(explorer.num_columns(), 3);

    // ── 3. Profile the dataset ──
    // profile() computes comprehensive statistics for every column.

    let profile = explorer.profile();

    assert!(profile.is_ok());
    let profile = profile.unwrap();

    assert_eq!(profile.row_count(), 10);
    assert_eq!(profile.column_count(), 3);

    // ── 4. Column profiles ──
    // Each column gets individual statistics.

    let age_profile = profile.column("age").expect("age column exists");
    assert!((age_profile.mean() - 47.5).abs() < 0.1);
    assert_eq!(age_profile.missing_count(), 0);
    assert_eq!(age_profile.missing_pct(), 0.0);

    let score_profile = profile.column("score").expect("score column exists");
    assert_eq!(score_profile.missing_count(), 3); // 3 NaN values
    assert!((score_profile.missing_pct() - 30.0).abs() < 0.1);

    // ── 5. Correlation matrix ──
    // DataExplorer computes pairwise Pearson correlations.

    let correlations = explorer.correlation_matrix();
    assert!(correlations.is_ok());

    let corr = correlations.unwrap();
    // Diagonal should be 1.0
    assert!((corr[[0, 0]] - 1.0).abs() < 0.001);
    // age and income should be highly correlated
    assert!(corr[[0, 1]] > 0.9);

    // ── 6. AlertConfig ──
    // Configure thresholds for data quality alerts.

    let alerts = AlertConfig::new()
        .missing_threshold(0.2)  // Alert if >20% missing
        .cardinality_threshold(0.9) // Alert if >90% unique
        .skew_threshold(2.0);    // Alert if skewness > 2.0

    assert!((alerts.missing_threshold() - 0.2).abs() < 0.001);

    // ── 7. Run alerts ──
    // check_alerts() returns a list of quality issues found.

    let issues = explorer.check_alerts(&alerts);
    // "score" has 30% missing, which exceeds the 20% threshold
    assert!(issues.iter().any(|issue| issue.column() == "score"));

    // ── 8. Dataset comparison ──
    // compare() highlights differences between two datasets.

    let data2 = Array2::from_shape_vec(
        (10, 3),
        vec![
            26.0, 31000.0, 89.0,
            31.0, 46000.0, 75.0,
            36.0, 56000.0, 73.0,
            41.0, 61000.0, 96.0,
            46.0, 71000.0, 80.0,
            51.0, 81000.0, 64.0,
            56.0, 91000.0, 78.0,
            61.0, 101000.0, 82.0,
            66.0, 111000.0, 70.0,
            71.0, 121000.0, 91.0,
        ],
    )
    .expect("valid shape");

    let explorer2 = DataExplorer::new(data2, columns);
    let comparison = explorer.compare(&explorer2);
    assert!(comparison.is_ok());

    // ── 9. Key concepts ──
    // - DataExplorer: entry point for dataset profiling
    // - DataProfile: comprehensive statistics per column
    // - ColumnProfile: mean, median, std, missing, min, max, skew
    // - correlation_matrix(): pairwise Pearson correlations
    // - AlertConfig: configurable data quality thresholds
    // - check_alerts(): returns quality issues exceeding thresholds
    // - compare(): highlights differences between two datasets
    // - Python uses polars DataFrames; Rust uses ndarray Array2

    println!("PASS: 05-ml/01_data_explorer");
}
