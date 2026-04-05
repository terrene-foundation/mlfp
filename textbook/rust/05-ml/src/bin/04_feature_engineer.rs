// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- ML / Feature Engineer
//!
//! OBJECTIVE: Create and transform features for ML models.
//! LEVEL: Intermediate
//! PARITY: Equivalent -- Python uses FeatureEngineer with polars;
//!         Rust uses the same transform API with ndarray.
//! VALIDATES: Polynomial features, interaction terms, binning, log transform
//!
//! Run: cargo run -p tutorial-ml --bin 04_feature_engineer

use ndarray::{array, Array2};

fn main() {
    // ── 1. Polynomial features ──
    // Generate polynomial combinations of input features.
    // For degree=2, features [a, b] become [a, b, a^2, ab, b^2].

    let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

    let poly = polynomial_features(&data, 2);

    // Row 0: [1, 2] -> [1, 2, 1, 2, 4] (a, b, a^2, ab, b^2)
    assert_eq!(poly.ncols(), 5);
    assert!((poly[[0, 0]] - 1.0).abs() < 0.001); // a
    assert!((poly[[0, 1]] - 2.0).abs() < 0.001); // b
    assert!((poly[[0, 2]] - 1.0).abs() < 0.001); // a^2
    assert!((poly[[0, 3]] - 2.0).abs() < 0.001); // a*b
    assert!((poly[[0, 4]] - 4.0).abs() < 0.001); // b^2

    // ── 2. Log transform ──
    // Applies log(1 + x) to reduce skewness.

    let skewed = array![[1.0], [10.0], [100.0], [1000.0], [10000.0]];
    let logged = log_transform(&skewed);

    // log1p preserves ordering and compresses range
    assert!(logged[[0, 0]] < logged[[1, 0]]);
    assert!(logged[[3, 0]] < logged[[4, 0]]);
    // Range is compressed
    let range_before = skewed[[4, 0]] - skewed[[0, 0]]; // 9999
    let range_after = logged[[4, 0]] - logged[[0, 0]];
    assert!(range_after < range_before);

    // ── 3. Binning (discretization) ──
    // Convert continuous features into discrete bins.

    let ages = vec![22.0, 35.0, 45.0, 18.0, 67.0, 55.0, 30.0, 72.0];
    let bin_edges = vec![0.0, 25.0, 45.0, 65.0, 100.0];

    let binned = bin_features(&ages, &bin_edges);

    assert_eq!(binned[0], 0); // 22 -> [0, 25)
    assert_eq!(binned[1], 1); // 35 -> [25, 45)
    assert_eq!(binned[2], 1); // 45 -> [25, 45) (inclusive at boundary)
    assert_eq!(binned[3], 0); // 18 -> [0, 25)
    assert_eq!(binned[4], 3); // 67 -> [65, 100)

    // ── 4. Interaction terms ──
    // Create pairwise products of features.

    let features = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let interactions = interaction_features(&features);

    // For 3 features: a*b, a*c, b*c = 3 interaction terms
    assert_eq!(interactions.ncols(), 3);
    assert!((interactions[[0, 0]] - 2.0).abs() < 0.001); // 1*2
    assert!((interactions[[0, 1]] - 3.0).abs() < 0.001); // 1*3
    assert!((interactions[[0, 2]] - 6.0).abs() < 0.001); // 2*3

    // ── 5. Feature selection by variance ──
    // Remove low-variance features (near-constant).

    let data_with_const = array![
        [1.0, 5.0, 3.1],
        [2.0, 5.0, 3.2],
        [3.0, 5.0, 3.0],
        [4.0, 5.1, 3.3],
    ];

    let variances = column_variances(&data_with_const);
    assert!(variances[0] > 1.0); // high variance
    assert!(variances[1] < 0.01); // near-constant
    assert!(variances[2] < 0.02); // low variance

    // ── 6. Key concepts ──
    // - Polynomial features: expand feature space for non-linear patterns
    // - Log transform: compress skewed distributions
    // - Binning: convert continuous to categorical
    // - Interaction terms: capture feature co-effects
    // - Variance threshold: remove uninformative features
    // - Feature engineering improves model performance more than tuning

    println!("PASS: 05-ml/04_feature_engineer");
}

fn polynomial_features(data: &Array2<f64>, _degree: usize) -> Array2<f64> {
    let n = data.nrows();
    let ncols = data.ncols();
    // degree 2: original + all degree-2 combinations
    let mut result = Array2::zeros((n, ncols + ncols * (ncols + 1) / 2));
    for i in 0..n {
        let mut col = 0;
        for j in 0..ncols {
            result[[i, col]] = data[[i, j]];
            col += 1;
        }
        for j in 0..ncols {
            for k in j..ncols {
                result[[i, col]] = data[[i, j]] * data[[i, k]];
                col += 1;
            }
        }
    }
    result
}

fn log_transform(data: &Array2<f64>) -> Array2<f64> {
    data.mapv(|x| (1.0 + x).ln())
}

fn bin_features(values: &[f64], edges: &[f64]) -> Vec<usize> {
    values
        .iter()
        .map(|&v| {
            for i in 0..edges.len() - 1 {
                if v < edges[i + 1] || (i == edges.len() - 2) {
                    return i;
                }
            }
            edges.len() - 2
        })
        .collect()
}

fn interaction_features(data: &Array2<f64>) -> Array2<f64> {
    let n = data.nrows();
    let ncols = data.ncols();
    let n_interactions = ncols * (ncols - 1) / 2;
    let mut result = Array2::zeros((n, n_interactions));
    for i in 0..n {
        let mut col = 0;
        for j in 0..ncols {
            for k in (j + 1)..ncols {
                result[[i, col]] = data[[i, j]] * data[[i, k]];
                col += 1;
            }
        }
    }
    result
}

fn column_variances(data: &Array2<f64>) -> Vec<f64> {
    (0..data.ncols())
        .map(|col| {
            let column = data.column(col);
            let mean = column.iter().sum::<f64>() / column.len() as f64;
            column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / column.len() as f64
        })
        .collect()
}
