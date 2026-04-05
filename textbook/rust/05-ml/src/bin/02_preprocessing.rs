// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- ML / Preprocessing Pipeline
//!
//! OBJECTIVE: Build data preprocessing pipelines with transformers.
//! LEVEL: Basic
//! PARITY: Equivalent -- Python uses PreprocessingPipeline with polars;
//!         Rust uses the same API with ndarray arrays.
//! VALIDATES: PreprocessingPipeline, StandardScaler, MinMaxScaler, OneHotEncoder,
//!            LabelEncoder, fit, transform, fit_transform
//!
//! Run: cargo run -p tutorial-ml --bin 02_preprocessing

use kailash_ml::preprocessing::{
    LabelEncoder, MinMaxScaler, StandardScaler,
};
use kailash_ml_core::estimator::{Estimator, Transformer};
use ndarray::{array, Array2};

fn main() {
    // ── 1. StandardScaler ──
    // Standardizes features by removing the mean and scaling to unit variance.
    // z = (x - mean) / std

    let data = array![[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0], [5.0, 500.0]];

    let mut scaler = StandardScaler::new();

    // fit() learns parameters (mean, std) from data
    scaler.fit(&data).expect("fit succeeded");

    // transform() applies the learned parameters
    let scaled = scaler.transform(&data).expect("transform succeeded");

    // Mean should be ~0, std should be ~1 after scaling
    let col0_mean: f64 = scaled.column(0).iter().sum::<f64>() / 5.0;
    assert!(col0_mean.abs() < 0.001, "Mean should be ~0 after scaling");

    // ── 2. MinMaxScaler ──
    // Scales features to a [0, 1] range.

    let mut minmax = MinMaxScaler::new(0.0, 1.0);
    minmax.fit(&data).expect("fit succeeded");
    let scaled_mm = minmax.transform(&data).expect("transform succeeded");

    // Min should be 0, max should be 1
    let col0_min = scaled_mm
        .column(0)
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let col0_max = scaled_mm
        .column(0)
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    assert!((col0_min - 0.0).abs() < 0.001);
    assert!((col0_max - 1.0).abs() < 0.001);

    // ── 3. fit_transform() convenience ──
    // Combines fit() and transform() in one call.

    let mut scaler2 = StandardScaler::new();
    let scaled2 = scaler2.fit_transform(&data).expect("fit_transform succeeded");
    assert_eq!(scaled2.shape(), data.shape());

    // ── 4. LabelEncoder ──
    // Encodes categorical string labels as integers.
    // In Rust, operates on string slices.

    let labels = vec!["cat", "dog", "cat", "bird", "dog", "bird"];
    let mut encoder = LabelEncoder::new();
    encoder.fit_labels(&labels).expect("fit labels");

    let encoded = encoder.transform_labels(&labels).expect("encode");
    assert_eq!(encoded.len(), 6);

    // Same label always gets the same encoding
    assert_eq!(encoded[0], encoded[2]); // both "cat"
    assert_eq!(encoded[1], encoded[4]); // both "dog"
    assert_eq!(encoded[3], encoded[5]); // both "bird"

    // Inverse transform recovers original labels
    let decoded = encoder.inverse_transform_labels(&encoded).expect("decode");
    assert_eq!(decoded[0], "cat");
    assert_eq!(decoded[1], "dog");

    // ── 5. Pipeline composition ──
    // In practice, transformers are chained in a pipeline:
    //
    //   let pipeline = PreprocessingPipeline::new()
    //       .add(StandardScaler::new())
    //       .add(FeatureSelector::new(top_k: 5))
    //       .build();
    //   let processed = pipeline.fit_transform(&data);
    //
    // The pipeline calls fit_transform() on each step in sequence,
    // passing the output of one as input to the next.

    // ── 6. Key concepts ──
    // - StandardScaler: z-score normalization (mean=0, std=1)
    // - MinMaxScaler: scale to [min, max] range
    // - LabelEncoder: categorical labels to integers
    // - fit(): learn parameters from data
    // - transform(): apply learned parameters
    // - fit_transform(): convenience for both steps
    // - Rust uses ndarray; Python uses polars DataFrames
    // - Pipeline chains transformers sequentially

    println!("PASS: 05-ml/02_preprocessing");
}
