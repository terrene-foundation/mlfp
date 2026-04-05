// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- ML / Training Pipeline
//!
//! OBJECTIVE: Build end-to-end ML training pipelines with split, train, evaluate.
//! LEVEL: Intermediate
//! PARITY: Equivalent -- Python uses TrainingPipeline with polars;
//!         Rust uses the same pipeline pattern with ndarray and kailash-ml estimators.
//! VALIDATES: Train/test split, model fitting, prediction, evaluation metrics
//!
//! Run: cargo run -p tutorial-ml --bin 07_training_pipeline

use kailash_ml::linear::LinearRegression;
use kailash_ml_core::estimator::{Estimator, Predictor};
use ndarray::{array, Array1, Array2};

fn main() {
    // ── 1. Prepare data ──
    // A simple dataset: predict y from x1, x2.

    let x = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
        [5.0, 6.0],
        [6.0, 7.0],
        [7.0, 8.0],
        [8.0, 9.0],
        [9.0, 10.0],
        [10.0, 11.0],
    ];

    // y = 2*x1 + 3*x2 + noise
    let y = array![8.1, 13.0, 18.2, 23.1, 28.0, 32.9, 38.1, 43.0, 48.2, 53.1];

    // ── 2. Train/test split ──
    // Split data into training and test sets (80/20).

    let split_idx = (x.nrows() as f64 * 0.8) as usize;

    let x_train = x.slice(ndarray::s![..split_idx, ..]).to_owned();
    let y_train = y.slice(ndarray::s![..split_idx]).to_owned();
    let x_test = x.slice(ndarray::s![split_idx.., ..]).to_owned();
    let y_test = y.slice(ndarray::s![split_idx..]).to_owned();

    assert_eq!(x_train.nrows(), 8);
    assert_eq!(x_test.nrows(), 2);

    // ── 3. Train a model ──
    // LinearRegression from kailash-ml.

    let mut model = LinearRegression::new();
    model.fit(&x_train, &y_train).expect("training succeeded");

    // ── 4. Predict ──

    let predictions = model.predict(&x_test).expect("prediction succeeded");
    assert_eq!(predictions.len(), 2);

    // Predictions should be close to actual values
    for (pred, actual) in predictions.iter().zip(y_test.iter()) {
        assert!(
            (pred - actual).abs() < 2.0,
            "Prediction {pred} should be close to {actual}"
        );
    }

    // ── 5. Evaluation metrics ──

    let mse = mean_squared_error(&y_test, &predictions);
    assert!(mse < 2.0, "MSE should be small for a good linear fit");

    let rmse = mse.sqrt();
    assert!(rmse < 1.5);

    let r2 = r_squared(&y_test, &predictions);
    assert!(r2 > 0.9, "R-squared should be high for near-linear data");

    // ── 6. Pipeline pattern ──
    // In practice, the full pipeline is:
    //   1. Load data (ASCENTDataLoader)
    //   2. Preprocess (StandardScaler, encoding)
    //   3. Feature engineering
    //   4. Train/test split
    //   5. Model training
    //   6. Evaluation
    //   7. Model registry (if metrics pass threshold)
    //
    //   let pipeline = TrainingPipeline::new()
    //       .preprocessor(StandardScaler::new())
    //       .model(LinearRegression::new())
    //       .evaluator(RegressionEvaluator::new())
    //       .build();
    //   let result = pipeline.fit_evaluate(&x, &y).await;

    // ── 7. Key concepts ──
    // - Train/test split: 80/20 for evaluation
    // - fit(): learn model parameters from training data
    // - predict(): generate predictions on new data
    // - MSE, RMSE, R-squared for regression evaluation
    // - Pipeline chains preprocessing, training, evaluation
    // - Rust uses ndarray; Python uses polars DataFrames

    println!("PASS: 05-ml/07_training_pipeline");
}

fn mean_squared_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let diff = y_true - y_pred;
    diff.mapv(|x| x * x).mean().unwrap()
}

fn r_squared(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let mean = y_true.mean().unwrap();
    let ss_res: f64 = y_true.iter().zip(y_pred.iter()).map(|(t, p)| (t - p).powi(2)).sum();
    let ss_tot: f64 = y_true.iter().map(|t| (t - mean).powi(2)).sum();
    1.0 - ss_res / ss_tot
}
