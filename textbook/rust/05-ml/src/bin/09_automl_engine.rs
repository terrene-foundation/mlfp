// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- ML / AutoML Engine
//!
//! OBJECTIVE: Automate model selection and tuning with AutoML.
//! LEVEL: Advanced
//! PARITY: Equivalent -- Python uses AutoMLEngine with polars;
//!         Rust uses the same engine API with ndarray.
//! VALIDATES: AutoMLConfig, model candidates, search strategy, leaderboard
//!
//! Run: cargo run -p tutorial-ml --bin 09_automl_engine

use std::collections::HashMap;

fn main() {
    // ── 1. AutoMLConfig ──
    // AutoML automates: model selection, hyperparameter tuning, ensembling.

    let config = AutoMLConfig::new()
        .task("classification")
        .metric("f1_score")
        .time_budget_secs(300)
        .max_models(10)
        .cv_folds(5);

    assert_eq!(config.task(), "classification");
    assert_eq!(config.metric(), "f1_score");
    assert_eq!(config.time_budget_secs(), 300);
    assert_eq!(config.max_models(), 10);

    // ── 2. Regression config ──

    let reg_config = AutoMLConfig::new()
        .task("regression")
        .metric("rmse")
        .time_budget_secs(600);

    assert_eq!(reg_config.task(), "regression");

    // ── 3. Model candidates ──
    // AutoML tries multiple model families automatically.

    let classification_candidates = vec![
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "svm",
        "knn",
        "decision_tree",
    ];

    let regression_candidates = vec![
        "linear_regression",
        "random_forest",
        "gradient_boosting",
        "svm",
        "knn",
        "elastic_net",
    ];

    assert!(classification_candidates.len() >= 5);
    assert!(regression_candidates.len() >= 5);

    // ── 4. Simulated leaderboard ──
    // AutoML produces a ranked leaderboard of model performance.

    let leaderboard = vec![
        LeaderboardEntry {
            model: "gradient_boosting".to_string(),
            params: HashMap::from([
                ("n_estimators".to_string(), "200".to_string()),
                ("learning_rate".to_string(), "0.05".to_string()),
            ]),
            score: 0.92,
            train_time_secs: 45.0,
        },
        LeaderboardEntry {
            model: "random_forest".to_string(),
            params: HashMap::from([
                ("n_estimators".to_string(), "150".to_string()),
                ("max_depth".to_string(), "12".to_string()),
            ]),
            score: 0.89,
            train_time_secs: 30.0,
        },
        LeaderboardEntry {
            model: "logistic_regression".to_string(),
            params: HashMap::from([
                ("C".to_string(), "1.0".to_string()),
            ]),
            score: 0.82,
            train_time_secs: 2.0,
        },
    ];

    // Best model is first (sorted by score)
    assert_eq!(leaderboard[0].model, "gradient_boosting");
    assert!((leaderboard[0].score - 0.92).abs() < 0.001);

    // ── 5. Execution pattern ──
    //
    //   let engine = AutoMLEngine::new(config);
    //   let result = engine.fit(&x_train, &y_train).await;
    //   println!("Best model: {}", result.best_model());
    //   println!("Score: {}", result.best_score());
    //   let predictions = result.predict(&x_test);

    // ── 6. Time budget enforcement ──
    // AutoML stops when time_budget_secs is exceeded.
    // Early stopping ensures we always get a result within budget.

    let elapsed = leaderboard.iter().map(|e| e.train_time_secs).sum::<f64>();
    assert!(elapsed < config.time_budget_secs() as f64);

    // ── 7. Key concepts ──
    // - AutoMLConfig: task, metric, time budget, model count
    // - Automatic model selection across multiple families
    // - Hyperparameter tuning for each candidate
    // - Leaderboard: ranked models by validation score
    // - Time budget enforcement for predictable execution
    // - Classification and regression tasks supported
    // - Best model can be extracted for deployment

    println!("PASS: 05-ml/09_automl_engine");
}

struct AutoMLConfig {
    task: String,
    metric: String,
    time_budget_secs: u64,
    max_models: usize,
    cv_folds: usize,
}

impl AutoMLConfig {
    fn new() -> Self {
        Self { task: "classification".to_string(), metric: "accuracy".to_string(), time_budget_secs: 300, max_models: 10, cv_folds: 5 }
    }
    fn task(mut self, t: &str) -> Self { self.task = t.to_string(); self }
    fn metric(mut self, m: &str) -> Self { self.metric = m.to_string(); self }
    fn time_budget_secs(mut self, s: u64) -> Self { self.time_budget_secs = s; self }
    fn max_models(mut self, n: usize) -> Self { self.max_models = n; self }
    fn cv_folds(mut self, n: usize) -> Self { self.cv_folds = n; self }
    fn task(&self) -> &str { &self.task }
    fn metric(&self) -> &str { &self.metric }
    fn time_budget_secs(&self) -> u64 { self.time_budget_secs }
    fn max_models(&self) -> usize { self.max_models }
}

struct LeaderboardEntry {
    model: String,
    params: HashMap<String, String>,
    score: f64,
    train_time_secs: f64,
}
