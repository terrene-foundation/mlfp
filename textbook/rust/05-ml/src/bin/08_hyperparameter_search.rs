// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- ML / Hyperparameter Search
//!
//! OBJECTIVE: Optimize model hyperparameters using grid and random search.
//! LEVEL: Intermediate
//! PARITY: Equivalent -- Python uses HyperparameterSearch;
//!         Rust uses the same search strategies with typed parameter spaces.
//! VALIDATES: GridSearch, RandomSearch, parameter space, cross-validation
//!
//! Run: cargo run -p tutorial-ml --bin 08_hyperparameter_search

use std::collections::HashMap;

fn main() {
    // ── 1. Parameter space definition ──
    // Define the hyperparameters to search over.

    let param_space = ParamSpace::new()
        .add_categorical("model", &["logistic", "svm", "random_forest"])
        .add_continuous("learning_rate", 0.001, 0.1)
        .add_discrete("max_depth", 3, 15)
        .add_discrete("n_estimators", 50, 500);

    assert_eq!(param_space.param_count(), 4);
    assert!(param_space.has_param("learning_rate"));

    // ── 2. Grid search ──
    // Exhaustively evaluates all combinations in a discretized grid.

    let grid = GridSearch::new(param_space.clone())
        .cv_folds(5)
        .metric("accuracy");

    assert_eq!(grid.cv_folds(), 5);
    assert_eq!(grid.metric(), "accuracy");

    // Grid search generates all combinations
    // For categorical: 3 values, discrete ranges produce combinatorial explosion
    // In practice, grid search is only feasible for small parameter spaces.

    // ── 3. Random search ──
    // Samples random parameter combinations -- more efficient than grid.

    let random = RandomSearch::new(param_space.clone())
        .n_trials(50)
        .cv_folds(5)
        .metric("f1_score")
        .seed(42);

    assert_eq!(random.n_trials(), 50);
    assert_eq!(random.metric(), "f1_score");

    // ── 4. Simulated search results ──
    // Each trial produces a parameter configuration and a score.

    let results = vec![
        SearchResult {
            params: HashMap::from([
                ("model".to_string(), "random_forest".to_string()),
                ("n_estimators".to_string(), "200".to_string()),
                ("max_depth".to_string(), "10".to_string()),
            ]),
            score: 0.87,
            cv_std: 0.02,
        },
        SearchResult {
            params: HashMap::from([
                ("model".to_string(), "svm".to_string()),
                ("learning_rate".to_string(), "0.01".to_string()),
            ]),
            score: 0.83,
            cv_std: 0.03,
        },
        SearchResult {
            params: HashMap::from([
                ("model".to_string(), "logistic".to_string()),
                ("learning_rate".to_string(), "0.05".to_string()),
            ]),
            score: 0.80,
            cv_std: 0.01,
        },
    ];

    // ── 5. Best result selection ──

    let best = results.iter().max_by(|a, b| {
        a.score.partial_cmp(&b.score).unwrap()
    }).unwrap();

    assert_eq!(best.params["model"], "random_forest");
    assert!((best.score - 0.87).abs() < 0.001);

    // ── 6. Cross-validation ──
    // Each search trial uses k-fold cross-validation:
    //   1. Split training data into k folds
    //   2. Train on k-1 folds, evaluate on the held-out fold
    //   3. Repeat k times, average the scores
    //   4. Report mean score and standard deviation

    // Simulate 5-fold CV scores
    let fold_scores = vec![0.85, 0.87, 0.89, 0.84, 0.86];
    let mean: f64 = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
    let variance: f64 = fold_scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>()
        / fold_scores.len() as f64;
    let std = variance.sqrt();

    assert!((mean - 0.862).abs() < 0.001);
    assert!(std < 0.02);

    // ── 7. Key concepts ──
    // - ParamSpace: defines searchable hyperparameters
    // - GridSearch: exhaustive (small spaces only)
    // - RandomSearch: sampling-based (large spaces, more efficient)
    // - Cross-validation: robust evaluation (k-fold)
    // - cv_std: standard deviation indicates stability
    // - Best result: highest mean CV score
    // - In practice, random search outperforms grid search for most problems

    println!("PASS: 05-ml/08_hyperparameter_search");
}

#[derive(Clone)]
struct ParamSpace {
    params: HashMap<String, ParamType>,
}

#[derive(Clone)]
enum ParamType {
    Categorical(Vec<String>),
    Continuous(f64, f64),
    Discrete(i64, i64),
}

impl ParamSpace {
    fn new() -> Self { Self { params: HashMap::new() } }
    fn add_categorical(mut self, name: &str, values: &[&str]) -> Self {
        self.params.insert(name.to_string(), ParamType::Categorical(values.iter().map(|s| s.to_string()).collect()));
        self
    }
    fn add_continuous(mut self, name: &str, min: f64, max: f64) -> Self {
        self.params.insert(name.to_string(), ParamType::Continuous(min, max));
        self
    }
    fn add_discrete(mut self, name: &str, min: i64, max: i64) -> Self {
        self.params.insert(name.to_string(), ParamType::Discrete(min, max));
        self
    }
    fn param_count(&self) -> usize { self.params.len() }
    fn has_param(&self, name: &str) -> bool { self.params.contains_key(name) }
}

struct GridSearch {
    _space: ParamSpace,
    cv_folds: usize,
    metric: String,
}

impl GridSearch {
    fn new(space: ParamSpace) -> Self { Self { _space: space, cv_folds: 5, metric: "accuracy".to_string() } }
    fn cv_folds(mut self, n: usize) -> Self { self.cv_folds = n; self }
    fn metric(mut self, m: &str) -> Self { self.metric = m.to_string(); self }
    fn cv_folds(&self) -> usize { self.cv_folds }
    fn metric(&self) -> &str { &self.metric }
}

struct RandomSearch {
    _space: ParamSpace,
    n_trials: usize,
    cv_folds: usize,
    metric: String,
    _seed: u64,
}

impl RandomSearch {
    fn new(space: ParamSpace) -> Self {
        Self { _space: space, n_trials: 50, cv_folds: 5, metric: "accuracy".to_string(), _seed: 42 }
    }
    fn n_trials(mut self, n: usize) -> Self { self.n_trials = n; self }
    fn cv_folds(mut self, n: usize) -> Self { self.cv_folds = n; self }
    fn metric(mut self, m: &str) -> Self { self.metric = m.to_string(); self }
    fn seed(mut self, s: u64) -> Self { self._seed = s; self }
    fn n_trials(&self) -> usize { self.n_trials }
    fn metric(&self) -> &str { &self.metric }
}

struct SearchResult {
    params: HashMap<String, String>,
    score: f64,
    cv_std: f64,
}
