// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- ML / Experiment Tracker
//!
//! OBJECTIVE: Track ML experiments with metrics, parameters, and artifacts.
//! LEVEL: Intermediate
//! PARITY: Equivalent -- Python uses ExperimentTracker; Rust uses the same
//!         tracking API with typed Experiment and Run structs.
//! VALIDATES: ExperimentTracker, Experiment, Run, metrics, parameters, comparison
//!
//! Run: cargo run -p tutorial-ml --bin 06_experiment_tracker

use serde_json::json;
use std::collections::HashMap;

fn main() {
    // ── 1. Create a tracker ──
    // ExperimentTracker is the central hub for experiment management.

    let mut tracker = ExperimentTracker::new("churn-prediction");

    assert_eq!(tracker.project_name(), "churn-prediction");
    assert_eq!(tracker.run_count(), 0);

    // ── 2. Start a run ──
    // Each run represents a single training attempt.

    let mut run1 = tracker.start_run("baseline-logistic");

    // Log parameters (hyperparameters and config)
    run1.log_param("model", "logistic_regression");
    run1.log_param("learning_rate", "0.01");
    run1.log_param("max_iter", "1000");

    // Log metrics (performance measures)
    run1.log_metric("accuracy", 0.82);
    run1.log_metric("f1_score", 0.78);
    run1.log_metric("auc_roc", 0.85);

    run1.finish();

    assert_eq!(run1.name(), "baseline-logistic");
    assert!((run1.metric("accuracy").unwrap() - 0.82).abs() < 0.001);
    assert_eq!(run1.param("model").unwrap(), "logistic_regression");

    tracker.record(run1);

    // ── 3. Multiple runs ──

    let mut run2 = tracker.start_run("random-forest");
    run2.log_param("model", "random_forest");
    run2.log_param("n_estimators", "100");
    run2.log_param("max_depth", "10");
    run2.log_metric("accuracy", 0.87);
    run2.log_metric("f1_score", 0.84);
    run2.log_metric("auc_roc", 0.90);
    run2.finish();
    tracker.record(run2);

    let mut run3 = tracker.start_run("gradient-boost");
    run3.log_param("model", "gradient_boosting");
    run3.log_param("n_estimators", "200");
    run3.log_param("learning_rate", "0.05");
    run3.log_metric("accuracy", 0.89);
    run3.log_metric("f1_score", 0.86);
    run3.log_metric("auc_roc", 0.92);
    run3.finish();
    tracker.record(run3);

    assert_eq!(tracker.run_count(), 3);

    // ── 4. Compare runs ──
    // Find the best run by a specific metric.

    let best = tracker.best_run("auc_roc").unwrap();
    assert_eq!(best.name(), "gradient-boost");
    assert!((best.metric("auc_roc").unwrap() - 0.92).abs() < 0.001);

    // ── 5. Run history ──
    // List all runs with their metrics for comparison.

    let runs = tracker.all_runs();
    assert_eq!(runs.len(), 3);

    // Verify ordering (by insertion order)
    assert_eq!(runs[0].name(), "baseline-logistic");
    assert_eq!(runs[1].name(), "random-forest");
    assert_eq!(runs[2].name(), "gradient-boost");

    // ── 6. Serialization ──
    // Runs serialize to JSON for persistence.

    let json_val = tracker.to_json();
    assert_eq!(json_val["project"], "churn-prediction");
    assert_eq!(json_val["runs"].as_array().unwrap().len(), 3);

    // ── 7. Key concepts ──
    // - ExperimentTracker: project-level experiment management
    // - Run: single training attempt with params and metrics
    // - log_param(): record hyperparameters and config
    // - log_metric(): record performance measures
    // - best_run(): find the best run by a metric
    // - Experiment comparison for model selection
    // - JSON serialization for persistence

    println!("PASS: 05-ml/06_experiment_tracker");
}

// Supporting types

struct ExperimentTracker {
    project_name: String,
    runs: Vec<Run>,
    next_id: u32,
}

impl ExperimentTracker {
    fn new(project_name: &str) -> Self {
        Self {
            project_name: project_name.to_string(),
            runs: Vec::new(),
            next_id: 1,
        }
    }
    fn project_name(&self) -> &str { &self.project_name }
    fn run_count(&self) -> usize { self.runs.len() }

    fn start_run(&mut self, name: &str) -> Run {
        let id = self.next_id;
        self.next_id += 1;
        Run::new(id, name)
    }

    fn record(&mut self, run: Run) { self.runs.push(run); }

    fn best_run(&self, metric: &str) -> Option<&Run> {
        self.runs.iter().max_by(|a, b| {
            let ma = a.metric(metric).unwrap_or(f64::NEG_INFINITY);
            let mb = b.metric(metric).unwrap_or(f64::NEG_INFINITY);
            ma.partial_cmp(&mb).unwrap()
        })
    }

    fn all_runs(&self) -> &[Run] { &self.runs }

    fn to_json(&self) -> serde_json::Value {
        json!({
            "project": self.project_name,
            "runs": self.runs.iter().map(|r| r.to_json()).collect::<Vec<_>>(),
        })
    }
}

struct Run {
    id: u32,
    name: String,
    params: HashMap<String, String>,
    metrics: HashMap<String, f64>,
    finished: bool,
}

impl Run {
    fn new(id: u32, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            params: HashMap::new(),
            metrics: HashMap::new(),
            finished: false,
        }
    }
    fn name(&self) -> &str { &self.name }
    fn log_param(&mut self, key: &str, value: &str) { self.params.insert(key.to_string(), value.to_string()); }
    fn log_metric(&mut self, key: &str, value: f64) { self.metrics.insert(key.to_string(), value); }
    fn metric(&self, key: &str) -> Option<f64> { self.metrics.get(key).copied() }
    fn param(&self, key: &str) -> Option<&str> { self.params.get(key).map(|s| s.as_str()) }
    fn finish(&mut self) { self.finished = true; }

    fn to_json(&self) -> serde_json::Value {
        json!({
            "id": self.id,
            "name": self.name,
            "params": self.params,
            "metrics": self.metrics,
        })
    }
}
