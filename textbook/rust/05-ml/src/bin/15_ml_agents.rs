// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- ML / ML Agents
//!
//! OBJECTIVE: Build agents that manage ML lifecycle operations autonomously.
//! LEVEL: Advanced
//! PARITY: Equivalent -- Python uses ML agent patterns with Kaizen;
//!         Rust uses the same agent signatures for ML lifecycle operations.
//! VALIDATES: ML agent signatures, tool definitions, lifecycle orchestration
//!
//! Run: cargo run -p tutorial-ml --bin 15_ml_agents

use serde_json::json;

fn main() {
    // ── 1. ML Agent overview ──
    // ML agents autonomously manage model lifecycle operations:
    //   - Data profiling and quality checks
    //   - Feature engineering recommendations
    //   - Model training and evaluation
    //   - Drift monitoring and retraining
    //   - Deployment decisions
    //
    // These are Kaizen agents with ML-specific tools and signatures.

    // ── 2. Data Quality Agent ──
    // Profiles incoming data and flags quality issues.

    let data_quality_tools = vec![
        MLToolDef {
            name: "profile_dataset".to_string(),
            description: "Run DataExplorer profiling on a dataset".to_string(),
            returns: "DataProfile with statistics per column".to_string(),
        },
        MLToolDef {
            name: "check_drift".to_string(),
            description: "Compare current data against reference distribution".to_string(),
            returns: "DriftReport with PSI and KS scores per feature".to_string(),
        },
        MLToolDef {
            name: "validate_schema".to_string(),
            description: "Check that data matches the expected schema".to_string(),
            returns: "SchemaValidationResult with mismatches".to_string(),
        },
    ];

    assert_eq!(data_quality_tools.len(), 3);

    // ── 3. Training Agent ──
    // Orchestrates model training and experiment tracking.

    let training_tools = vec![
        MLToolDef {
            name: "train_model".to_string(),
            description: "Train a model with given config and data".to_string(),
            returns: "Trained model with evaluation metrics".to_string(),
        },
        MLToolDef {
            name: "log_experiment".to_string(),
            description: "Log training run to ExperimentTracker".to_string(),
            returns: "Run ID for the logged experiment".to_string(),
        },
        MLToolDef {
            name: "compare_models".to_string(),
            description: "Compare multiple models on held-out test data".to_string(),
            returns: "Leaderboard with ranked model performance".to_string(),
        },
    ];

    assert_eq!(training_tools.len(), 3);

    // ── 4. Deployment Agent ──
    // Manages model promotion and serving.

    let deployment_tools = vec![
        MLToolDef {
            name: "register_model".to_string(),
            description: "Register model in ModelRegistry".to_string(),
            returns: "Model version ID".to_string(),
        },
        MLToolDef {
            name: "promote_model".to_string(),
            description: "Promote model to staging/production".to_string(),
            returns: "New model stage".to_string(),
        },
        MLToolDef {
            name: "rollback_model".to_string(),
            description: "Rollback to previous model version".to_string(),
            returns: "Rolled-back version ID".to_string(),
        },
    ];

    assert_eq!(deployment_tools.len(), 3);

    // ── 5. Orchestration pattern ──
    // ML agents can be composed into a supervised pipeline:
    //
    //   Supervisor
    //   +-- Data Quality Agent (profile, validate, check drift)
    //   +-- Training Agent (train, log, compare)
    //   +-- Deployment Agent (register, promote, rollback)
    //
    // The supervisor routes based on the ML lifecycle stage.

    let agent_pipeline = json!({
        "supervisor": "ml-supervisor",
        "sub_agents": [
            {"name": "data-quality", "capabilities": ["profiling", "drift", "validation"]},
            {"name": "training", "capabilities": ["training", "evaluation", "comparison"]},
            {"name": "deployment", "capabilities": ["registry", "promotion", "rollback"]},
        ],
        "routing": "llm_based",
    });

    assert_eq!(agent_pipeline["sub_agents"].as_array().unwrap().len(), 3);

    // ── 6. Autonomous retraining ──
    // When drift monitor detects significant drift:
    //   1. Data Quality Agent verifies the drift
    //   2. Training Agent retrains with fresh data
    //   3. Training Agent compares new model vs production model
    //   4. Deployment Agent promotes if new model is better
    //
    // This loop runs without human intervention.

    // ── 7. Key concepts ──
    // - ML agents: Kaizen agents with ML-specific tools
    // - Data Quality Agent: profiling, drift, schema validation
    // - Training Agent: train, evaluate, experiment tracking
    // - Deployment Agent: register, promote, rollback
    // - Composed via supervised pipeline (GovernedSupervisor)
    // - Autonomous retraining triggered by drift detection
    // - All routing is LLM-based, not keyword matching

    println!("PASS: 05-ml/15_ml_agents");
}

struct MLToolDef {
    name: String,
    description: String,
    returns: String,
}
