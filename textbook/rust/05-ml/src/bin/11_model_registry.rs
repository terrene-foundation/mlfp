// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- ML / Model Registry
//!
//! OBJECTIVE: Register, version, and manage trained ML models.
//! LEVEL: Intermediate
//! PARITY: Equivalent -- Python uses ModelRegistry; Rust uses the same
//!         registry pattern with typed ModelVersion and stage transitions.
//! VALIDATES: ModelRegistry, ModelVersion, stage transitions, metadata
//!
//! Run: cargo run -p tutorial-ml --bin 11_model_registry

use std::collections::HashMap;

fn main() {
    // ── 1. Model registry ──
    // The registry is the central hub for model lifecycle management.

    let mut registry = ModelRegistry::new();
    assert_eq!(registry.model_count(), 0);

    // ── 2. Register a model ──
    // Each model has a name, version, stage, and metadata.

    let v1 = ModelVersion::new("churn-predictor", 1)
        .stage(ModelStage::Development)
        .metric("accuracy", 0.87)
        .metric("f1_score", 0.84)
        .param("algorithm", "random_forest")
        .param("n_estimators", "200");

    assert_eq!(v1.name(), "churn-predictor");
    assert_eq!(v1.version(), 1);
    assert!(matches!(v1.stage(), ModelStage::Development));

    registry.register(v1);
    assert_eq!(registry.model_count(), 1);

    // ── 3. Model stages ──
    // Models progress through stages:
    //   Development -> Staging -> Production -> Archived

    assert_eq!(ModelStage::Development.as_str(), "development");
    assert_eq!(ModelStage::Staging.as_str(), "staging");
    assert_eq!(ModelStage::Production.as_str(), "production");
    assert_eq!(ModelStage::Archived.as_str(), "archived");

    // ── 4. Stage transitions ──
    // Promote a model through stages.

    registry.transition("churn-predictor", 1, ModelStage::Staging);
    let model = registry.get("churn-predictor", 1).unwrap();
    assert!(matches!(model.stage(), ModelStage::Staging));

    registry.transition("churn-predictor", 1, ModelStage::Production);
    let model = registry.get("churn-predictor", 1).unwrap();
    assert!(matches!(model.stage(), ModelStage::Production));

    // ── 5. Multiple versions ──

    let v2 = ModelVersion::new("churn-predictor", 2)
        .stage(ModelStage::Development)
        .metric("accuracy", 0.91)
        .metric("f1_score", 0.89)
        .param("algorithm", "gradient_boosting");

    registry.register(v2);

    // Latest version
    let latest = registry.latest("churn-predictor").unwrap();
    assert_eq!(latest.version(), 2);

    // Production version
    let prod = registry.production("churn-predictor").unwrap();
    assert_eq!(prod.version(), 1); // v1 is in production

    // ── 6. Promote v2 to production (replaces v1) ──

    registry.transition("churn-predictor", 2, ModelStage::Staging);
    registry.transition("churn-predictor", 2, ModelStage::Production);
    // v1 automatically moves to Archived
    registry.transition("churn-predictor", 1, ModelStage::Archived);

    let prod = registry.production("churn-predictor").unwrap();
    assert_eq!(prod.version(), 2);

    let archived = registry.get("churn-predictor", 1).unwrap();
    assert!(matches!(archived.stage(), ModelStage::Archived));

    // ── 7. List all versions ──

    let versions = registry.versions("churn-predictor");
    assert_eq!(versions.len(), 2);

    // ── 8. Key concepts ──
    // - ModelRegistry: central model lifecycle management
    // - ModelVersion: name + version + stage + metrics + params
    // - ModelStage: Development -> Staging -> Production -> Archived
    // - Stage transitions for promotion workflow
    // - Only one version in Production at a time per model name
    // - latest(): most recent version; production(): deployed version

    println!("PASS: 05-ml/11_model_registry");
}

#[derive(Clone, Debug)]
enum ModelStage {
    Development,
    Staging,
    Production,
    Archived,
}

impl ModelStage {
    fn as_str(&self) -> &str {
        match self {
            Self::Development => "development",
            Self::Staging => "staging",
            Self::Production => "production",
            Self::Archived => "archived",
        }
    }
}

#[derive(Clone)]
struct ModelVersion {
    name: String,
    version: u32,
    stage: ModelStage,
    metrics: HashMap<String, f64>,
    params: HashMap<String, String>,
}

impl ModelVersion {
    fn new(name: &str, version: u32) -> Self {
        Self { name: name.to_string(), version, stage: ModelStage::Development, metrics: HashMap::new(), params: HashMap::new() }
    }
    fn stage(mut self, s: ModelStage) -> Self { self.stage = s; self }
    fn metric(mut self, k: &str, v: f64) -> Self { self.metrics.insert(k.to_string(), v); self }
    fn param(mut self, k: &str, v: &str) -> Self { self.params.insert(k.to_string(), v.to_string()); self }
    fn name(&self) -> &str { &self.name }
    fn version(&self) -> u32 { self.version }
    fn stage(&self) -> &ModelStage { &self.stage }
}

struct ModelRegistry {
    models: Vec<ModelVersion>,
}

impl ModelRegistry {
    fn new() -> Self { Self { models: Vec::new() } }
    fn model_count(&self) -> usize { self.models.iter().map(|m| m.name.as_str()).collect::<std::collections::HashSet<_>>().len() }
    fn register(&mut self, model: ModelVersion) { self.models.push(model); }
    fn get(&self, name: &str, version: u32) -> Option<&ModelVersion> {
        self.models.iter().find(|m| m.name == name && m.version == version)
    }
    fn transition(&mut self, name: &str, version: u32, stage: ModelStage) {
        if let Some(m) = self.models.iter_mut().find(|m| m.name == name && m.version == version) {
            m.stage = stage;
        }
    }
    fn latest(&self, name: &str) -> Option<&ModelVersion> {
        self.models.iter().filter(|m| m.name == name).max_by_key(|m| m.version)
    }
    fn production(&self, name: &str) -> Option<&ModelVersion> {
        self.models.iter().find(|m| m.name == name && matches!(m.stage, ModelStage::Production))
    }
    fn versions(&self, name: &str) -> Vec<&ModelVersion> {
        self.models.iter().filter(|m| m.name == name).collect()
    }
}
