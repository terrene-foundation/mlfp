// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- ML / Feature Store
//!
//! OBJECTIVE: Register, version, and retrieve feature sets for ML pipelines.
//! LEVEL: Intermediate
//! PARITY: Equivalent -- Python uses FeatureStore with polars;
//!         Rust uses FeatureStore with typed FeatureSet and versioning.
//! VALIDATES: FeatureStore, FeatureSet, FeatureSpec, versioning, retrieval
//!
//! Run: cargo run -p tutorial-ml --bin 05_feature_store

use serde_json::json;
use std::collections::HashMap;

fn main() {
    // ── 1. FeatureSpec -- individual feature metadata ──
    // Each feature has a name, data type, description, and optional tags.

    let age_spec = FeatureSpec {
        name: "age".to_string(),
        dtype: "float64".to_string(),
        description: "Customer age in years".to_string(),
        tags: vec!["demographic".to_string()],
    };

    assert_eq!(age_spec.name, "age");
    assert_eq!(age_spec.dtype, "float64");

    // ── 2. FeatureSet -- a group of related features ──
    // Feature sets bundle related features with metadata and versioning.

    let mut feature_set = FeatureSet::new("customer_features")
        .description("Core customer demographic features")
        .add_feature(age_spec)
        .add_feature(FeatureSpec {
            name: "income".to_string(),
            dtype: "float64".to_string(),
            description: "Annual income".to_string(),
            tags: vec!["financial".to_string()],
        })
        .add_feature(FeatureSpec {
            name: "tenure_months".to_string(),
            dtype: "int32".to_string(),
            description: "Months as customer".to_string(),
            tags: vec!["engagement".to_string()],
        });

    assert_eq!(feature_set.name(), "customer_features");
    assert_eq!(feature_set.feature_count(), 3);
    assert_eq!(feature_set.version(), 1);

    // ── 3. FeatureStore -- register and retrieve ──
    // The store manages all feature sets across the organization.

    let mut store = FeatureStore::new();

    store.register(feature_set);
    assert!(store.has_feature_set("customer_features"));
    assert!(!store.has_feature_set("nonexistent"));
    assert_eq!(store.count(), 1);

    // ── 4. Retrieve feature set ──

    let retrieved = store.get("customer_features").unwrap();
    assert_eq!(retrieved.feature_count(), 3);
    assert!(retrieved.has_feature("age"));
    assert!(retrieved.has_feature("income"));

    // ── 5. Versioning ──
    // Updating a feature set increments its version.

    let updated = FeatureSet::new("customer_features")
        .description("Customer features v2 with churn risk")
        .add_feature(FeatureSpec {
            name: "age".to_string(),
            dtype: "float64".to_string(),
            description: "Customer age".to_string(),
            tags: vec!["demographic".to_string()],
        })
        .add_feature(FeatureSpec {
            name: "churn_risk".to_string(),
            dtype: "float64".to_string(),
            description: "Predicted churn probability".to_string(),
            tags: vec!["derived".to_string()],
        });

    store.update("customer_features", updated);
    let v2 = store.get("customer_features").unwrap();
    assert_eq!(v2.version(), 2);

    // ── 6. Feature discovery by tag ──
    // Search for features across all sets by tag.

    let financial_features = store.find_by_tag("financial");
    // After update, "income" is no longer in the set
    assert_eq!(financial_features.len(), 0);

    let derived_features = store.find_by_tag("derived");
    assert_eq!(derived_features.len(), 1);
    assert_eq!(derived_features[0], "churn_risk");

    // ── 7. Serialization ──
    // Feature sets serialize to JSON for storage and API transfer.

    let fs = store.get("customer_features").unwrap();
    let json_val = fs.to_json();
    assert_eq!(json_val["name"], "customer_features");

    // ── 8. Key concepts ──
    // - FeatureSpec: individual feature metadata (name, dtype, tags)
    // - FeatureSet: bundle of related features with versioning
    // - FeatureStore: central registry for feature sets
    // - Versioning: update() increments version automatically
    // - Tag-based discovery for feature search
    // - Serialization to JSON for persistence

    println!("PASS: 05-ml/05_feature_store");
}

// Supporting types for the tutorial (would be in kailash-ml in production)

#[derive(Clone)]
struct FeatureSpec {
    name: String,
    dtype: String,
    description: String,
    tags: Vec<String>,
}

#[derive(Clone)]
struct FeatureSet {
    name: String,
    description: String,
    features: Vec<FeatureSpec>,
    version: u32,
}

impl FeatureSet {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: String::new(),
            features: Vec::new(),
            version: 1,
        }
    }

    fn description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    fn add_feature(mut self, spec: FeatureSpec) -> Self {
        self.features.push(spec);
        self
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn feature_count(&self) -> usize {
        self.features.len()
    }

    fn version(&self) -> u32 {
        self.version
    }

    fn has_feature(&self, name: &str) -> bool {
        self.features.iter().any(|f| f.name == name)
    }

    fn to_json(&self) -> serde_json::Value {
        json!({
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "features": self.features.iter().map(|f| json!({
                "name": f.name,
                "dtype": f.dtype,
                "description": f.description,
                "tags": f.tags,
            })).collect::<Vec<_>>(),
        })
    }
}

struct FeatureStore {
    sets: HashMap<String, FeatureSet>,
}

impl FeatureStore {
    fn new() -> Self {
        Self {
            sets: HashMap::new(),
        }
    }

    fn register(&mut self, set: FeatureSet) {
        self.sets.insert(set.name.clone(), set);
    }

    fn has_feature_set(&self, name: &str) -> bool {
        self.sets.contains_key(name)
    }

    fn count(&self) -> usize {
        self.sets.len()
    }

    fn get(&self, name: &str) -> Option<&FeatureSet> {
        self.sets.get(name)
    }

    fn update(&mut self, name: &str, mut set: FeatureSet) {
        if let Some(existing) = self.sets.get(name) {
            set.version = existing.version + 1;
        }
        self.sets.insert(name.to_string(), set);
    }

    fn find_by_tag(&self, tag: &str) -> Vec<String> {
        let mut results = Vec::new();
        for set in self.sets.values() {
            for feature in &set.features {
                if feature.tags.iter().any(|t| t == tag) {
                    results.push(feature.name.clone());
                }
            }
        }
        results
    }
}
