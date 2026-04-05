// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Align / Adapter Registry
//!
//! OBJECTIVE: Manage adapter lifecycle — registration, versioning, lookup, and promotion.
//! LEVEL: Intermediate
//! PARITY: Full -- kailash-align-serving provides DefaultAdapterManager with concurrent
//!         registration, lookup, listing, and removal. AdapterId is a typed UUID wrapper.
//! VALIDATES: DefaultAdapterManager, AdapterId, AdapterInfo, AdapterMetadata, versioning
//!
//! Run: cargo run -p tutorial-align --bin 06_adapter_registry

use std::path::PathBuf;

use kailash_align_serving::adapter::{AdapterMetadata, DefaultAdapterManager, TrainingMethod};
use kailash_align_serving::model::{AdapterId, AdapterInfo};

fn main() {
    // ── 1. Create adapter manager ──
    // DefaultAdapterManager is a concurrent registry backed by DashMap.
    // It is Clone-able (wraps Arc<DashMap>) and thread-safe.

    let manager = DefaultAdapterManager::new();
    assert!(manager.is_empty());
    assert_eq!(manager.len(), 0);

    // ── 2. Register an adapter ──
    // Registration associates an AdapterId with runtime info and metadata.

    let id_v1 = AdapterId::new();
    let info_v1 = AdapterInfo {
        id: id_v1.clone(),
        path: PathBuf::from("/models/adapters/finance-v1.bin"),
        name: "finance-v1".into(),
        scale: 1.0,
        ..Default::default()
    };
    let meta_v1 = AdapterMetadata {
        name: "finance-v1".into(),
        description: "Financial analysis adapter v1".into(),
        method: TrainingMethod::Lora,
        rank: 16,
        alpha: 32.0,
        target_modules: vec!["q_proj".into(), "v_proj".into()],
        base_model: "llama-3-8b".into(),
        version: "1.0".into(),
        tags: vec!["finance".into(), "production".into()],
        ..Default::default()
    };

    manager.register(id_v1.clone(), info_v1, meta_v1);

    assert!(!manager.is_empty());
    assert_eq!(manager.len(), 1);
    assert!(manager.contains(&id_v1));

    // ── 3. Lookup adapter info ──

    let info = manager.get_info(&id_v1).expect("adapter should exist");
    assert_eq!(info.name, "finance-v1");
    assert!((info.scale - 1.0).abs() < f32::EPSILON);

    // ── 4. Lookup adapter metadata ──

    let meta = manager.get_metadata(&id_v1).expect("metadata should exist");
    assert_eq!(meta.method, TrainingMethod::Lora);
    assert_eq!(meta.rank, 16);
    assert_eq!(meta.version, "1.0");

    // ── 5. Register a second version ──
    // Versioning is handled by the version field in AdapterMetadata.
    // Each adapter gets a unique AdapterId regardless of version.

    let id_v2 = AdapterId::new();
    let info_v2 = AdapterInfo {
        id: id_v2.clone(),
        path: PathBuf::from("/models/adapters/finance-v2.bin"),
        name: "finance-v2".into(),
        scale: 1.0,
        ..Default::default()
    };
    let meta_v2 = AdapterMetadata {
        name: "finance-v2".into(),
        description: "Financial analysis adapter v2 — improved on Q3 data".into(),
        method: TrainingMethod::Lora,
        rank: 32,  // Increased rank for v2
        alpha: 64.0,
        target_modules: vec!["q_proj".into(), "k_proj".into(), "v_proj".into(), "o_proj".into()],
        base_model: "llama-3-8b".into(),
        version: "2.0".into(),
        tags: vec!["finance".into(), "staging".into()],
        ..Default::default()
    };

    manager.register(id_v2.clone(), info_v2, meta_v2);

    assert_eq!(manager.len(), 2);
    assert!(manager.contains(&id_v1));
    assert!(manager.contains(&id_v2));

    // ── 6. List all adapters ──

    let all_adapters = manager.list();
    assert_eq!(all_adapters.len(), 2);

    // ── 7. AdapterId is a typed UUID wrapper ──
    // Prevents accidental confusion with other UUIDs in the system.

    let uuid = uuid::Uuid::new_v4();
    let id_from_uuid = AdapterId::from_uuid(uuid);
    assert_eq!(*id_from_uuid.as_uuid(), uuid);

    // AdapterId has Display
    let display = format!("{}", id_v1);
    assert!(!display.is_empty());

    // AdapterId is Hash-able (can be used as HashMap key)
    let mut map = std::collections::HashMap::new();
    map.insert(id_v1.clone(), "production");
    map.insert(id_v2.clone(), "staging");
    assert_eq!(map.get(&id_v1), Some(&"production"));

    // ── 8. Remove an adapter ──
    // Removing unregisters from the manager. The actual adapter file
    // on disk is not deleted (that is the caller's responsibility).

    let removed = manager.remove(&id_v1);
    assert!(removed.is_ok(), "should return removed adapter");
    assert_eq!(manager.len(), 1);
    assert!(!manager.contains(&id_v1));
    assert!(manager.contains(&id_v2));

    // Removing a non-existent adapter returns None
    let fake_id = AdapterId::new();
    assert!(manager.remove(&fake_id).is_err());

    // ── 9. Promotion pattern ──
    // A common workflow: train adapter -> register as staging -> evaluate
    // -> promote to production by updating tags.
    //
    // In Python:
    //   registry.promote("finance-v2", from_stage="staging", to_stage="production")
    //
    // In Rust, re-register with updated metadata:

    let promoted_meta = AdapterMetadata {
        name: "finance-v2".into(),
        description: "Financial analysis adapter v2 — promoted to production".into(),
        method: TrainingMethod::Lora,
        rank: 32,
        alpha: 64.0,
        target_modules: vec!["q_proj".into(), "k_proj".into(), "v_proj".into(), "o_proj".into()],
        base_model: "llama-3-8b".into(),
        version: "2.0".into(),
        tags: vec!["finance".into(), "production".into()], // Changed from staging
        ..Default::default()
    };

    let promoted_info = AdapterInfo {
        id: id_v2.clone(),
        path: PathBuf::from("/models/adapters/finance-v2.bin"),
        name: "finance-v2".into(),
        scale: 1.0,
        ..Default::default()
    };

    // Re-register replaces the existing entry
    manager.register(id_v2.clone(), promoted_info, promoted_meta);
    assert_eq!(manager.len(), 1); // Still 1 (replaced, not added)

    let updated_meta = manager.get_metadata(&id_v2).expect("should exist");
    assert!(updated_meta.tags.contains(&"production".to_string()));

    // ── 10. Key concepts ──
    // - DefaultAdapterManager: concurrent adapter registry (Clone + Send + Sync)
    // - AdapterId: typed UUID wrapper for adapter references
    // - AdapterInfo: runtime state (path, scale, loaded_at)
    // - AdapterMetadata: persistent provenance (method, rank, version, tags)
    // - Versioning: version field in metadata, unique AdapterId per version
    // - Promotion: re-register with updated tags (staging -> production)
    // - Thread-safe: backed by DashMap, safe for concurrent access

    println!("PASS: 07-align/06_adapter_registry");
}
