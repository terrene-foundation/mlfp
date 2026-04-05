// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- PACT / Knowledge Clearance
//!
//! OBJECTIVE: Control knowledge access based on clearance levels.
//! LEVEL: Advanced
//! PARITY: Full -- Both SDKs use KnowledgeStore with clearance-gated access.
//! VALIDATES: KnowledgeStore, KnowledgeItem, clearance-gated retrieval
//!
//! Run: cargo run -p tutorial-pact --bin 06_knowledge

use kailash_governance::knowledge::{KnowledgeItem, KnowledgeStore};

fn main() {
    // ── 1. KnowledgeStore ──
    // A clearance-gated repository of organizational knowledge.
    // Items are only accessible to actors with sufficient clearance.

    let mut store = KnowledgeStore::new();

    // ── 2. Add knowledge items ──
    // Each item has content, a category, and a required clearance level.

    store.add(KnowledgeItem::new(
        "company-handbook",
        "Employee handbook and policies",
        1, // Public - anyone can access
    ));

    store.add(KnowledgeItem::new(
        "api-architecture",
        "Internal API design and endpoints",
        2, // Internal
    ));

    store.add(KnowledgeItem::new(
        "customer-data-schema",
        "PII data schema and handling procedures",
        3, // Confidential
    ));

    store.add(KnowledgeItem::new(
        "security-keys",
        "Production encryption keys and rotation schedule",
        5, // Top Secret
    ));

    assert_eq!(store.total_items(), 4);

    // ── 3. Clearance-gated retrieval ──
    // retrieve() returns only items the actor can access.

    let public_access = store.retrieve(1);
    assert_eq!(public_access.len(), 1);
    assert_eq!(public_access[0].id(), "company-handbook");

    let internal_access = store.retrieve(2);
    assert_eq!(internal_access.len(), 2);

    let confidential_access = store.retrieve(3);
    assert_eq!(confidential_access.len(), 3);

    let top_secret_access = store.retrieve(5);
    assert_eq!(top_secret_access.len(), 4); // Can see everything

    // ── 4. Specific item lookup ──
    // get() returns an item only if the actor has clearance.

    assert!(store.get("company-handbook", 1).is_some());
    assert!(store.get("security-keys", 3).is_none()); // Needs level 5
    assert!(store.get("security-keys", 5).is_some());

    // ── 5. Knowledge categories ──
    // Items can be tagged for category-based retrieval.

    let mut categorized = KnowledgeStore::new();

    categorized.add(
        KnowledgeItem::new("handbook", "Employee handbook", 1)
            .category("hr"),
    );
    categorized.add(
        KnowledgeItem::new("api-docs", "API documentation", 2)
            .category("engineering"),
    );
    categorized.add(
        KnowledgeItem::new("runbooks", "Incident runbooks", 2)
            .category("engineering"),
    );

    let eng_items = categorized.by_category("engineering", 2);
    assert_eq!(eng_items.len(), 2);

    let hr_items = categorized.by_category("hr", 1);
    assert_eq!(hr_items.len(), 1);

    // ── 6. Audit trail ──
    // Every retrieval is logged for compliance.

    let audit = store.audit_log();
    assert!(audit.len() > 0, "Retrievals are logged");

    // ── 7. Key concepts ──
    // - KnowledgeStore: clearance-gated knowledge repository
    // - KnowledgeItem: content + clearance level + optional category
    // - retrieve(level): returns all items accessible at that clearance
    // - get(id, level): specific item lookup with clearance check
    // - Category-based retrieval for organized access
    // - Audit trail for every retrieval operation
    // - Fail-closed: insufficient clearance = no access

    println!("PASS: 06-pact/06_knowledge");
}
