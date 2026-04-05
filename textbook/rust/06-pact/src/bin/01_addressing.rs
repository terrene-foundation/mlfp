// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- PACT / D/T/R Addressing
//!
//! OBJECTIVE: Parse and validate D/T/R positional addresses.
//! LEVEL: Basic
//! PARITY: Full -- Python has Address.parse() with identical grammar;
//!         Rust uses Address::parse() with the same validation rules.
//! VALIDATES: Address, AddressSegment, NodeType, parse, depth, parent
//!
//! Run: cargo run -p tutorial-pact --bin 01_addressing

use kailash_governance::addressing::{Address, AddressSegment, NodeType};

fn main() {
    // ── 1. NodeType enum ──
    // PACT has three node types: Department (D), Team (T), Role (R).

    assert_eq!(NodeType::Department.as_char(), 'D');
    assert_eq!(NodeType::Team.as_char(), 'T');
    assert_eq!(NodeType::Role.as_char(), 'R');

    // ── 2. Parse a single AddressSegment ──
    // Segments look like "D1", "R2", "T3" -- a type char plus a 1-based index.

    let seg = AddressSegment::parse("D1").expect("valid segment");
    assert!(matches!(seg.node_type(), NodeType::Department));
    assert_eq!(seg.sequence(), 1);

    let seg_r = AddressSegment::parse("R3").expect("valid segment");
    assert!(matches!(seg_r.node_type(), NodeType::Role));
    assert_eq!(seg_r.sequence(), 3);

    // ── 3. Parse a full Address ──
    // Addresses are hyphen-separated segments: D1-R1-D2-R1-T1-R1
    // Grammar rule: every D or T must be immediately followed by exactly one R.

    let addr = Address::parse("D1-R1-D2-R1-T1-R1").expect("valid address");
    assert_eq!(addr.segments().len(), 6);
    assert_eq!(addr.depth(), 6);

    // ── 4. Segment inspection ──

    assert!(matches!(addr.segments()[0].node_type(), NodeType::Department));
    assert!(matches!(addr.segments()[1].node_type(), NodeType::Role));
    assert_eq!(addr.segments()[0].sequence(), 1);

    // ── 5. Last segment ──

    let last = addr.last_segment();
    assert!(matches!(last.node_type(), NodeType::Role));
    assert_eq!(last.sequence(), 1);

    // ── 6. Parent address ──
    // Dropping the last segment gives the parent.

    let parent = addr.parent().expect("has parent");
    assert_eq!(parent.depth(), 5);

    // Root-level address
    let root = Address::parse("D1-R1").expect("valid");
    let root_parent = root.parent().expect("D1 is parent");
    assert_eq!(root_parent.depth(), 1);
    assert!(root_parent.parent().is_none());

    // ── 7. Invalid addresses ──
    // Grammar violations are detected at parse time.

    // Empty string
    assert!(Address::parse("").is_err());

    // Invalid segment format
    assert!(Address::parse("X1-R1").is_err());

    // Missing R after D
    assert!(Address::parse("D1-D2").is_err());

    // ── 8. Address display ──
    // Addresses have a canonical string representation.

    let addr_str = format!("{}", addr);
    assert_eq!(addr_str, "D1-R1-D2-R1-T1-R1");

    // ── 9. Key concepts ──
    // - NodeType: Department (D), Team (T), Role (R)
    // - AddressSegment: type + sequence (e.g., D1, R3, T2)
    // - Address: hyphen-separated segments with grammar rules
    // - Grammar: every D/T must be followed by exactly one R
    // - depth(): number of segments
    // - parent(): address without the last segment
    // - Parse-time validation catches grammar violations

    println!("PASS: 06-pact/01_addressing");
}
