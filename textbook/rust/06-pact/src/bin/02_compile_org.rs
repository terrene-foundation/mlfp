// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- PACT / Organization Compilation
//!
//! OBJECTIVE: Compile organizational YAML into a validated governance structure.
//! LEVEL: Intermediate
//! PARITY: Full -- Python uses compile_org() from YAML;
//!         Rust uses the same compilation with typed OrgTree output.
//! VALIDATES: YAML parsing, organization tree, role resolution, address validation
//!
//! Run: cargo run -p tutorial-pact --bin 02_compile_org

use kailash_governance::compilation::compile_org;

fn main() {
    // ── 1. Organization YAML ──
    // PACT organizations are defined in YAML with D/T/R hierarchy.

    let yaml = r#"
organization:
  name: "Terrene Foundation"
  departments:
    - name: "Engineering"
      roles:
        - name: "CTO"
          clearance: 5
        - name: "Senior Engineer"
          clearance: 3
      teams:
        - name: "Backend"
          roles:
            - name: "Lead Developer"
              clearance: 4
            - name: "Developer"
              clearance: 2
        - name: "Frontend"
          roles:
            - name: "Lead Developer"
              clearance: 4
    - name: "Operations"
      roles:
        - name: "COO"
          clearance: 5
      teams:
        - name: "DevOps"
          roles:
            - name: "SRE"
              clearance: 3
"#;

    // ── 2. Compile the organization ──
    // compile_org() parses YAML and builds a validated OrgTree.

    let org = compile_org(yaml).expect("valid YAML");

    assert_eq!(org.name(), "Terrene Foundation");
    assert_eq!(org.department_count(), 2);

    // ── 3. Address resolution ──
    // Every node in the tree has a D/T/R address.
    //
    // D1-R1       = Engineering / CTO
    // D1-R2       = Engineering / Senior Engineer
    // D1-T1-R1    = Engineering / Backend / Lead Developer
    // D1-T1-R2    = Engineering / Backend / Developer
    // D1-T2-R1    = Engineering / Frontend / Lead Developer
    // D2-R1       = Operations / COO
    // D2-T1-R1    = Operations / DevOps / SRE

    let cto = org.resolve("D1-R1").expect("CTO exists");
    assert_eq!(cto.name(), "CTO");
    assert_eq!(cto.clearance(), 5);

    let dev = org.resolve("D1-T1-R2").expect("Developer exists");
    assert_eq!(dev.name(), "Developer");
    assert_eq!(dev.clearance(), 2);

    let sre = org.resolve("D2-T1-R1").expect("SRE exists");
    assert_eq!(sre.name(), "SRE");
    assert_eq!(sre.clearance(), 3);

    // ── 4. Invalid resolution ──
    // Resolving a nonexistent address returns None.

    assert!(org.resolve("D3-R1").is_none());
    assert!(org.resolve("D1-T5-R1").is_none());

    // ── 5. Department listing ──

    let depts = org.departments();
    assert_eq!(depts.len(), 2);
    assert_eq!(depts[0].name(), "Engineering");
    assert_eq!(depts[1].name(), "Operations");

    // ── 6. Key concepts ──
    // - YAML defines organizational hierarchy (D/T/R)
    // - compile_org() validates and builds an OrgTree
    // - Every node gets a deterministic D/T/R address
    // - resolve(): map address to node with name and clearance
    // - Clearance levels: 1 (lowest) to 5 (highest)
    // - Departments contain roles and teams; teams contain roles

    println!("PASS: 06-pact/02_compile_org");
}
