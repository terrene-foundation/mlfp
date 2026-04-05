// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Core / Connections
//!
//! OBJECTIVE: Understand connect() method, port wiring, default connections, and explicit mapping.
//! LEVEL: Basic
//! PARITY: Equivalent -- DIV-020: Rust uses only connect() with 4 parameters.
//!         Python has both connect() and add_connection() as aliases; Rust has only connect().
//! VALIDATES: WorkflowBuilder::connect, ConnectionData, InputRoute, Workflow::input_routes
//!
//! Run: cargo run -p tutorial-core --bin 04_connections

use std::sync::Arc;

use kailash_core::{
    node::NodeRegistry,
    nodes::{
        control_flow::register_control_flow_nodes, system::register_system_nodes,
        transform::register_transform_nodes,
    },
    runtime::{Runtime, RuntimeConfig},
    workflow::WorkflowBuilder,
};
use kailash_value::{value_map, Value, ValueMap};

fn main() {
    let mut registry = NodeRegistry::new();
    register_system_nodes(&mut registry);
    register_control_flow_nodes(&mut registry);
    register_transform_nodes(&mut registry);
    let registry = Arc::new(registry);

    let runtime = Runtime::new(RuntimeConfig::default(), Arc::clone(&registry));

    // ── Basic Connection: Same Port Names ──
    // The most common pattern: connect output "data" to input "data".
    //
    // connect(source_id, source_output, target_id, target_input)
    //
    // This is always 4 parameters. There is no 2-parameter or 3-parameter
    // form in Rust (DIV-020).

    let mut builder = WorkflowBuilder::new();
    builder
        .add_node("NoOpNode", "a", ValueMap::new())
        .add_node("NoOpNode", "b", ValueMap::new())
        .connect("a", "data", "b", "data");

    let workflow = builder.build(&registry).expect("same-port build");
    assert_eq!(workflow.connection_count(), 1);

    let result = runtime
        .execute_sync(&workflow, value_map! { "data" => "direct" })
        .expect("same-port execute");

    let b_output = result.results.get("b").expect("b output");
    assert_eq!(b_output.get("data" as &str), Some(&Value::from("direct")));

    // ── Cross-Port Mapping ──
    // Connect output "result" of one node to input "data" of another.
    // JSONTransformNode outputs "result"; NoOpNode accepts "data".

    let mut builder = WorkflowBuilder::new();
    let json_config = value_map! {
        "expression" => "name"
    };
    builder
        .add_node("NoOpNode", "source", ValueMap::new())
        .add_node("JSONTransformNode", "extractor", json_config)
        .add_node("NoOpNode", "sink", ValueMap::new())
        .connect("source", "data", "extractor", "data")
        .connect("extractor", "result", "sink", "data");

    let workflow = builder.build(&registry).expect("cross-port build");
    assert_eq!(workflow.connection_count(), 2);

    // Build a nested object as input.
    let input_obj = Value::Object({
        let mut m = std::collections::BTreeMap::new();
        m.insert(Arc::from("name"), Value::from("Kailash"));
        m.insert(Arc::from("version"), Value::from("1.0"));
        m
    });

    let result = runtime
        .execute_sync(
            &workflow,
            {
                let mut m = ValueMap::new();
                m.insert(Arc::from("data"), input_obj);
                m
            },
        )
        .expect("cross-port execute");

    // The extractor pulled "name" from the object; the sink received it.
    let sink_out = result.results.get("sink").expect("sink output");
    assert_eq!(
        sink_out.get("data" as &str),
        Some(&Value::from("Kailash"))
    );

    // ── Fan-Out: One Source to Multiple Targets ──
    // A single node's output can feed into multiple downstream nodes.

    let mut builder = WorkflowBuilder::new();
    builder
        .add_node("NoOpNode", "origin", ValueMap::new())
        .add_node("NoOpNode", "branch_a", ValueMap::new())
        .add_node("NoOpNode", "branch_b", ValueMap::new())
        .connect("origin", "data", "branch_a", "data")
        .connect("origin", "data", "branch_b", "data");

    let workflow = builder.build(&registry).expect("fan-out build");
    assert_eq!(workflow.node_count(), 3);
    assert_eq!(workflow.connection_count(), 2);

    // Both branches are at the same execution level (level 1) since they
    // only depend on origin (level 0). They execute concurrently.
    let levels = workflow.execution_levels();
    assert_eq!(levels.len(), 2); // level 0: origin, level 1: branch_a + branch_b

    let result = runtime
        .execute_sync(&workflow, value_map! { "data" => "shared" })
        .expect("fan-out execute");

    let a_out = result.results.get("branch_a").expect("branch_a output");
    let b_out = result.results.get("branch_b").expect("branch_b output");
    assert_eq!(a_out.get("data" as &str), Some(&Value::from("shared")));
    assert_eq!(b_out.get("data" as &str), Some(&Value::from("shared")));

    // ── Fan-In: Multiple Sources to One Target ──
    // MergeNode collects inputs from multiple upstream nodes.

    let mut builder = WorkflowBuilder::new();
    builder
        .add_node("NoOpNode", "left", ValueMap::new())
        .add_node("NoOpNode", "right", ValueMap::new())
        .add_node("MergeNode", "merge", ValueMap::new())
        .connect("left", "data", "merge", "left_data")
        .connect("right", "data", "merge", "right_data");

    let workflow = builder.build(&registry).expect("fan-in build");
    assert_eq!(workflow.connection_count(), 2);

    // Level 0: left + right (independent), Level 1: merge
    let levels = workflow.execution_levels();
    assert_eq!(levels.len(), 2);

    let result = runtime
        .execute_sync(
            &workflow,
            value_map! { "data" => "hello" },
        )
        .expect("fan-in execute");

    let merge_out = result.results.get("merge").expect("merge output");
    // MergeNode puts all non-null inputs into a "merged" object.
    let merged = merge_out
        .get("merged" as &str)
        .expect("merged key")
        .as_object()
        .expect("merged is object");
    assert!(merged.contains_key("left_data" as &str));
    assert!(merged.contains_key("right_data" as &str));

    // ── Diamond Pattern ──
    // source -> [branch_a, branch_b] -> merge
    // Tests that data flows correctly through a diamond DAG.

    let mut builder = WorkflowBuilder::new();
    builder
        .add_node("NoOpNode", "source", ValueMap::new())
        .add_node("NoOpNode", "left_path", ValueMap::new())
        .add_node("NoOpNode", "right_path", ValueMap::new())
        .add_node("MergeNode", "join", ValueMap::new())
        .connect("source", "data", "left_path", "data")
        .connect("source", "data", "right_path", "data")
        .connect("left_path", "data", "join", "from_left")
        .connect("right_path", "data", "from_right", "from_right");

    // Oops -- that last connection has a wrong target_node. Let's fix it.
    let mut builder = WorkflowBuilder::new();
    builder
        .add_node("NoOpNode", "source", ValueMap::new())
        .add_node("NoOpNode", "left_path", ValueMap::new())
        .add_node("NoOpNode", "right_path", ValueMap::new())
        .add_node("MergeNode", "join", ValueMap::new())
        .connect("source", "data", "left_path", "data")
        .connect("source", "data", "right_path", "data")
        .connect("left_path", "data", "join", "from_left")
        .connect("right_path", "data", "join", "from_right");

    let workflow = builder.build(&registry).expect("diamond build");
    assert_eq!(workflow.node_count(), 4);
    assert_eq!(workflow.connection_count(), 4);

    // Execution levels:
    //   Level 0: source
    //   Level 1: left_path, right_path (parallel)
    //   Level 2: join
    assert_eq!(workflow.execution_levels().len(), 3);

    let result = runtime
        .execute_sync(&workflow, value_map! { "data" => "diamond" })
        .expect("diamond execute");

    let join_out = result.results.get("join").expect("join output");
    let merged = join_out
        .get("merged" as &str)
        .expect("merged key")
        .as_object()
        .expect("merged is object");
    assert_eq!(
        merged.get("from_left" as &str),
        Some(&Value::from("diamond"))
    );
    assert_eq!(
        merged.get("from_right" as &str),
        Some(&Value::from("diamond"))
    );

    // ── Input Routes (Pre-Computed at Build Time) ──
    // The Workflow pre-computes input routes for O(1) lookup during execution.
    // This is a Rust-specific optimization (DIV-007).

    let _join_node = workflow.get_node("join").expect("join node");
    let join_idx = *workflow
        .execution_levels()
        .last()
        .expect("last level")
        .first()
        .expect("first node in last level");

    let routes = workflow.input_routes(join_idx);
    assert_eq!(routes.len(), 2); // Two incoming connections

    // Each route specifies source_node index, source_output, and target_input.
    let route_inputs: Vec<&str> = routes.iter().map(|r| r.target_input.as_ref()).collect();
    assert!(route_inputs.contains(&"from_left"));
    assert!(route_inputs.contains(&"from_right"));

    // ── Connection Validation ──
    // Connections referencing non-existent nodes fail at build() time.

    let mut builder = WorkflowBuilder::new();
    builder
        .add_node("NoOpNode", "a", ValueMap::new())
        .connect("a", "data", "ghost", "data");

    let err = builder.build(&registry).unwrap_err();
    assert!(format!("{err}").contains("ghost"));

    println!("PASS: 00-core/04_connections");
}
