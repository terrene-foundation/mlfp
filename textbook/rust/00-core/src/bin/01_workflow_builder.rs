// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Core / Workflow Builder
//!
//! OBJECTIVE: Build and validate workflows using WorkflowBuilder, add_node, connect, and build.
//! LEVEL: Basic
//! PARITY: Equivalent -- DIV-007: Rust build() does more pre-computation (topological sort,
//!         execution levels, input routing) than Python's build().
//! VALIDATES: WorkflowBuilder, Workflow, NodeRegistry, BuildError
//!
//! Run: cargo run -p tutorial-core --bin 01_workflow_builder

use std::sync::Arc;

use kailash_core::{
    error::BuildError,
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
    // -- Registry Setup --
    // Before building any workflow, we need a NodeRegistry populated with the
    // node types we plan to use. Registration is explicit in Rust -- there is
    // no global auto-discovery.

    let mut registry = NodeRegistry::new();
    register_system_nodes(&mut registry);
    register_control_flow_nodes(&mut registry);
    register_transform_nodes(&mut registry);

    // System: NoOpNode, LogNode
    // Control flow: SwitchNode, MergeNode, LoopNode, ConditionalNode, WaitNode,
    //               ParallelNode, ErrorHandlerNode, RetryNode
    // Transform: JSONTransformNode, TextTransformNode, DataMapperNode,
    //            SchemaValidatorNode, FormatConverterNode, ArrayOperationsNode,
    //            StringOperationsNode, MathOperationsNode
    assert!(registry.get_metadata("NoOpNode").is_some());
    assert!(registry.get_metadata("LogNode").is_some());
    assert_eq!(registry.len(), 18); // 2 system + 8 control flow + 8 transform

    let registry = Arc::new(registry);

    // -- Creating a WorkflowBuilder --
    // WorkflowBuilder collects node and connection specs, then validates
    // everything at build() time.

    let builder = WorkflowBuilder::new();
    assert!(!builder.has_nodes());

    // -- Adding Nodes --
    // add_node(type_name, node_id, config)
    //  - type_name: must match a registered factory
    //  - node_id:   unique within this workflow
    //  - config:    ValueMap passed to the factory's create()

    let mut builder = WorkflowBuilder::new();
    builder
        .add_node("NoOpNode", "source", ValueMap::new())
        .add_node("NoOpNode", "sink", ValueMap::new());
    assert!(builder.has_nodes());

    // -- Connecting Nodes --
    // connect(source_id, source_output, target_id, target_input)
    // Always 4 parameters -- Rust has only connect(), no add_connection() alias.
    // (DIV-020)

    builder.connect("source", "data", "sink", "data");

    // -- Building the Workflow --
    // build() consumes the builder and returns Result<Workflow, BuildError>.
    // Validation steps:
    //   1. Empty workflow check
    //   2. Duplicate node ID check
    //   3. Node type resolution via registry
    //   4. Node instantiation via factories
    //   5. Connection validation (source/target existence)
    //   6. Cycle detection (when cycles disabled)
    //   7. Topological sort computation
    //   8. Execution level computation (DIV-007)
    //   9. Input routing pre-computation (DIV-007)

    let workflow = builder.build(&registry).expect("build should succeed");
    assert_eq!(workflow.node_count(), 2);
    assert_eq!(workflow.connection_count(), 1);

    // Execution levels: source is level 0, sink is level 1.
    let levels = workflow.execution_levels();
    assert_eq!(levels.len(), 2);

    // -- Execution --
    // runtime.execute(workflow, inputs) -- NEVER workflow.execute(runtime).
    // The Runtime owns the execution loop, semaphore, and metrics.

    let runtime = Runtime::new(RuntimeConfig::default(), Arc::clone(&registry));

    let inputs = value_map! { "data" => "hello" };
    let result = runtime
        .execute_sync(&workflow, inputs)
        .expect("execute should succeed");

    // ExecutionResult contains per-node output maps and metadata (DIV-008).
    assert_eq!(result.metadata.nodes_executed, 2);
    assert_eq!(result.metadata.levels_executed, 2);
    assert!(!result.run_id.is_empty());

    // The sink received the data from the source via the connection.
    let sink_output = result.results.get("sink").expect("sink should have output");
    assert_eq!(
        sink_output.get("data" as &str),
        Some(&Value::from("hello"))
    );

    // -- Auto-Generated IDs --
    // add_node_auto_id() generates snake_case IDs from the type name.

    let mut builder = WorkflowBuilder::new();
    let id = builder.add_node_auto_id("NoOpNode", ValueMap::new());
    assert_eq!(id, "no_op_node_0");
    let id2 = builder.add_node_auto_id("NoOpNode", ValueMap::new());
    assert_eq!(id2, "no_op_node_1");

    // -- Error Handling: Empty Workflow --
    let empty_builder = WorkflowBuilder::new();
    let err = empty_builder.build(&registry).unwrap_err();
    assert!(matches!(err, BuildError::EmptyWorkflow));

    // -- Error Handling: Duplicate Node ID --
    let mut dup_builder = WorkflowBuilder::new();
    dup_builder
        .add_node("NoOpNode", "same_id", ValueMap::new())
        .add_node("NoOpNode", "same_id", ValueMap::new());
    let err = dup_builder.build(&registry).unwrap_err();
    assert!(matches!(err, BuildError::DuplicateNodeId { .. }));

    // -- Error Handling: Unknown Node Type --
    let mut unknown_builder = WorkflowBuilder::new();
    unknown_builder.add_node("NonExistentNode", "x", ValueMap::new());
    let err = unknown_builder.build(&registry).unwrap_err();
    assert!(matches!(err, BuildError::UnknownNodeType { .. }));

    // -- Error Handling: Invalid Connection --
    let mut bad_conn_builder = WorkflowBuilder::new();
    bad_conn_builder
        .add_node("NoOpNode", "a", ValueMap::new())
        .connect("a", "data", "nonexistent", "data");
    let err = bad_conn_builder.build(&registry).unwrap_err();
    assert!(matches!(err, BuildError::InvalidConnection { .. }));

    // -- Three-Node Pipeline --
    // Demonstrates a linear pipeline: producer -> transformer -> consumer.

    let mut builder = WorkflowBuilder::new();
    builder
        .add_node("NoOpNode", "producer", ValueMap::new())
        .add_node("NoOpNode", "transformer", ValueMap::new())
        .add_node("NoOpNode", "consumer", ValueMap::new())
        .connect("producer", "data", "transformer", "data")
        .connect("transformer", "data", "consumer", "data");

    let workflow = builder.build(&registry).expect("three-node build");
    assert_eq!(workflow.node_count(), 3);
    assert_eq!(workflow.connection_count(), 2);

    // Execution levels: producer(0) -> transformer(1) -> consumer(2)
    assert_eq!(workflow.execution_levels().len(), 3);

    let result = runtime
        .execute_sync(&workflow, value_map! { "data" => "pipeline" })
        .expect("three-node execute");

    let consumer_out = result.results.get("consumer").expect("consumer output");
    assert_eq!(
        consumer_out.get("data" as &str),
        Some(&Value::from("pipeline"))
    );

    // -- Workflow Definition Export --
    // to_definition() serializes the builder state without consuming it.

    let mut builder = WorkflowBuilder::new();
    builder
        .add_node("NoOpNode", "a", ValueMap::new())
        .add_node("NoOpNode", "b", ValueMap::new())
        .connect("a", "data", "b", "data");

    let def = builder.to_definition();
    assert_eq!(def.nodes.len(), 2);
    assert_eq!(def.connections.len(), 1);
    assert_eq!(def.version, "1.0");

    println!("PASS: 00-core/01_workflow_builder");
}
