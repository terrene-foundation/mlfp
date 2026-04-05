// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Core / Node Types
//!
//! OBJECTIVE: Explore built-in node types and the NodeRegistry discovery system.
//! LEVEL: Basic
//! PARITY: Equivalent -- DIV-011: Rust uses registration functions
//!         (register_system_nodes, register_control_flow_nodes, register_transform_nodes)
//!         rather than Python's automatic node discovery. No Agent struct equivalent here;
//!         nodes are individual types implementing the Node trait.
//! VALIDATES: NodeRegistry, NodeMetadata, NodeFactory, register_*_nodes
//!
//! Run: cargo run -p tutorial-core --bin 02_node_types

use std::sync::Arc;

use kailash_core::{
    node::{NodeFactory, NodeMetadata, NodeRegistry, ParamType},
    nodes::{
        control_flow::register_control_flow_nodes,
        system::{register_system_nodes, HandlerFn, HandlerNodeFactory, NoOpNodeFactory},
        transform::register_transform_nodes,
    },
    runtime::{Runtime, RuntimeConfig},
    workflow::WorkflowBuilder,
};
use kailash_value::{value_map, Value, ValueMap};

fn main() {
    // ── NodeRegistry: The Type Catalog ──
    // The registry maps type-name strings to NodeFactory implementations.
    // It is constructed at startup, then shared immutably via Arc.

    let mut registry = NodeRegistry::new();
    assert!(registry.is_empty());
    assert_eq!(registry.len(), 0);

    // ── Bulk Registration ──
    // Each module provides a register_*_nodes function that populates
    // the registry with all node types in that module.

    register_system_nodes(&mut registry); // NoOpNode, LogNode
    assert_eq!(registry.len(), 2);

    register_control_flow_nodes(&mut registry); // 8 nodes
    assert_eq!(registry.len(), 10);

    register_transform_nodes(&mut registry); // 8 nodes
    assert_eq!(registry.len(), 18);

    // ── Type Discovery ──
    // list_types() returns all registered type name strings.

    let types = registry.list_types();
    assert!(types.contains(&"NoOpNode"));
    assert!(types.contains(&"LogNode"));
    assert!(types.contains(&"SwitchNode"));
    assert!(types.contains(&"JSONTransformNode"));

    // ── NodeMetadata ──
    // Each factory exposes metadata for documentation and discovery.

    let noop_meta = registry.get_metadata("NoOpNode").expect("NoOpNode registered");
    assert_eq!(noop_meta.type_name, "NoOpNode");
    assert_eq!(noop_meta.category, "system");
    assert!(!noop_meta.description.is_empty());
    assert!(!noop_meta.version.is_empty());

    // Metadata includes parameter definitions.
    assert!(!noop_meta.output_params.is_empty());
    let data_param = &noop_meta.output_params[0];
    assert_eq!(data_param.name.as_ref(), "data");
    assert_eq!(data_param.param_type, ParamType::Any);

    // ── System Nodes ──
    // NoOpNode: pass-through, forwards "data" input to "data" output.
    // LogNode: structured logging via tracing, then forwards data.

    let log_meta = registry.get_metadata("LogNode").expect("LogNode registered");
    assert_eq!(log_meta.category, "system");
    // LogNode accepts data, message, and level inputs.
    assert_eq!(log_meta.input_params.len(), 3);

    // ── Control Flow Nodes ──
    // SwitchNode: route by condition matching against cases map
    // MergeNode: combine multiple inputs into one object
    // LoopNode: iterate over array items
    // ConditionalNode: if/else based on truthiness
    // WaitNode: delay execution
    // ParallelNode: fork to multiple branches
    // ErrorHandlerNode: try/catch fallback
    // RetryNode: retry with backoff

    let switch_meta = registry
        .get_metadata("SwitchNode")
        .expect("SwitchNode registered");
    assert_eq!(switch_meta.category, "control_flow");

    let loop_meta = registry
        .get_metadata("LoopNode")
        .expect("LoopNode registered");
    assert_eq!(loop_meta.category, "control_flow");
    // LoopNode requires "items" (Array) and optional "max_iterations" (Integer).
    assert!(loop_meta
        .input_params
        .iter()
        .any(|p| p.name.as_ref() == "items" && p.required));

    // ── Transform Nodes ──
    // JSONTransformNode: dot-notation path access
    // TextTransformNode: string manipulation
    // DataMapperNode: field mapping between objects
    // SchemaValidatorNode: basic type validation
    // FormatConverterNode: Value <-> JSON string
    // ArrayOperationsNode: array manipulation
    // StringOperationsNode: string utilities
    // MathOperationsNode: arithmetic and statistics

    let json_meta = registry
        .get_metadata("JSONTransformNode")
        .expect("JSONTransformNode registered");
    assert_eq!(json_meta.category, "transform");

    let math_meta = registry
        .get_metadata("MathOperationsNode")
        .expect("MathOperationsNode registered");
    assert_eq!(math_meta.category, "transform");

    // ── Creating Nodes from the Registry ──
    // create_node(type_name, config) calls the factory's create() method.

    let noop = registry
        .create_node("NoOpNode", ValueMap::new())
        .expect("create NoOpNode");
    assert_eq!(noop.type_name(), "NoOpNode");

    // Factory-level access: use a factory directly for more control.
    let factory = NoOpNodeFactory::new();
    let node = factory.create(ValueMap::new()).expect("factory create");
    assert_eq!(node.type_name(), "NoOpNode");

    // ── HandlerNode: Inline Custom Logic ──
    // HandlerNode wraps an arbitrary async closure as a node.
    // It is registered via HandlerNodeFactory (not register_system_nodes).

    let handler: HandlerFn = Arc::new(|inputs, _ctx| {
        Box::pin(async move {
            let greeting = inputs
                .get("name" as &str)
                .and_then(|v| v.as_str())
                .unwrap_or("World");
            let mut out = ValueMap::new();
            out.insert(
                Arc::from("message"),
                Value::from(format!("Hello, {greeting}!")),
            );
            Ok(out)
        })
    });

    registry.register(Box::new(HandlerNodeFactory::new(
        "GreeterNode",
        "Produces a greeting from a name",
        handler,
        vec![kailash_core::node::ParamDef::new(
            "name",
            ParamType::String,
            false,
        )],
        vec![kailash_core::node::ParamDef::new(
            "message",
            ParamType::String,
            false,
        )],
    )));

    assert!(registry.get_metadata("GreeterNode").is_some());
    assert_eq!(registry.len(), 19);

    // ── End-to-End: Run a Workflow with Mixed Node Types ──

    let registry = Arc::new(registry);
    let runtime = Runtime::new(RuntimeConfig::default(), Arc::clone(&registry));

    let mut builder = WorkflowBuilder::new();
    builder
        .add_node("NoOpNode", "source", ValueMap::new())
        .add_node("GreeterNode", "greeter", ValueMap::new())
        .connect("source", "data", "greeter", "name");

    let workflow = builder.build(&registry).expect("mixed-type build");
    assert_eq!(workflow.node_count(), 2);

    let result = runtime
        .execute_sync(&workflow, value_map! { "data" => "Kailash" })
        .expect("mixed-type execute");

    let greeter_out = result
        .results
        .get("greeter")
        .expect("greeter should produce output");
    assert_eq!(
        greeter_out.get("message" as &str),
        Some(&Value::from("Hello, Kailash!"))
    );

    // ── Metadata Introspection ──
    // List all registered types with their categories.

    let all_types = registry.list_types();
    assert!(all_types.len() >= 19);

    // Every registered type must have non-empty metadata.
    for type_name in &all_types {
        let meta: &NodeMetadata = registry
            .get_metadata(type_name)
            .expect("metadata must exist for registered type");
        assert!(!meta.type_name.is_empty());
        assert!(!meta.category.is_empty());
        assert!(!meta.version.is_empty());
    }

    println!("PASS: 00-core/02_node_types");
}
