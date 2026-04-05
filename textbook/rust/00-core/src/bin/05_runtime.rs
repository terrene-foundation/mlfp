// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Core / Runtime
//!
//! OBJECTIVE: Use Runtime::new(), execute() (async), and execute_sync() to run workflows.
//! LEVEL: Intermediate
//! PARITY: Equivalent -- DIV-005: Single Runtime with sync wrapper. Python has separate
//!         LocalRuntime and AsyncLocalRuntime; Rust unifies them.
//!         DIV-008: ExecutionResult includes metadata (nodes_executed, levels_executed, duration).
//! VALIDATES: Runtime, RuntimeConfig, ExecutionResult, ExecutionMetadata
//!
//! Run: cargo run -p tutorial-core --bin 05_runtime

use std::sync::Arc;
use std::time::Duration;

use kailash_core::{
    node::NodeRegistry,
    nodes::{
        control_flow::register_control_flow_nodes,
        system::register_system_nodes,
    },
    runtime::{ConditionalMode, Runtime, RuntimeConfig, ValidationMode},
    workflow::WorkflowBuilder,
};
use kailash_value::{value_map, Value, ValueMap};

// The Runtime is the execution engine. It:
//  1. Takes a validated Workflow and inputs
//  2. Executes nodes level-by-level with configurable parallelism
//  3. Returns ExecutionResult with per-node outputs and metadata

#[tokio::main]
async fn main() {
    let mut registry = NodeRegistry::new();
    register_system_nodes(&mut registry);
    register_control_flow_nodes(&mut registry);
    let registry = Arc::new(registry);

    // ── RuntimeConfig ──
    // Controls execution behavior: parallelism, timeouts, conditional
    // execution, debug mode, and more.

    let config = RuntimeConfig::default();

    // Default values:
    assert!(!config.debug);
    assert!(!config.enable_cycles);
    assert_eq!(config.conditional_execution, ConditionalMode::SkipBranches);
    assert_eq!(config.connection_validation, ValidationMode::Strict);
    assert!(config.max_concurrent_nodes >= 1); // defaults to available parallelism
    assert!(config.node_timeout.is_none());
    assert!(config.workflow_timeout.is_none());
    assert!(config.node_timeouts.is_empty());
    assert!(config.strict_input_validation);

    // ── Creating a Runtime ──
    // Runtime::new(config, registry) -- registry is shared via Arc.

    let runtime = Runtime::new(RuntimeConfig::default(), Arc::clone(&registry));

    // The runtime exposes its config and registry for inspection.
    assert!(!runtime.config().debug);
    assert!(!runtime.registry().is_empty());

    // ── Synchronous Execution: execute_sync ��─
    // Blocking wrapper around execute(). Creates a tokio runtime internally
    // if none is active, or uses block_in_place if inside one.
    // (DIV-005: Python has LocalRuntime for sync; Rust has one Runtime for both.)

    let mut builder = WorkflowBuilder::new();
    builder
        .add_node("NoOpNode", "a", ValueMap::new())
        .add_node("NoOpNode", "b", ValueMap::new())
        .connect("a", "data", "b", "data");

    let workflow = builder.build(&registry).expect("build");

    let result = runtime
        .execute_sync(&workflow, value_map! { "data" => "sync" })
        .expect("execute_sync");

    // ExecutionResult fields (DIV-008):
    assert!(!result.run_id.is_empty()); // UUID
    assert_eq!(result.metadata.nodes_executed, 2);
    assert_eq!(result.metadata.levels_executed, 2);
    assert!(result.metadata.total_duration > Duration::ZERO);

    // Per-node durations are tracked.
    assert!(result.metadata.node_durations.contains_key("a"));
    assert!(result.metadata.node_durations.contains_key("b"));

    // Per-node outputs are keyed by node ID.
    assert!(result.results.contains_key("a"));
    assert!(result.results.contains_key("b"));

    let b_output = &result.results["b"];
    assert_eq!(b_output.get("data" as &str), Some(&Value::from("sync")));

    // ── Async Execution: execute ──
    // The primary execution path. Uses tokio::spawn for level-based parallelism.

    let result = runtime
        .execute(&workflow, value_map! { "data" => "async" })
        .await
        .expect("execute async");

    assert_eq!(result.metadata.nodes_executed, 2);
    let b_output = &result.results["b"];
    assert_eq!(b_output.get("data" as &str), Some(&Value::from("async")));

    // ── Runtime with Debug Mode ──
    // Debug mode enables detailed tracing of node inputs and outputs.

    let debug_config = RuntimeConfig {
        debug: true,
        ..RuntimeConfig::default()
    };
    let debug_runtime = Runtime::new(debug_config, Arc::clone(&registry));
    assert!(debug_runtime.config().debug);

    let result = debug_runtime
        .execute_sync(&workflow, value_map! { "data" => "debug" })
        .expect("debug execute");

    assert_eq!(result.metadata.nodes_executed, 2);

    // ── Parallel Execution ──
    // Independent nodes at the same level run concurrently.
    // max_concurrent_nodes controls the semaphore.

    let parallel_config = RuntimeConfig {
        max_concurrent_nodes: 4,
        ..RuntimeConfig::default()
    };
    let parallel_runtime = Runtime::new(parallel_config, Arc::clone(&registry));
    assert_eq!(parallel_runtime.config().max_concurrent_nodes, 4);

    // Build a workflow with 3 independent sink nodes.
    let mut builder = WorkflowBuilder::new();
    builder
        .add_node("NoOpNode", "source", ValueMap::new())
        .add_node("NoOpNode", "sink_1", ValueMap::new())
        .add_node("NoOpNode", "sink_2", ValueMap::new())
        .add_node("NoOpNode", "sink_3", ValueMap::new())
        .connect("source", "data", "sink_1", "data")
        .connect("source", "data", "sink_2", "data")
        .connect("source", "data", "sink_3", "data");

    let workflow = builder.build(&registry).expect("parallel build");

    // sink_1, sink_2, sink_3 are all at level 1 -- they run in parallel.
    assert_eq!(workflow.execution_levels().len(), 2);

    let result = parallel_runtime
        .execute(&workflow, value_map! { "data" => "parallel" })
        .await
        .expect("parallel execute");

    assert_eq!(result.metadata.nodes_executed, 4);
    assert_eq!(result.metadata.levels_executed, 2);

    for sink_id in ["sink_1", "sink_2", "sink_3"] {
        let output = result.results.get(sink_id).expect("sink output");
        assert_eq!(output.get("data" as &str), Some(&Value::from("parallel")));
    }

    // ── Node Timeout ──
    // Global per-node timeout. WaitNode with a 5-second delay will fail
    // if we set a 50ms timeout.

    let timeout_config = RuntimeConfig {
        node_timeout: Some(Duration::from_millis(50)),
        ..RuntimeConfig::default()
    };
    let timeout_runtime = Runtime::new(timeout_config, Arc::clone(&registry));

    let wait_config = value_map! { "duration_ms" => 5000_i64 };
    let mut builder = WorkflowBuilder::new();
    builder.add_node("WaitNode", "slow", wait_config);

    let workflow = builder.build(&registry).expect("timeout build");

    let err = timeout_runtime
        .execute_sync(&workflow, ValueMap::new())
        .unwrap_err();

    // The error is a timeout.
    assert!(format!("{err}").contains("timed out") || format!("{err}").contains("timeout"));

    // ── Per-Node Timeout Overrides ──
    // Override the global timeout for specific nodes.

    let mut per_node_config = RuntimeConfig {
        node_timeout: Some(Duration::from_millis(50)),
        ..RuntimeConfig::default()
    };
    // Give "slow" node a generous timeout while keeping the global tight.
    per_node_config
        .node_timeouts
        .insert("slow".into(), Duration::from_secs(10));

    let per_node_runtime = Runtime::new(per_node_config, Arc::clone(&registry));

    // Use a short wait this time so the test finishes quickly.
    let wait_config = value_map! { "duration_ms" => 10_i64 };
    let mut builder = WorkflowBuilder::new();
    builder.add_node("WaitNode", "slow", wait_config);

    let workflow = builder.build(&registry).expect("per-node timeout build");

    let result = per_node_runtime
        .execute_sync(&workflow, ValueMap::new())
        .expect("per-node timeout execute");

    assert_eq!(result.metadata.nodes_executed, 1);

    // ── Multiple Executions on Same Runtime ──
    // The runtime is reusable across multiple workflow executions.

    let simple_wf = {
        let mut b = WorkflowBuilder::new();
        b.add_node("NoOpNode", "n", ValueMap::new());
        b.build(&registry).expect("simple build")
    };

    let r1 = runtime
        .execute_sync(&simple_wf, value_map! { "data" => "first" })
        .expect("first run");
    let r2 = runtime
        .execute_sync(&simple_wf, value_map! { "data" => "second" })
        .expect("second run");

    // Each run gets a unique run_id.
    assert_ne!(r1.run_id, r2.run_id);

    // ── Workflow Reuse ──
    // The same Workflow can be executed multiple times with different inputs.

    let r3 = runtime
        .execute_sync(&simple_wf, value_map! { "data" => "third" })
        .expect("third run");

    assert_ne!(r2.run_id, r3.run_id);
    let n_out = &r3.results["n"];
    assert_eq!(n_out.get("data" as &str), Some(&Value::from("third")));

    // ── RuntimeMetrics ──
    // The runtime tracks process-level metrics.

    let metrics = runtime.metrics();
    // Metrics are accessible via the runtime -- exact API depends on
    // what counters are exposed, but the Arc is always available.
    assert!(Arc::strong_count(metrics) >= 1);

    // ── Conditional Execution Mode ──
    // ConditionalMode::SkipBranches (default): skip nodes with missing required inputs.
    // ConditionalMode::EvaluateAll: execute all nodes regardless.

    let skip_config = RuntimeConfig {
        conditional_execution: ConditionalMode::SkipBranches,
        ..RuntimeConfig::default()
    };
    assert_eq!(
        skip_config.conditional_execution,
        ConditionalMode::SkipBranches
    );

    let eval_config = RuntimeConfig {
        conditional_execution: ConditionalMode::EvaluateAll,
        ..RuntimeConfig::default()
    };
    assert_eq!(
        eval_config.conditional_execution,
        ConditionalMode::EvaluateAll
    );

    println!("PASS: 00-core/05_runtime");
}
