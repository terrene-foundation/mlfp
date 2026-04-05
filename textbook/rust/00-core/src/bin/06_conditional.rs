// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Core / Conditional Logic
//!
//! OBJECTIVE: Use ConditionalNode and SwitchNode for branching and routing in workflows.
//! LEVEL: Intermediate
//! PARITY: Equivalent -- DIV-017: SwitchNode cases are configured via the "cases" key
//!         in the node config (a ValueMap), not as runtime inputs.
//!         DIV-018: SwitchNode default_branch is also set in node config.
//! VALIDATES: ConditionalNode, SwitchNode, MergeNode, branching patterns
//!
//! Run: cargo run -p tutorial-core --bin 06_conditional

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

    // ── ConditionalNode: If/Else ──
    // Evaluates "condition" for truthiness, returns "if_value" or "else_value".
    //
    // Truthiness rules (Value::is_truthy):
    //   Falsy: Null, Bool(false), Integer(0), Float(0.0), empty String/Bytes/Array/Object
    //   Truthy: everything else

    let mut builder = WorkflowBuilder::new();
    builder.add_node("ConditionalNode", "cond", ValueMap::new());

    let workflow = builder.build(&registry).expect("conditional build");

    // Truthy condition: Bool(true) -> returns if_value.
    let result = runtime
        .execute_sync(
            &workflow,
            value_map! {
                "condition" => true,
                "if_value" => "yes",
                "else_value" => "no"
            },
        )
        .expect("truthy execute");

    let cond_out = &result.results["cond"];
    assert_eq!(cond_out.get("result" as &str), Some(&Value::from("yes")));

    // Falsy condition: Bool(false) -> returns else_value.
    let result = runtime
        .execute_sync(
            &workflow,
            value_map! {
                "condition" => false,
                "if_value" => "yes",
                "else_value" => "no"
            },
        )
        .expect("falsy execute");

    let cond_out = &result.results["cond"];
    assert_eq!(cond_out.get("result" as &str), Some(&Value::from("no")));

    // Integer truthiness: 0 is falsy, non-zero is truthy.
    let result = runtime
        .execute_sync(
            &workflow,
            value_map! {
                "condition" => 0_i64,
                "if_value" => "nonzero",
                "else_value" => "zero"
            },
        )
        .expect("integer-falsy execute");

    let cond_out = &result.results["cond"];
    assert_eq!(cond_out.get("result" as &str), Some(&Value::from("zero")));

    let result = runtime
        .execute_sync(
            &workflow,
            value_map! {
                "condition" => 42_i64,
                "if_value" => "nonzero",
                "else_value" => "zero"
            },
        )
        .expect("integer-truthy execute");

    let cond_out = &result.results["cond"];
    assert_eq!(
        cond_out.get("result" as &str),
        Some(&Value::from("nonzero"))
    );

    // String truthiness: empty is falsy, non-empty is truthy.
    let result = runtime
        .execute_sync(
            &workflow,
            value_map! {
                "condition" => "",
                "if_value" => "has_text",
                "else_value" => "empty"
            },
        )
        .expect("string-falsy execute");

    let cond_out = &result.results["cond"];
    assert_eq!(cond_out.get("result" as &str), Some(&Value::from("empty")));

    // Missing if_value/else_value returns Null.
    let result = runtime
        .execute_sync(
            &workflow,
            value_map! { "condition" => true },
        )
        .expect("missing-value execute");

    let cond_out = &result.results["cond"];
    assert_eq!(cond_out.get("result" as &str), Some(&Value::Null));

    // ── SwitchNode: Multi-Way Routing ��─
    // Matches "condition" input against configured "cases" map.
    // Cases are set in node config (DIV-017), not as runtime inputs.
    //
    // Config keys:
    //   "cases": Object mapping condition-value-strings to branch names
    //   "default_branch": optional fallback branch name (DIV-018)

    let switch_config = value_map! {
        "cases" => Value::Object({
            let mut m = std::collections::BTreeMap::new();
            m.insert(Arc::from("low"), Value::from("route_low"));
            m.insert(Arc::from("medium"), Value::from("route_medium"));
            m.insert(Arc::from("high"), Value::from("route_high"));
            m
        }),
        "default_branch" => "route_unknown"
    };

    let mut builder = WorkflowBuilder::new();
    builder.add_node("SwitchNode", "router", switch_config);

    let workflow = builder.build(&registry).expect("switch build");

    // Match "medium" -> "route_medium"
    let result = runtime
        .execute_sync(
            &workflow,
            value_map! {
                "condition" => "medium",
                "data" => "payload"
            },
        )
        .expect("switch-medium execute");

    let router_out = &result.results["router"];
    assert_eq!(
        router_out.get("matched" as &str),
        Some(&Value::from("route_medium"))
    );
    // Data is forwarded to the output.
    assert_eq!(
        router_out.get("data" as &str),
        Some(&Value::from("payload"))
    );

    // Match "high" -> "route_high"
    let result = runtime
        .execute_sync(
            &workflow,
            value_map! { "condition" => "high" },
        )
        .expect("switch-high execute");

    assert_eq!(
        result.results["router"].get("matched" as &str),
        Some(&Value::from("route_high"))
    );

    // No match -> default_branch -> "route_unknown"
    let result = runtime
        .execute_sync(
            &workflow,
            value_map! { "condition" => "extreme" },
        )
        .expect("switch-default execute");

    assert_eq!(
        result.results["router"].get("matched" as &str),
        Some(&Value::from("route_unknown"))
    );

    // ── SwitchNode: Integer Conditions ──
    // Non-string conditions are converted via Display before matching.

    let switch_config = value_map! {
        "cases" => Value::Object({
            let mut m = std::collections::BTreeMap::new();
            m.insert(Arc::from("1"), Value::from("one"));
            m.insert(Arc::from("2"), Value::from("two"));
            m.insert(Arc::from("3"), Value::from("three"));
            m
        }),
        "default_branch" => "other"
    };

    let mut builder = WorkflowBuilder::new();
    builder.add_node("SwitchNode", "num_router", switch_config);

    let workflow = builder.build(&registry).expect("int-switch build");

    let result = runtime
        .execute_sync(&workflow, value_map! { "condition" => 2_i64 })
        .expect("int-switch execute");

    assert_eq!(
        result.results["num_router"].get("matched" as &str),
        Some(&Value::from("two"))
    );

    // ── SwitchNode Without Default: Error on No Match ──

    let switch_config = value_map! {
        "cases" => Value::Object({
            let mut m = std::collections::BTreeMap::new();
            m.insert(Arc::from("yes"), Value::from("affirmative"));
            m
        })
        // No "default_branch" key.
    };

    let mut builder = WorkflowBuilder::new();
    builder.add_node("SwitchNode", "strict_router", switch_config);

    let workflow = builder.build(&registry).expect("no-default switch build");

    // "yes" matches fine.
    let result = runtime
        .execute_sync(&workflow, value_map! { "condition" => "yes" })
        .expect("switch-yes execute");

    assert_eq!(
        result.results["strict_router"].get("matched" as &str),
        Some(&Value::from("affirmative"))
    );

    // "no" does not match and there is no default -> error.
    let err = runtime
        .execute_sync(&workflow, value_map! { "condition" => "no" })
        .unwrap_err();

    // Use Debug format to see through RuntimeError::NodeFailed wrapper.
    let err_debug = format!("{err:?}");
    assert!(
        err_debug.contains("no matching case"),
        "unexpected error: {err_debug}"
    );

    // ── LoopNode: Array Iteration ──
    // LoopNode iterates over "items" and collects results.

    let mut builder = WorkflowBuilder::new();
    builder.add_node("LoopNode", "looper", ValueMap::new());

    let workflow = builder.build(&registry).expect("loop build");

    let items = Value::Array(vec![
        Value::from("a"),
        Value::from("b"),
        Value::from("c"),
    ]);

    let result = runtime
        .execute_sync(
            &workflow,
            {
                let mut m = ValueMap::new();
                m.insert(Arc::from("items"), items);
                m
            },
        )
        .expect("loop execute");

    let loop_out = &result.results["looper"];
    assert_eq!(loop_out.get("count" as &str), Some(&Value::Integer(3)));
    let results = loop_out
        .get("results" as &str)
        .expect("results key")
        .as_array()
        .expect("results is array");
    assert_eq!(results.len(), 3);

    // ── LoopNode with max_iterations ──

    let items = Value::Array((0..100).map(Value::from).collect());

    let result = runtime
        .execute_sync(
            &workflow,
            {
                let mut m = ValueMap::new();
                m.insert(Arc::from("items"), items);
                m.insert(Arc::from("max_iterations"), Value::Integer(5));
                m
            },
        )
        .expect("loop-limited execute");

    let loop_out = &result.results["looper"];
    assert_eq!(loop_out.get("count" as &str), Some(&Value::Integer(5)));

    // ── Combined Pattern: Conditional + SwitchNode ──
    // Use ConditionalNode to decide, then SwitchNode to route.
    //
    // Important: when a node has incoming connections, the runtime merges
    // the node's build-time config into the input map. Strict input
    // validation then rejects config keys (like "cases") that are not
    // declared as input parameters. Since SwitchNode reads "cases" at
    // factory creation time (not at execution time), we disable strict
    // validation for this combined workflow.

    let relaxed_config = RuntimeConfig {
        strict_input_validation: false,
        ..RuntimeConfig::default()
    };
    let relaxed_runtime = Runtime::new(relaxed_config, Arc::clone(&registry));

    let switch_config = value_map! {
        "cases" => Value::Object({
            let mut m = std::collections::BTreeMap::new();
            m.insert(Arc::from("process"), Value::from("processing_branch"));
            m.insert(Arc::from("skip"), Value::from("skip_branch"));
            m
        }),
        "default_branch" => "error_branch"
    };

    let mut builder = WorkflowBuilder::new();
    builder
        .add_node("ConditionalNode", "gate", ValueMap::new())
        .add_node("SwitchNode", "router", switch_config)
        .connect("gate", "result", "router", "condition");

    let workflow = builder.build(&registry).expect("combined build");

    // Truthy condition -> if_value "process" -> SwitchNode matches "process".
    let result = relaxed_runtime
        .execute_sync(
            &workflow,
            value_map! {
                "condition" => true,
                "if_value" => "process",
                "else_value" => "skip"
            },
        )
        .expect("combined-truthy execute");

    assert_eq!(
        result.results["router"].get("matched" as &str),
        Some(&Value::from("processing_branch"))
    );

    // Falsy condition -> else_value "skip" -> SwitchNode matches "skip".
    let result = relaxed_runtime
        .execute_sync(
            &workflow,
            value_map! {
                "condition" => false,
                "if_value" => "process",
                "else_value" => "skip"
            },
        )
        .expect("combined-falsy execute");

    assert_eq!(
        result.results["router"].get("matched" as &str),
        Some(&Value::from("skip_branch"))
    );

    println!("PASS: 00-core/06_conditional");
}
