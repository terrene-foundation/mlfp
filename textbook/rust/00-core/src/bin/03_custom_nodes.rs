// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Core / Custom Nodes
//!
//! OBJECTIVE: Implement the Node trait for custom types and register them via NodeFactory.
//! LEVEL: Intermediate
//! PARITY: Rust-only -- DIV-014: Rust has no from_function/from_class convenience helpers.
//!         Custom nodes must implement the Node trait directly and provide a NodeFactory.
//! VALIDATES: Node trait, NodeFactory trait, NodeMetadata, ParamDef, ParamType
//!
//! Run: cargo run -p tutorial-core --bin 03_custom_nodes

use std::{future::Future, pin::Pin, sync::Arc};

use kailash_core::{
    error::NodeError,
    node::{ExecutionContext, Node, NodeFactory, NodeMetadata, NodeRegistry, ParamDef, ParamType},
    runtime::{Runtime, RuntimeConfig},
    workflow::WorkflowBuilder,
};
use kailash_value::{Value, ValueMap};

// ── Custom Node: DoubleNode ──
// Doubles a numeric input. Demonstrates the full Node trait implementation.

struct DoubleNode {
    input_params: Vec<ParamDef>,
    output_params: Vec<ParamDef>,
}

impl DoubleNode {
    fn new() -> Self {
        Self {
            input_params: vec![ParamDef::new("value", ParamType::Any, true)
                .with_description("A number to double (Integer or Float)")],
            output_params: vec![ParamDef::new("result", ParamType::Any, false)
                .with_description("The doubled value")],
        }
    }
}

impl Node for DoubleNode {
    fn type_name(&self) -> &str {
        "DoubleNode"
    }

    fn input_params(&self) -> &[ParamDef] {
        &self.input_params
    }

    fn output_params(&self) -> &[ParamDef] {
        &self.output_params
    }

    fn execute(
        &self,
        inputs: ValueMap,
        _ctx: &ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = Result<ValueMap, NodeError>> + Send + '_>> {
        Box::pin(async move {
            let value = inputs
                .get("value" as &str)
                .ok_or_else(|| NodeError::MissingInput {
                    name: "value".into(),
                })?;

            let doubled = match value {
                Value::Integer(i) => Value::Integer(i * 2),
                Value::Float(f) => Value::Float(f * 2.0),
                other => {
                    return Err(NodeError::InvalidInput {
                        name: "value".into(),
                        expected: "integer or float".into(),
                        got: format!("{other}"),
                    });
                }
            };

            let mut output = ValueMap::new();
            output.insert(Arc::from("result"), doubled);
            Ok(output)
        })
    }
}

// ── Custom NodeFactory: DoubleNodeFactory ──
// Every custom node needs a corresponding factory for registry integration.

struct DoubleNodeFactory {
    metadata: NodeMetadata,
}

impl DoubleNodeFactory {
    fn new() -> Self {
        Self {
            metadata: NodeMetadata {
                type_name: "DoubleNode".into(),
                description: "Doubles a numeric input".into(),
                category: "math".into(),
                input_params: vec![ParamDef::new("value", ParamType::Any, true)],
                output_params: vec![ParamDef::new("result", ParamType::Any, false)],
                version: "1.0.0".into(),
                author: "Tutorial".into(),
                tags: vec!["math".into(), "transform".into()],
            },
        }
    }
}

impl NodeFactory for DoubleNodeFactory {
    fn create(&self, _config: ValueMap) -> Result<Box<dyn Node>, NodeError> {
        Ok(Box::new(DoubleNode::new()))
    }

    fn metadata(&self) -> &NodeMetadata {
        &self.metadata
    }
}

// ── Custom Node: ConcatNode ──
// Concatenates two string inputs. Demonstrates multi-input nodes.

struct ConcatNode {
    input_params: Vec<ParamDef>,
    output_params: Vec<ParamDef>,
    separator: String,
}

impl ConcatNode {
    fn new(separator: String) -> Self {
        Self {
            input_params: vec![
                ParamDef::new("left", ParamType::String, true)
                    .with_description("First string"),
                ParamDef::new("right", ParamType::String, true)
                    .with_description("Second string"),
            ],
            output_params: vec![ParamDef::new("result", ParamType::String, false)
                .with_description("Concatenated result")],
            separator,
        }
    }
}

impl Node for ConcatNode {
    fn type_name(&self) -> &str {
        "ConcatNode"
    }

    fn input_params(&self) -> &[ParamDef] {
        &self.input_params
    }

    fn output_params(&self) -> &[ParamDef] {
        &self.output_params
    }

    fn execute(
        &self,
        inputs: ValueMap,
        _ctx: &ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = Result<ValueMap, NodeError>> + Send + '_>> {
        // Capture the separator before the async block.
        let sep = self.separator.clone();
        Box::pin(async move {
            let left = inputs
                .get("left" as &str)
                .and_then(|v| v.as_str())
                .ok_or_else(|| NodeError::MissingInput {
                    name: "left".into(),
                })?;

            let right = inputs
                .get("right" as &str)
                .and_then(|v| v.as_str())
                .ok_or_else(|| NodeError::MissingInput {
                    name: "right".into(),
                })?;

            let result = format!("{left}{sep}{right}");
            let mut output = ValueMap::new();
            output.insert(Arc::from("result"), Value::from(result.as_str()));
            Ok(output)
        })
    }
}

// ── ConcatNodeFactory: Configurable via ValueMap ──
// The factory reads a "separator" key from the config to customize the node.

struct ConcatNodeFactory {
    metadata: NodeMetadata,
}

impl ConcatNodeFactory {
    fn new() -> Self {
        Self {
            metadata: NodeMetadata {
                type_name: "ConcatNode".into(),
                description: "Concatenates two strings with a configurable separator".into(),
                category: "text".into(),
                input_params: vec![
                    ParamDef::new("left", ParamType::String, true),
                    ParamDef::new("right", ParamType::String, true),
                ],
                output_params: vec![ParamDef::new("result", ParamType::String, false)],
                version: "1.0.0".into(),
                author: "Tutorial".into(),
                tags: vec!["text".into(), "concat".into()],
            },
        }
    }
}

impl NodeFactory for ConcatNodeFactory {
    fn create(&self, config: ValueMap) -> Result<Box<dyn Node>, NodeError> {
        // Read the separator from config, defaulting to empty string.
        let separator = config
            .get("separator" as &str)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Ok(Box::new(ConcatNode::new(separator)))
    }

    fn metadata(&self) -> &NodeMetadata {
        &self.metadata
    }
}

fn main() {
    // ── Register Custom Nodes ──

    let mut registry = NodeRegistry::new();
    kailash_core::nodes::system::register_system_nodes(&mut registry);
    registry.register(Box::new(DoubleNodeFactory::new()));
    registry.register(Box::new(ConcatNodeFactory::new()));

    assert!(registry.get_metadata("DoubleNode").is_some());
    assert!(registry.get_metadata("ConcatNode").is_some());

    // ── Metadata Verification ──

    let double_meta = registry.get_metadata("DoubleNode").expect("DoubleNode");
    assert_eq!(double_meta.category, "math");
    assert_eq!(double_meta.version, "1.0.0");
    assert_eq!(double_meta.input_params.len(), 1);
    assert_eq!(double_meta.output_params.len(), 1);

    // ── Create Nodes from Registry ──

    let double = registry
        .create_node("DoubleNode", ValueMap::new())
        .expect("create DoubleNode");
    assert_eq!(double.type_name(), "DoubleNode");

    // ConcatNode with custom separator via config.
    let config = {
        let mut m = ValueMap::new();
        m.insert(Arc::from("separator"), Value::from(" "));
        m
    };
    let concat = registry
        .create_node("ConcatNode", config)
        .expect("create ConcatNode with config");
    assert_eq!(concat.type_name(), "ConcatNode");

    // ── Run DoubleNode in a Workflow ──

    let registry = Arc::new(registry);
    let runtime = Runtime::new(RuntimeConfig::default(), Arc::clone(&registry));

    let mut builder = WorkflowBuilder::new();
    builder
        .add_node("NoOpNode", "input", ValueMap::new())
        .add_node("DoubleNode", "doubler", ValueMap::new())
        .connect("input", "data", "doubler", "value");

    let workflow = builder.build(&registry).expect("double workflow build");

    let result = runtime
        .execute_sync(
            &workflow,
            {
                let mut m = ValueMap::new();
                m.insert(Arc::from("data"), Value::Integer(21));
                m
            },
        )
        .expect("double workflow execute");

    let doubled = result
        .results
        .get("doubler")
        .expect("doubler output")
        .get("result" as &str)
        .expect("result key");
    assert_eq!(doubled, &Value::Integer(42));

    // Float doubling.
    let result = runtime
        .execute_sync(
            &workflow,
            {
                let mut m = ValueMap::new();
                m.insert(Arc::from("data"), Value::Float(1.5));
                m
            },
        )
        .expect("float double execute");

    let doubled = result
        .results
        .get("doubler")
        .expect("doubler output")
        .get("result" as &str)
        .expect("result key");
    assert_eq!(doubled, &Value::Float(3.0));

    // ── Run ConcatNode with Config ──

    let mut builder = WorkflowBuilder::new();
    let concat_config = {
        let mut m = ValueMap::new();
        m.insert(Arc::from("separator"), Value::from(", "));
        m
    };
    builder
        .add_node("NoOpNode", "left_src", ValueMap::new())
        .add_node("NoOpNode", "right_src", ValueMap::new())
        .add_node("ConcatNode", "concat", concat_config)
        .connect("left_src", "data", "concat", "left")
        .connect("right_src", "data", "concat", "right");

    let workflow = builder.build(&registry).expect("concat workflow build");

    // We need separate input paths. NoOpNode forwards "data" -> "data",
    // so left_src receives "data" from workflow inputs. For right_src,
    // we also pass "data". Both source nodes receive the same workflow inputs,
    // so we use a single NoOpNode feeding into both concat inputs instead.

    let mut builder = WorkflowBuilder::new();
    let concat_config = {
        let mut m = ValueMap::new();
        m.insert(Arc::from("separator"), Value::from(" + "));
        m
    };
    builder
        .add_node("ConcatNode", "concat", concat_config);

    let workflow = builder.build(&registry).expect("standalone concat build");

    let result = runtime
        .execute_sync(
            &workflow,
            {
                let mut m = ValueMap::new();
                m.insert(Arc::from("left"), Value::from("alpha"));
                m.insert(Arc::from("right"), Value::from("beta"));
                m
            },
        )
        .expect("concat execute");

    let concatenated = result
        .results
        .get("concat")
        .expect("concat output")
        .get("result" as &str)
        .expect("result key");
    assert_eq!(concatenated, &Value::from("alpha + beta"));

    // ── Error Handling in Custom Nodes ──
    // DoubleNode rejects non-numeric input with InvalidInput.

    let mut builder = WorkflowBuilder::new();
    builder.add_node("DoubleNode", "doubler", ValueMap::new());

    let workflow = builder.build(&registry).expect("error test build");

    let err = runtime
        .execute_sync(
            &workflow,
            {
                let mut m = ValueMap::new();
                m.insert(Arc::from("value"), Value::from("not a number"));
                m
            },
        )
        .unwrap_err();

    // The runtime wraps node errors in RuntimeError::NodeFailed.
    assert!(format!("{err}").contains("invalid input"));

    // ── Object Safety ──
    // Box<dyn Node> is valid -- the trait is object-safe.

    let node: Box<dyn Node> = Box::new(DoubleNode::new());
    assert_eq!(node.type_name(), "DoubleNode");
    assert_eq!(node.input_params().len(), 1);
    assert_eq!(node.output_params().len(), 1);

    println!("PASS: 00-core/03_custom_nodes");
}
