// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Kaizen / Signatures
//!
//! OBJECTIVE: Define AI agent interfaces using Signature, InputField, OutputField.
//! LEVEL: Basic
//! PARITY: Equivalent -- Python uses class-based Signature with metaclass (DSPy-inspired);
//!         Rust uses SignatureBuilder with typed InputField/OutputField structs.
//! VALIDATES: Signature, SignatureBuilder, InputField, OutputField, field metadata
//!
//! Run: cargo run -p tutorial-kaizen --bin 01_signatures

use kailash_kaizen::agent::{
    InputField, OutputField, Signature, SignatureBuilder,
};

fn main() {
    // ── 1. Build a Signature (recommended) ──
    // Signatures declare WHAT an agent does -- its inputs and outputs.
    // The description becomes the agent's system instructions.
    // InputField/OutputField carry a description and optional default.
    //
    // Python equivalent:
    //   class QASignature(Signature):
    //       """You are a helpful question-answering assistant."""
    //       question: str = InputField(description="The question to answer")
    //       context: str = InputField(description="Additional context", default="")
    //       answer: str = OutputField(description="Clear, accurate answer")
    //       confidence: float = OutputField(description="Confidence score 0.0 to 1.0")

    let qa_sig = SignatureBuilder::new("QASignature")
        .description("You are a helpful question-answering assistant.")
        .input(InputField::new("question", "The question to answer"))
        .input(InputField::new("context", "Additional context").with_default(""))
        .output(OutputField::new("answer", "Clear, accurate answer"))
        .output(OutputField::new("confidence", "Confidence score 0.0 to 1.0"))
        .build();

    assert_eq!(qa_sig.name(), "QASignature");
    assert_eq!(qa_sig.inputs().len(), 2);
    assert_eq!(qa_sig.outputs().len(), 2);

    // Check input field properties
    let question = &qa_sig.inputs()[0];
    assert_eq!(question.name(), "question");
    assert!(question.is_required()); // No default -> required

    let context = &qa_sig.inputs()[1];
    assert_eq!(context.name(), "context");
    assert!(!context.is_required()); // Has default -> optional

    // ── 2. Signature with intent and guidelines ──
    // intent defines WHY the agent exists (high-level purpose).
    // guidelines define HOW the agent should behave (constraints).

    let support_sig = SignatureBuilder::new("CustomerSupport")
        .description("You are a customer support agent for an e-commerce platform.")
        .intent("Resolve customer issues efficiently and empathetically")
        .guideline("Acknowledge concerns before proposing solutions")
        .guideline("Use empathetic language")
        .guideline("Escalate if unresolved in 3 turns")
        .input(InputField::new("customer_message", "Customer's message"))
        .input(InputField::new("order_history", "Recent order history").with_default(""))
        .output(OutputField::new("response", "Support response"))
        .output(OutputField::new("action", "Action: resolve, escalate, transfer"))
        .build();

    assert_eq!(
        support_sig.intent().unwrap(),
        "Resolve customer issues efficiently and empathetically"
    );
    assert_eq!(support_sig.guidelines().len(), 3);

    // ── 3. Programmatic Signature (dynamic) ──
    // For use cases where inputs/outputs aren't known at compile time.

    let sig = Signature::quick(
        "search",
        "Search for relevant information",
        &["query", "context"],
        &["result"],
    );

    assert_eq!(sig.name(), "search");
    assert_eq!(sig.inputs().len(), 2);
    assert_eq!(sig.outputs().len(), 1);

    // ── 4. Empty signature ──
    // Valid but unusual -- no fields.

    let empty = SignatureBuilder::new("Empty")
        .description("A signature with no fields.")
        .build();

    assert_eq!(empty.inputs().len(), 0);
    assert_eq!(empty.outputs().len(), 0);

    // ── 5. Field metadata ──
    // Additional metadata can be attached to fields via key-value pairs.

    let rich_sig = SignatureBuilder::new("RichSignature")
        .description("Signature with rich field metadata.")
        .input(
            InputField::new("data", "Input data")
                .with_metadata("format", "json")
                .with_metadata("max_length", "10000"),
        )
        .output(OutputField::new("result", "Processed result"))
        .build();

    let data_field = &rich_sig.inputs()[0];
    assert_eq!(data_field.metadata("format"), Some("json"));
    assert_eq!(data_field.metadata("max_length"), Some("10000"));

    // ── 6. Key concepts ──
    // - Signature: declares agent interface (inputs + outputs)
    // - SignatureBuilder: builder pattern for constructing signatures
    // - InputField: input parameter with name, description, optional default
    // - OutputField: output parameter with name and description
    // - intent: why the agent exists (high-level purpose)
    // - guidelines: constraints on agent behavior
    // - metadata: arbitrary key-value pairs on fields
    // - Python uses class-based metaclass pattern; Rust uses builder pattern

    println!("PASS: 03-kaizen/01_signatures");
}
