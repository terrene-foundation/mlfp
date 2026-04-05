// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Kaizen / Structured Output
//!
//! OBJECTIVE: Define structured output schemas for agent responses.
//! LEVEL: Advanced
//! PARITY: Equivalent -- Python uses typed Signature OutputField;
//!         Rust uses response_schema module with serde-based output types.
//! VALIDATES: Signature with typed OutputField, structured response patterns,
//!            multi-field output, response schema generation
//!
//! Run: cargo run -p tutorial-kaizen --bin 06_structured_output

use kailash_kaizen::agent::{
    InputField, OutputField, SignatureBuilder,
};
use kailash_kaizen::response_schema::ResponseSchema;

fn main() {
    // ── 1. Typed output fields ──
    // OutputField types tell the LLM the expected format. The agent's
    // response is parsed to match these declared fields.
    //
    // Python equivalent:
    //   class SentimentSignature(Signature):
    //       text: str = InputField(description="Text to analyze")
    //       sentiment: str = OutputField(description="positive, negative, neutral")
    //       confidence: float = OutputField(description="Confidence 0.0 to 1.0")
    //       keywords: str = OutputField(description="Comma-separated key phrases")

    let sentiment_sig = SignatureBuilder::new("Sentiment")
        .description("Analyze the sentiment of the given text.")
        .input(InputField::new("text", "Text to analyze"))
        .output(OutputField::new("sentiment", "One of: positive, negative, neutral"))
        .output(OutputField::new("confidence", "Confidence score 0.0 to 1.0"))
        .output(OutputField::new("keywords", "Comma-separated key phrases"))
        .build();

    assert_eq!(sentiment_sig.outputs().len(), 3);
    assert_eq!(sentiment_sig.outputs()[0].name(), "sentiment");
    assert_eq!(sentiment_sig.outputs()[1].name(), "confidence");
    assert_eq!(sentiment_sig.outputs()[2].name(), "keywords");

    // ── 2. Multi-field structured output ──
    // Signatures can request complex structured responses.

    let entities_sig = SignatureBuilder::new("ExtractEntities")
        .description("Extract named entities from text.")
        .input(InputField::new("text", "Text to process"))
        .input(InputField::new("context", "Domain context").with_default("general"))
        .output(OutputField::new("people", "JSON array of person names"))
        .output(OutputField::new("organizations", "JSON array of org names"))
        .output(OutputField::new("locations", "JSON array of locations"))
        .output(OutputField::new("dates", "JSON array of dates mentioned"))
        .output(OutputField::new("summary", "One-sentence summary"))
        .build();

    assert_eq!(entities_sig.outputs().len(), 5);
    assert_eq!(entities_sig.inputs().len(), 2);

    // ── 3. Boolean/decision output ──

    let moderation_sig = SignatureBuilder::new("Moderation")
        .description("Check if content violates guidelines.")
        .input(InputField::new("content", "Content to moderate"))
        .input(InputField::new("guidelines", "Moderation guidelines"))
        .output(OutputField::new("is_safe", "True if content is safe"))
        .output(OutputField::new("reason", "Explanation of decision"))
        .output(OutputField::new("category", "Violation category if unsafe, else 'none'"))
        .build();

    assert_eq!(moderation_sig.outputs()[0].name(), "is_safe");

    // ── 4. Invoice extraction -- structured data ──

    let invoice_sig = SignatureBuilder::new("InvoiceExtraction")
        .description("Extract structured data from an invoice.")
        .input(InputField::new("invoice_text", "Raw invoice text"))
        .output(OutputField::new("vendor_name", "Name of the vendor"))
        .output(OutputField::new("invoice_number", "Invoice number"))
        .output(OutputField::new("total_amount", "Total as string (e.g., '1234.56')"))
        .output(OutputField::new("currency", "Currency code (e.g., 'SGD', 'USD')"))
        .output(OutputField::new(
            "line_items",
            "JSON array of {description, quantity, price}",
        ))
        .build();

    assert_eq!(invoice_sig.outputs().len(), 5);

    // ── 5. ResponseSchema generation ──
    // ResponseSchema converts a Signature's outputs into a JSON Schema
    // that the LLM provider uses for structured output enforcement.

    let schema = ResponseSchema::from_signature(&sentiment_sig);

    assert_eq!(schema.field_count(), 3);
    assert!(schema.has_field("sentiment"));
    assert!(schema.has_field("confidence"));
    assert!(schema.has_field("keywords"));

    // Schema can be serialized to JSON for the LLM API
    let json_schema = schema.to_json_schema();
    assert_eq!(json_schema["type"], "object");
    assert!(json_schema["properties"]["sentiment"].is_object());

    // ── 6. Pattern: Signature -> Agent -> Structured response ──
    // In practice, a Signature is attached to an agent, and the agent's
    // LLM call is constrained to produce the declared output fields:
    //
    //   let agent = Agent::new(sentiment_sig, &model);
    //   let result = agent.run(inputs).await;
    //   println!("Sentiment: {}", result["sentiment"]);
    //   println!("Confidence: {}", result["confidence"]);

    // ── 7. Key concepts ──
    // - OutputField: declares expected output with name and description
    // - Multi-field signatures constrain LLM to structured responses
    // - ResponseSchema: converts outputs to JSON Schema for LLM APIs
    // - Rust uses SignatureBuilder; Python uses class-based Signature
    // - Both SDKs enforce structured output via provider-specific mechanisms

    println!("PASS: 03-kaizen/06_structured_output");
}
