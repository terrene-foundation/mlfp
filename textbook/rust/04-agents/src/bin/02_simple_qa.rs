// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Agents / Simple QA Agent
//!
//! OBJECTIVE: Build a single-shot question-answering agent with Signature.
//! LEVEL: Basic
//! PARITY: Equivalent -- Python uses SimpleQA agent class;
//!         Rust uses AgentConfig with a QA signature for single-shot execution.
//! VALIDATES: Single-shot agent pattern, QA signature, response structure
//!
//! Run: cargo run -p tutorial-agents --bin 02_simple_qa

use kailash_kaizen::agent::{
    AgentConfig, InputField, OutputField, SignatureBuilder,
};

fn main() {
    // ── 1. SimpleQA pattern ──
    // The simplest agent pattern: one input, one LLM call, one output.
    // No tools, no iteration, no memory.
    //
    // This is the "hello world" of agents -- useful for:
    //   - Classification tasks
    //   - Translation
    //   - Summarization
    //   - Simple Q&A

    let qa_sig = SignatureBuilder::new("SimpleQA")
        .description("You are a helpful question-answering assistant.")
        .input(InputField::new("question", "The question to answer"))
        .input(InputField::new("context", "Additional context").with_default(""))
        .output(OutputField::new("answer", "Clear, accurate answer"))
        .build();

    assert_eq!(qa_sig.name(), "SimpleQA");
    assert_eq!(qa_sig.inputs().len(), 2);
    assert_eq!(qa_sig.outputs().len(), 1);

    // ── 2. Agent configuration ──
    // AgentConfig pairs a signature with model and execution settings.

    let config = AgentConfig::new("simple-qa", "claude-sonnet-4-20250514")
        .max_turns(1) // Single-shot: exactly one LLM call
        .temperature(0.3); // Lower temperature for factual responses

    assert_eq!(config.name(), "simple-qa");
    assert_eq!(config.max_turns(), 1);

    // ── 3. Multi-output QA ──
    // Even simple agents can produce structured multi-field output.

    let detailed_sig = SignatureBuilder::new("DetailedQA")
        .description("Answer questions with confidence and sources.")
        .input(InputField::new("question", "The question to answer"))
        .output(OutputField::new("answer", "Clear, accurate answer"))
        .output(OutputField::new("confidence", "Confidence score 0.0 to 1.0"))
        .output(OutputField::new("sources", "List of sources used"))
        .build();

    assert_eq!(detailed_sig.outputs().len(), 3);

    // ── 4. Classification agent ──
    // Same single-shot pattern, different signature.

    let classifier_sig = SignatureBuilder::new("Classifier")
        .description("Classify the input text into a category.")
        .input(InputField::new("text", "Text to classify"))
        .input(InputField::new("categories", "Available categories"))
        .output(OutputField::new("category", "Most appropriate category"))
        .output(OutputField::new("confidence", "Confidence score"))
        .output(OutputField::new("reasoning", "Why this category"))
        .build();

    assert_eq!(classifier_sig.inputs().len(), 2);
    assert_eq!(classifier_sig.outputs().len(), 3);

    // ── 5. Translation agent ──

    let translator_sig = SignatureBuilder::new("Translator")
        .description("Translate text between languages.")
        .input(InputField::new("text", "Text to translate"))
        .input(InputField::new("source_language", "Source language"))
        .input(InputField::new("target_language", "Target language"))
        .output(OutputField::new("translation", "Translated text"))
        .output(OutputField::new("notes", "Translation notes or ambiguities"))
        .build();

    assert_eq!(translator_sig.inputs().len(), 3);

    // ── 6. Execution pattern ──
    // In practice, a SimpleQA agent runs synchronously:
    //
    //   let agent = SimpleQAAgent::new(qa_sig, config);
    //   let result = agent.run(inputs! {
    //       "question" => "What is Kailash SDK?",
    //       "context" => "Open-source workflow orchestration"
    //   }).await;
    //   println!("Answer: {}", result["answer"]);
    //
    // NOTE: We do not call run() here as it requires an LLM API key.

    // ── 7. Key concepts ──
    // - SimpleQA: single-shot, no tools, no memory
    // - max_turns(1) enforces exactly one LLM call
    // - Lower temperature (0.3) for factual responses
    // - Same pattern works for QA, classification, translation
    // - Signature declares what the agent does; config controls how

    println!("PASS: 04-agents/02_simple_qa");
}
