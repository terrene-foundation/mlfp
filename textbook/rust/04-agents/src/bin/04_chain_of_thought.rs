// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Agents / Chain of Thought
//!
//! OBJECTIVE: Build agents that use explicit chain-of-thought reasoning.
//! LEVEL: Intermediate
//! PARITY: Equivalent -- Both SDKs support CoT via signature design.
//!         Python uses class-based Signature; Rust uses SignatureBuilder.
//! VALIDATES: CoT signature pattern, reasoning traces, step-by-step output
//!
//! Run: cargo run -p tutorial-agents --bin 04_chain_of_thought

use kailash_kaizen::agent::{
    InputField, OutputField, SignatureBuilder,
};

fn main() {
    // ── 1. Chain of Thought pattern ──
    // CoT forces the LLM to show its reasoning before answering.
    // The key insight: adding a "reasoning" output field before the
    // "answer" field causes the model to think step-by-step.
    //
    // This improves accuracy on:
    //   - Math problems
    //   - Logical deductions
    //   - Multi-step analysis
    //   - Complex classification

    let cot_sig = SignatureBuilder::new("ChainOfThought")
        .description("Think step by step before answering the question.")
        .input(InputField::new("question", "The question to answer"))
        .output(OutputField::new("reasoning", "Step-by-step reasoning process"))
        .output(OutputField::new("answer", "Final answer based on reasoning"))
        .build();

    assert_eq!(cot_sig.outputs().len(), 2);
    // reasoning comes BEFORE answer -- this is critical for CoT
    assert_eq!(cot_sig.outputs()[0].name(), "reasoning");
    assert_eq!(cot_sig.outputs()[1].name(), "answer");

    // ── 2. CoT with confidence ──
    // Adding confidence after reasoning lets the model self-evaluate.

    let cot_conf_sig = SignatureBuilder::new("ChainOfThoughtWithConfidence")
        .description("Think step by step. Evaluate your confidence.")
        .input(InputField::new("question", "The question to answer"))
        .input(InputField::new("context", "Supporting context").with_default(""))
        .output(OutputField::new("reasoning", "Step-by-step thinking"))
        .output(OutputField::new("answer", "Final answer"))
        .output(OutputField::new("confidence", "Self-assessed confidence 0.0 to 1.0"))
        .build();

    assert_eq!(cot_conf_sig.outputs().len(), 3);

    // ── 3. Math reasoning agent ──
    // CoT is particularly effective for mathematical reasoning.

    let math_sig = SignatureBuilder::new("MathReasoning")
        .description(
            "Solve the math problem step by step. Show all work. \
             Check your answer before finalizing."
        )
        .input(InputField::new("problem", "Mathematical problem to solve"))
        .output(OutputField::new("steps", "Detailed solution steps"))
        .output(OutputField::new("answer", "Numerical answer"))
        .output(OutputField::new("verification", "Verification of the answer"))
        .build();

    assert_eq!(math_sig.outputs().len(), 3);

    // ── 4. Decision-making CoT ──
    // For complex decisions, CoT produces a structured analysis.

    let decision_sig = SignatureBuilder::new("DecisionAnalysis")
        .description(
            "Analyze the situation from multiple perspectives. \
             Consider pros, cons, and risks before deciding."
        )
        .input(InputField::new("situation", "The situation to analyze"))
        .input(InputField::new("constraints", "Constraints and requirements"))
        .output(OutputField::new("analysis", "Multi-perspective analysis"))
        .output(OutputField::new("pros", "Arguments in favor"))
        .output(OutputField::new("cons", "Arguments against"))
        .output(OutputField::new("risks", "Identified risks"))
        .output(OutputField::new("recommendation", "Final recommendation"))
        .build();

    assert_eq!(decision_sig.inputs().len(), 2);
    assert_eq!(decision_sig.outputs().len(), 5);

    // ── 5. CoT vs direct answering ──
    // Key insight: the difference is purely in the signature.
    // Same agent, same model, different signature -> different behavior.
    //
    // Direct:
    //   input: question -> output: answer
    //
    // Chain of Thought:
    //   input: question -> output: reasoning, answer
    //
    // The "reasoning" field before "answer" is what triggers CoT.
    // The model must produce reasoning first, which improves answer quality.

    // ── 6. Key concepts ──
    // - CoT: add "reasoning" output BEFORE "answer" output
    // - Model produces reasoning first, improving accuracy
    // - Effective for math, logic, classification, decisions
    // - Same model, different signature -> different behavior
    // - Self-evaluation: add "confidence" after reasoning
    // - Verification: add "verification" to catch errors

    println!("PASS: 04-agents/04_chain_of_thought");
}
