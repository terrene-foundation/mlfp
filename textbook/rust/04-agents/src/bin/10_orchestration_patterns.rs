// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Agents / Orchestration Patterns
//!
//! OBJECTIVE: Survey all agent orchestration patterns and when to use each.
//! LEVEL: Advanced
//! PARITY: Equivalent -- Both SDKs support the same orchestration taxonomy.
//! VALIDATES: Pattern taxonomy, selection criteria, composition rules
//!
//! Run: cargo run -p tutorial-agents --bin 10_orchestration_patterns

fn main() {
    // ── 1. Pattern Taxonomy ──
    // Kaizen provides a hierarchy of agent orchestration patterns,
    // from simplest to most complex.

    println!("=== Agent Orchestration Patterns ===");
    println!();

    // ── 2. Single-Shot (SimpleQA) ──
    // One input, one LLM call, one output. No tools, no iteration.
    //
    // Use when: Classification, translation, summarization, simple Q&A
    // Cost: 1 LLM call
    // Complexity: Minimal

    println!("Pattern 1: Single-Shot (SimpleQA)");
    println!("  - One LLM call, no tools, no memory");
    println!("  - Best for: classification, translation, summarization");

    // ── 3. Chain of Thought (CoT) ──
    // Single-shot with explicit reasoning before answering.
    //
    // Use when: Math, logic, complex classification
    // Cost: 1 LLM call (larger output)
    // Complexity: Low

    println!("Pattern 2: Chain of Thought (CoT)");
    println!("  - Reasoning output before answer output");
    println!("  - Best for: math, logic, complex decisions");

    // ── 4. ReAct ──
    // Iterative: Reason -> Act -> Observe -> Decide.
    //
    // Use when: Tasks requiring external information, multi-step research
    // Cost: N LLM calls (N = cycles used)
    // Complexity: Moderate

    println!("Pattern 3: ReAct");
    println!("  - Iterative reason-act-observe loop");
    println!("  - Best for: research, investigation, tool-heavy tasks");

    // ── 5. RAG (Retrieval-Augmented Generation) ──
    // ReAct specialized for knowledge retrieval.
    //
    // Use when: Questions about specific data, documentation, codebases
    // Cost: N LLM calls + retrieval costs
    // Complexity: Moderate

    println!("Pattern 4: RAG");
    println!("  - ReAct specialized for knowledge retrieval");
    println!("  - Best for: domain Q&A, documentation search");

    // ── 6. Pipeline ──
    // Sequential or routed multi-agent chain.
    //
    // Use when: Tasks with distinct processing stages
    // Cost: M LLM calls (M = stages)
    // Complexity: Moderate-High

    println!("Pattern 5: Pipeline");
    println!("  - Sequential, router, or parallel multi-agent");
    println!("  - Best for: multi-stage processing, support routing");

    // ── 7. Tree of Thoughts (ToT) ──
    // Parallel exploration of multiple reasoning paths.
    //
    // Use when: Creative problems, planning, puzzles
    // Cost: B * D LLM calls (B = branches, D = depth)
    // Complexity: High

    println!("Pattern 6: Tree of Thoughts");
    println!("  - Parallel branch exploration with pruning");
    println!("  - Best for: creative tasks, planning, optimization");

    // ── 8. Governed Supervisor ──
    // Supervisor routes to sub-agents under PACT governance.
    //
    // Use when: Complex tasks spanning multiple domains with budget/security
    // Cost: Variable (depends on sub-agent execution)
    // Complexity: High

    println!("Pattern 7: Governed Supervisor");
    println!("  - Multi-agent under governance policies");
    println!("  - Best for: enterprise, multi-domain, compliance");

    // ── 9. Selection criteria ──
    //
    // | Criterion       | Single | CoT  | ReAct | RAG  | Pipeline | ToT  | Supervisor |
    // |-----------------|--------|------|-------|------|----------|------|------------|
    // | Tools needed    | No     | No   | Yes   | Yes  | Varies   | No   | Yes        |
    // | Iterations      | 1      | 1    | N     | N    | M stages | B*D  | Variable   |
    // | Governance      | No     | No   | No    | No   | Optional | No   | Required   |
    // | Cost (relative) | $      | $    | $$    | $$   | $$$      | $$$$ | $$$$$      |

    // ── 10. Composition rules ──
    // Patterns can be composed:
    //   - Pipeline stage using ReAct agents
    //   - Supervisor delegating to Pipeline sub-agents
    //   - RAG agent within a Pipeline stage
    //   - ToT for the planning stage, ReAct for execution
    //
    // Rule: Start with the simplest pattern that works.
    // Escalate complexity only when simpler patterns fail.

    println!();
    println!("Selection rule: Start simple, escalate only when needed.");
    println!("  SimpleQA -> CoT -> ReAct -> RAG -> Pipeline -> ToT -> Supervisor");

    // ── 11. Key concepts ──
    // - 7 orchestration patterns from simple to complex
    // - Each pattern has cost, complexity, and applicability trade-offs
    // - Patterns are composable (pipeline of ReAct agents, etc.)
    // - Always start with the simplest pattern that meets requirements
    // - Governance (PACT) only required for supervisor pattern
    // - LLM-based routing at every level (no keyword dispatch)

    println!("\nPASS: 04-agents/10_orchestration_patterns");
}
