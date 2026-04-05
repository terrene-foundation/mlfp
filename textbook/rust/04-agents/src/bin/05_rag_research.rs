// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Agents / RAG Research Agent
//!
//! OBJECTIVE: Build a retrieval-augmented generation agent for research tasks.
//! LEVEL: Intermediate
//! PARITY: Equivalent -- Python has RAGResearchAgent;
//!         Rust uses the same pattern with retriever tool integration.
//! VALIDATES: RAG pattern, retriever tools, context injection, citation tracking
//!
//! Run: cargo run -p tutorial-agents --bin 05_rag_research

use kailash_kaizen::agent::{
    InputField, OutputField, SignatureBuilder,
};
use kaizen_agents::delegate_engine::ToolDef;

fn main() {
    // ── 1. RAG pattern ──
    // Retrieval-Augmented Generation adds external knowledge to LLM calls:
    //   1. RETRIEVE: Fetch relevant documents from a knowledge base
    //   2. AUGMENT: Inject retrieved context into the prompt
    //   3. GENERATE: LLM produces an answer grounded in the context
    //
    // This prevents hallucination by grounding responses in real data.

    // ── 2. RAG Signature ──
    // The signature captures the full RAG interaction.

    let rag_sig = SignatureBuilder::new("RAGResearch")
        .description(
            "You are a research assistant. Answer questions using only \
             the provided context. Cite your sources. If the context \
             doesn't contain the answer, say so."
        )
        .input(InputField::new("question", "Research question"))
        .input(InputField::new("context", "Retrieved documents").with_default(""))
        .output(OutputField::new("answer", "Answer grounded in the context"))
        .output(OutputField::new("sources", "Sources used (document IDs or titles)"))
        .output(OutputField::new(
            "confidence",
            "Confidence that the answer is fully supported by context",
        ))
        .build();

    assert_eq!(rag_sig.inputs().len(), 2);
    assert_eq!(rag_sig.outputs().len(), 3);

    // ── 3. Retriever tool ──
    // The retriever is a tool the agent calls to fetch relevant documents.
    // It takes a query and returns ranked passages.

    let retriever_tool = ToolDef::new(
        "search_knowledge_base",
        "Search the knowledge base for relevant documents",
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for the knowledge base"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of documents to retrieve",
                    "default": 5
                }
            },
            "required": ["query"]
        }),
    );

    assert_eq!(retriever_tool.name(), "search_knowledge_base");

    // ── 4. Additional RAG tools ──
    // Production RAG systems have multiple retrieval strategies.

    let _doc_lookup = ToolDef::new(
        "get_document",
        "Retrieve a specific document by ID",
        serde_json::json!({
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "Document ID"}
            },
            "required": ["document_id"]
        }),
    );

    let _web_search = ToolDef::new(
        "web_search",
        "Search the web for recent information",
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Web search query"},
                "max_results": {"type": "integer", "default": 3}
            },
            "required": ["query"]
        }),
    );

    // ── 5. Multi-source RAG ──
    // Advanced RAG agents search multiple sources and cross-reference.

    let multi_rag_sig = SignatureBuilder::new("MultiSourceRAG")
        .description(
            "Research across multiple knowledge bases. Cross-reference \
             findings. Flag contradictions between sources."
        )
        .input(InputField::new("question", "Research question"))
        .output(OutputField::new("answer", "Synthesized answer from all sources"))
        .output(OutputField::new("sources", "All sources used with relevance scores"))
        .output(OutputField::new(
            "contradictions",
            "Any contradictions found between sources",
        ))
        .output(OutputField::new("gaps", "Information gaps identified"))
        .build();

    assert_eq!(multi_rag_sig.outputs().len(), 4);

    // ── 6. RAG execution pattern ──
    // The agent autonomously decides when to retrieve:
    //
    //   // 1. Agent receives question
    //   // 2. Agent calls search_knowledge_base tool
    //   // 3. Agent receives retrieved context
    //   // 4. Agent generates answer citing the context
    //   // 5. If confidence is low, agent retrieves more
    //
    //   let agent = RAGAgent::new(rag_sig, tools, config);
    //   let result = agent.run(inputs!{
    //       "question" => "What is the Terrene Foundation?"
    //   }).await;
    //
    // NOTE: We do not call run() here as it requires an LLM API key.

    // ── 7. Key concepts ──
    // - RAG: Retrieve, Augment, Generate -- grounds LLM in real data
    // - Retriever tool: agent-callable search over knowledge bases
    // - Citation tracking: sources output field for traceability
    // - Confidence: self-assessment of context coverage
    // - Multi-source: cross-referencing, contradiction detection
    // - Agent autonomously decides when and what to retrieve

    println!("PASS: 04-agents/05_rag_research");
}
