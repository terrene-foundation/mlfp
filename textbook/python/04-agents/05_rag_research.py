# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Agents / RAG Research Agent
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build a RAGResearchAgent for retrieval-augmented generation
# LEVEL: Intermediate
# PARITY: Python-only — no Rust equivalent
# VALIDATES: RAGResearchAgent, RAGConfig, RAGSignature, vector store
#            integration, document management, convergence detection
#
# Run: uv run python textbook/python/04-agents/05_rag_research.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os

from kaizen_agents.agents.specialized.rag_research import (
    RAGResearchAgent,
    RAGConfig,
    RAGSignature,
    SAMPLE_AI_DOCUMENTS,
)
from kaizen.core.base_agent import BaseAgent
from kaizen.strategies.multi_cycle import MultiCycleStrategy

model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")

# ── 1. RAG pattern: Retrieve + Augment + Generate ─────────────────────
# RAG agents combine vector search with LLM generation:
#   1. RETRIEVE: Semantic search over a document corpus
#   2. AUGMENT: Inject relevant documents into the LLM context
#   3. GENERATE: LLM synthesizes an answer with source attribution
#
# RAGResearchAgent is an autonomous agent (MultiCycleStrategy) that
# can iteratively refine its search: query -> fetch -> analyze -> refine.

# ── 2. RAGSignature — retrieval-augmented output ──────────────────────

assert "query" in RAGSignature._signature_inputs

assert "answer" in RAGSignature._signature_outputs
assert "sources" in RAGSignature._signature_outputs
assert "confidence" in RAGSignature._signature_outputs
assert "relevant_excerpts" in RAGSignature._signature_outputs
assert (
    "tool_calls" in RAGSignature._signature_outputs
), "ADR-013: tool_calls for convergence (web_search, fetch_url, etc.)"

# ── 3. RAGConfig — retrieval-specific settings ────────────────────────

config = RAGConfig()

assert config.top_k_documents == 3, "Retrieve top 3 documents by default"
assert config.similarity_threshold == 0.3, "Minimum similarity score"
assert config.embedding_model == "all-MiniLM-L6-v2", "Default embedding model"
assert config.max_cycles == 15, "Research may need many cycles"
assert config.mcp_enabled is True, "Autonomous agents discover MCP tools"

# Custom retrieval configuration
custom_config = RAGConfig(
    top_k_documents=5,
    similarity_threshold=0.4,
    embedding_model="all-mpnet-base-v2",
    max_cycles=20,
)
assert custom_config.top_k_documents == 5
assert custom_config.similarity_threshold == 0.4

# ── 4. Sample documents (built-in knowledge base) ─────────────────────
# RAGResearchAgent ships with sample AI documents for testing.
# In production, you provide your own document corpus.

assert len(SAMPLE_AI_DOCUMENTS) == 5

doc_ids = {doc["id"] for doc in SAMPLE_AI_DOCUMENTS}
assert "doc1" in doc_ids  # Machine Learning
assert "doc2" in doc_ids  # Deep Learning
assert "doc3" in doc_ids  # NLP
assert "doc4" in doc_ids  # Computer Vision
assert "doc5" in doc_ids  # Reinforcement Learning

# Every document has id, title, content
for doc in SAMPLE_AI_DOCUMENTS:
    assert "id" in doc
    assert "title" in doc
    assert "content" in doc
    assert len(doc["content"]) > 50, "Documents have substantial content"

# ── 5. RAGResearchAgent instantiation ─────────────────────────────────
# The agent initializes a vector store with sample documents
# and uses MultiCycleStrategy for autonomous execution.

agent = RAGResearchAgent(
    llm_provider="mock",
    model=model,
)

assert isinstance(agent, RAGResearchAgent)
assert isinstance(agent, BaseAgent)
assert isinstance(
    agent.strategy, MultiCycleStrategy
), "RAG MUST use MultiCycleStrategy for autonomous research"

# ── 6. Vector store — document management ─────────────────────────────
# The agent manages a vector store for semantic search.
# Documents are added, searched, and can be cleared.

assert agent.get_document_count() == 5, "Sample docs loaded by default"

# Add a custom document
agent.add_document(
    doc_id="doc6",
    title="Quantum Computing",
    content="Quantum computing uses quantum mechanics principles such as "
    "superposition and entanglement to perform computations that would be "
    "intractable for classical computers.",
)
assert agent.get_document_count() == 6

# Clear all documents
agent.clear_documents()
assert agent.get_document_count() == 0

# ── 7. Convergence detection for research ─────────────────────────────
# RAG uses the same ADR-013 objective convergence as ReAct:
# tool_calls non-empty -> continue researching
# tool_calls empty     -> research complete

# Research in progress (needs more sources)
result_researching = {
    "tool_calls": [{"name": "web_search", "params": {"query": "deep learning"}}],
}
assert agent._check_convergence(result_researching) is False

# Research complete (no more tools needed)
result_done = {"tool_calls": [], "confidence": 0.9}
assert agent._check_convergence(result_done) is True

# Malformed tool_calls -> stop for safety
result_malformed = {"tool_calls": "not a list"}
assert agent._check_convergence(result_malformed) is True

# Subjective fallback: high confidence + comprehensive depth
result_subjective = {"confidence": 0.9, "research_depth": "comprehensive"}
assert agent._check_convergence(result_subjective) is True

# Default: converged (safe fallback)
assert agent._check_convergence({}) is True

# ── 8. Input validation ───────────────────────────────────────────────

empty_result = agent.run(query="")
assert empty_result["error"] == "INVALID_INPUT"
assert empty_result["sources"] == []
assert empty_result["confidence"] == 0.0

# ── 9. Return structure ───────────────────────────────────────────────
# RAGResearchAgent.run() returns:
# {
#     "answer": "Machine learning is a subset of AI...",
#     "sources": ["doc1", "doc2"],
#     "confidence": 0.85,
#     "relevant_excerpts": [
#         {
#             "title": "Introduction to Machine Learning",
#             "excerpt": "Machine learning is a subset of...",
#             "similarity": 0.87,
#         }
#     ],
#     "retrieval_quality": 0.82,  # average similarity
# }
#
# NOTE: Full run() requires an LLM API key and a populated vector
# store. The above documents the return format.

# ── 10. Custom vector store ───────────────────────────────────────────
# For production use, you can provide your own vector store with
# domain-specific documents instead of the built-in samples.
#
# from kaizen.retrieval.vector_store import SimpleVectorStore
#
# store = SimpleVectorStore(embedding_model="all-mpnet-base-v2")
# store.add_documents([
#     {"id": "d1", "title": "My Doc", "content": "..."},
# ])
#
# agent = RAGResearchAgent(
#     model=model,
#     vector_store=store,
#     top_k_documents=5,
# )

# ── 11. Node metadata ─────────────────────────────────────────────────

assert RAGResearchAgent.metadata.name == "RAGResearchAgent"
assert "rag" in RAGResearchAgent.metadata.tags
assert "vector-search" in RAGResearchAgent.metadata.tags
assert "semantic" in RAGResearchAgent.metadata.tags

print("PASS: 04-agents/05_rag_research")
