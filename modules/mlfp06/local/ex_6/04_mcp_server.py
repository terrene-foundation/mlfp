# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 6.4: MCP Server — Exposing Tools to External Agents
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Model Context Protocol (MCP) basics
#   - Define MCP tools with JSON-schema parameters
#   - Register tools on an MCPServer
#
# PREREQUISITES: 03_parallel_router.py
# ESTIMATED TIME: ~30 min
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from kailash_mcp import MCPServer, MCPTool

from shared.mlfp06.ex_6 import OUTPUT_DIR, load_squad_corpus


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load the SQuAD corpus
# ════════════════════════════════════════════════════════════════════════

# TODO: Load the SQuAD corpus
passages = ____

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert passages.height > 0
print("✓ Checkpoint 1 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Define three tool handlers
# ════════════════════════════════════════════════════════════════════════


def mcp_analyse_passage(passage: str, analysis_type: str = "factual") -> str:
    """Analyse a text passage from the specified perspective.

    Args:
        passage: The text to analyse.
        analysis_type: One of 'factual', 'semantic', 'structural'.
    """
    # TODO: Return a short formatted string describing what analysis
    # would run (this is a stub — in production it would delegate to
    # the matching specialist agent).
    return ____


def mcp_search_corpus(query: str, top_k: int = 3) -> str:
    """Search the SQuAD corpus for passages matching a query."""
    query_lower = query.lower()
    scored = []
    for row in passages.iter_rows(named=True):
        # TODO: Score each row by counting how many query words appear
        # in row["text"].lower(). Only keep rows with score > 0.
        score = ____
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [f"[{row['title']}] {row['text'][:200]}..." for _, row in scored[:top_k]]
    return "\n\n".join(results) if results else "No matches found."


def mcp_get_corpus_stats() -> str:
    """Get statistics about the available document corpus."""
    # TODO: Return a string with passage count, unique title count,
    # and the first 10 unique titles.
    return ____


# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert mcp_get_corpus_stats().startswith("Corpus:")
print("✓ Checkpoint 2 passed — 3 handlers callable\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Wrap handlers as MCPTools
# ════════════════════════════════════════════════════════════════════════

# TODO: Build a list of three MCPTool instances. Each needs:
#   - name, description, handler, parameters (JSON schema).
# See the reference at https://modelcontextprotocol.io and kailash_mcp docs.
mcp_tools = ____

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert len(mcp_tools) == 3
print("✓ Checkpoint 3 passed — 3 MCPTools defined\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Register tools on an MCPServer
# ════════════════════════════════════════════════════════════════════════

# TODO: Instantiate MCPServer(name="mlfp06-analysis-server") and register
# each tool in mcp_tools via mcp_server.register_tool(tool).
mcp_server = ____
for tool in mcp_tools:
    ____

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert mcp_server is not None
assert len(mcp_tools) == 3
print("✓ Checkpoint 4 passed — MCP server ready\n")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Singapore scenario: shared MCP tools across bank AI teams
# ════════════════════════════════════════════════════════════════════════
# A Singapore bank runs retail credit, corporate credit, and AML
# monitoring as separate AI teams. Legacy per-pair glue costs
# ~27 person-weeks per new shared tool. MCP collapses this to ~1
# person-week — ~S$91K saved per shared tool, with a single
# MAS-TRM-auditable surface.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] MCP as the "USB for AI agents"
  [x] MCPTool: handler + schema + description
  [x] MCPServer registration and capability introspection

  Next: 05_memory_and_security.py — agent memory and multi-agent
  security guards.
"""
)

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — six lenses before completion
# ══════════════════════════════════════════════════════════════════
# The LLM Observatory extends M5's Doctor's Bag for LLM/agent work.
# Six lenses:
#   1. Output        — is the generation coherent, factual, on-task?
#   2. Attention     — what does the model attend to internally?
#   3. Retrieval     — did we fetch the right context?  [RAG only]
#   4. Agent Trace   — what did the agent actually do?  [Agent only]
#   5. Alignment     — is it aligned with our intent?   [Fine-tune only]
#   6. Governance    — is it within policy?            [PACT only]
from shared.mlfp06.diagnostics import LLMObservatory

# Primary lens: Agent Trace (inter-agent handoffs, tool latency).
# Secondary: Governance (envelope verification when a supervisor is
# governed).
if False:  # scaffold — requires a live multi-agent setup
    obs = LLMObservatory(run_id="ex_6_multiagent_run")
    # for run_id, trace in supervisor.all_traces.items():
    #     obs.agent.register_trace(trace)
    # obs.agent.handoff_summary()  # inter-agent handoffs
    print("\n── LLM Observatory Report ──")
    findings = obs.report()

# ══════ EXPECTED OUTPUT (synthesised reference) ══════
# ════════════════════════════════════════════════════════════════
#   LLM Observatory — composite Prescription Pad
# ════════════════════════════════════════════════════════════════
#   [✓] Agent      (HEALTHY): 3 workers, 7 handoffs, mean tool-call
#       latency 840ms, no stuck loops across all runs.
#   [?] Governance (UNKNOWN): no PACT engine attached in this lesson;
#       attach supervisor.audit to light up this lens.
#   [?] Output / Retrieval / Alignment / Attention (n/a)
# ════════════════════════════════════════════════════════════════
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:
#
#  [AGENT LENS] 7 handoffs across 3 workers is the signature of a
#     healthy Supervisor-Worker pattern — supervisor delegates, workers
#     report back, supervisor synthesises. Mean latency 840ms per tool
#     call is dominated by LLM inference, not tool execution. Watch for:
#     (a) a worker that handoffs 0 times = it's not being used;
#     (b) latency >5s = a tool is I/O bound and needs caching.
#  [GOVERNANCE LENS] UNKNOWN is expected in ex_6 — governance shows up
#     in ex_7 where the GovernedSupervisor attaches its audit trail.
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
