# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 6.4: MCP Server — Exposing Tools to External Agents
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Understand Model Context Protocol (MCP) as a standard way to
#     expose tools to any MCP-compatible agent
#   - Register typed tool handlers on a kailash-mcp MCPServer — the type
#     hints and docstring BECOME the tool's published JSON schema
#   - Run the server on the stdio transport and call it from a real MCP
#     client: discover the tools, call them, watch bad input be rejected
#   - Expose a specialist AGENT as an MCP tool, not just a lookup
#
# PREREQUISITES: 03_parallel_router.py
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Load the shared corpus (tools will search it)
#   2. Create the server and register three typed tool handlers
#      (analyse_passage runs a specialist agent; search_corpus; stats)
#   3. Discover the tools through an MCP client over stdio and read the
#      schemas the server publishes
#   4. Call every tool through the protocol, including one call the
#      schema must reject
#
# HOW TO RUN: as a script (python .../04_mcp_server.py).  Task 3 launches
# THIS file as a stdio MCP server subprocess (with --serve-mcp), so the
# file you write is the server the client talks to.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import sys

# stdio transport: while serving, stdout carries the MCP JSON-RPC frames
# and nothing else.  Every print() in this file must go to stderr in
# server mode, or the client receives garbage instead of protocol frames.
SERVE_MODE = "--serve-mcp" in sys.argv
if SERVE_MODE:
    _PROTOCOL_STDOUT = sys.stdout
    sys.stdout = sys.stderr

import asyncio
import json
import time
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import polars as pl
from kailash_mcp import MCPClient, MCPServer

from shared.mlfp06._ollama_bootstrap import preflight_ollama
from shared.mlfp06.ex_6 import (
    MODEL,
    OUTPUT_DIR,
    build_specialists,
    load_squad_corpus,
    run_checked,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Model Context Protocol
# ════════════════════════════════════════════════════════════════════════
# MCP is a small protocol that lets an AI agent discover and call
# tools exposed by a server. A "tool" is a function plus a JSON
# schema describing its parameters plus a natural-language
# description. Any MCP-compatible client — an IDE assistant, a desktop
# chat app, a Kaizen agent — can list the server's tools (the JSON-RPC
# method `tools/list`) and invoke them (`tools/call`).
#
# Non-technical analogy: MCP is USB for AI agents. Your specialists
# are devices; MCP is the plug. Any agent that speaks MCP can
# discover what's attached and use it, without custom glue code
# per agent type.
#
# COMPONENTS:
#   - Server:    registers tools with schemas, listens on a transport
#   - Transport: stdio for a local subprocess, HTTP/SSE for remote
#   - Tool:      handler function + JSON schema + description
#   - Resource:  read-only data the agent can access
#
# WHERE THE SCHEMA COMES FROM (kailash-mcp):
#   You decorate a plain Python function with `@server.tool()`.  The
#   server derives the tool's published JSON input schema from the
#   function's TYPE HINTS (str -> "string", int -> "integer",
#   Literal["a", "b"] -> an enum) and its description from the
#   docstring.  Precise type hints = a precise schema, and the server
#   validates every incoming call against it before your code runs.
#
# WHY THIS MATTERS FOR MULTI-AGENT:
# MCP lets your agents share tools without hard-coding imports. A
# Kaizen supervisor can call an MCP tool exposed by a Python service;
# a desktop assistant can call the same tool; an IDE plugin can call
# it. One registration, many consumers.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load the corpus (tools will operate on it)
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 1: Load SQuAD corpus for MCP tools")
print("=" * 70)

passages = load_squad_corpus()
print(
    f"Corpus: {passages.height} passages across "
    f"{passages['title'].n_unique()} titles"
)

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert passages.height > 0
print("✓ Checkpoint 1 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Create the server, then register three typed tool handlers
# ════════════════════════════════════════════════════════════════════════
# In kailash-mcp you register a plain function ON the server with the
# @server.tool() decorator, so the server comes first.  We declare it
# on the stdio transport (a local subprocess); HTTP/SSE is a
# constructor flag away when you need remote agents.

print("=" * 70)
print("TASK 2: Create MCPServer and register three tools")
print("=" * 70)

mcp_server = MCPServer(name="mlfp06-analysis-server", transport="stdio")

# analyse_passage delegates to the Exercise 6.1 specialist agents, so an
# external MCP client gets real factual / semantic / structural analysis.
SPECIALISTS = dict(zip(("factual", "semantic", "structural"), build_specialists()))


@mcp_server.tool()
async def analyse_passage(
    passage: str,
    analysis_type: Literal["factual", "semantic", "structural"] = "factual",
    question: str = "What are the key points of this passage?",
) -> str:
    """Analyse a passage with the factual, semantic, or structural specialist agent.

    Args:
        passage: The text to analyse.
        analysis_type: Which specialist runs: factual, semantic, or structural.
        question: The question the analysis should focus on.

    Returns:
        The specialist's structured output as JSON text.
    """
    result = await run_checked(
        SPECIALISTS[analysis_type], document=passage, question=question
    )
    return json.dumps(result, default=str)


@mcp_server.tool()
def search_corpus(query: str, top_k: int = 3) -> str:
    """Search the document corpus for passages matching a query.

    Args:
        query: Search query text.
        top_k: Maximum results to return.

    Returns:
        Matching passages with their titles.
    """
    query_lower = query.lower()
    scored = []
    for row in passages.iter_rows(named=True):
        score = sum(1 for w in query_lower.split() if w in row["text"].lower())
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [f"[{row['title']}] {row['text'][:200]}..." for _, row in scored[:top_k]]
    return "\n\n".join(results) if results else "No matches found."


@mcp_server.tool()
def get_corpus_stats() -> str:
    """Get statistics about the available document corpus.

    Returns:
        Corpus statistics including size, topics, and coverage.
    """
    return (
        f"Corpus: {passages.height} passages, "
        f"{passages['title'].n_unique()} unique topics\n"
        f"First 10 topics: {passages['title'].unique().sort().to_list()[:10]}"
    )


registered_names = sorted(mcp_server.get_tool_stats()["tools"])
print(f"Registered on the server: {registered_names}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert registered_names == ["analyse_passage", "get_corpus_stats", "search_corpus"]
assert get_corpus_stats().startswith("Corpus:")
print("✓ Checkpoint 2 passed — 3 handlers registered\n")

# ── Server mode ──────────────────────────────────────────────────────────
# When the MCP client launches this file with --serve-mcp, hand stdout
# back to the protocol and serve until the client disconnects.  Nothing
# below this point runs in server mode.
if SERVE_MODE:
    sys.stdout = _PROTOCOL_STDOUT
    mcp_server.run()
    raise SystemExit(0)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Discover the tools through a real MCP client (stdio)
# ════════════════════════════════════════════════════════════════════════
# The client starts this file as a subprocess server and speaks MCP to
# it over stdin/stdout: `initialize`, then `tools/list`.  What comes
# back is exactly what any external agent would see.

print("=" * 70)
print("TASK 3: Discover tools over the MCP protocol")
print("=" * 70)

_this_file = globals().get("__file__")
if _this_file is None:
    raise RuntimeError(
        "Task 3 launches this exercise file as a stdio MCP server, so it "
        "must run as a script: python modules/mlfp06/.../04_mcp_server.py"
    )
SERVER_CONFIG = {
    "transport": "stdio",
    "command": sys.executable,
    "args": [str(Path(_this_file).resolve()), "--serve-mcp"],
}
client = MCPClient()

discovered = asyncio.run(client.discover_tools(SERVER_CONFIG, timeout=120))
schemas = {tool["name"]: tool["parameters"] for tool in discovered}
for tool in discovered:
    params = tool["parameters"].get("properties", {})
    print(f"  {tool['name']}: {(tool['description'] or '').splitlines()[0]}")
    for name, spec in params.items():
        extra = f" enum={spec['enum']}" if "enum" in spec else ""
        print(f"      {name}: {spec.get('type')}{extra}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert sorted(schemas) == registered_names, (
    "Task 3: the client should discover all 3 tools — an empty list means "
    "the server subprocess failed to start (check stderr)"
)
assert schemas["analyse_passage"]["properties"]["analysis_type"]["enum"] == [
    "factual",
    "semantic",
    "structural",
], "The Literal type hint should be published as an enum"
assert schemas["search_corpus"]["required"] == ["query"]
print("\n✓ Checkpoint 3 passed — schemas published from the type hints\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Call every tool through the protocol
# ════════════════════════════════════════════════════════════════════════
# Each call is a `tools/call` round trip.  The last call sends an
# analysis_type that is not in the enum: the SERVER must reject it
# before the handler runs.

print("=" * 70)
print("TASK 4: Call the tools over MCP")
print("=" * 70)

preflight_ollama(required_models=[MODEL])  # analyse_passage runs an agent

calls = [
    ("get_corpus_stats", {}),
    ("search_corpus", {"query": "university students", "top_k": 2}),
    (
        "analyse_passage",
        {
            "passage": passages["text"][0],
            "analysis_type": "structural",
            "question": passages["question"][0],
        },
    ),
    ("analyse_passage", {"passage": "Some text.", "analysis_type": "sentiment"}),
]


async def call_all() -> list[dict]:
    log = []
    for name, arguments in calls:
        t0 = time.perf_counter()
        response = await client.call_tool(SERVER_CONFIG, name, arguments, timeout=300)
        mcp_result = response.get("result")
        log.append(
            {
                "tool": name,
                "transport_ok": bool(response.get("success")),
                "is_error": bool(getattr(mcp_result, "isError", False)),
                "seconds": time.perf_counter() - t0,
                "content": response.get("content") or response.get("error", ""),
            }
        )
    return log


call_log = pl.DataFrame(asyncio.run(call_all()))
for row in call_log.iter_rows(named=True):
    status = "ERROR" if row["is_error"] else "ok"
    print(f"\n  {row['tool']} [{status}, {row['seconds']:.1f}s]")
    print(f"    {row['content'][:300]}")

trace_path = OUTPUT_DIR / "ex6_mcp_server_trace.txt"
trace_path.write_text(
    f"Server: {mcp_server.name}\nTools: {registered_names}\n\n"
    + "\n".join(
        f"{r['tool']}: error={r['is_error']} {r['seconds']:.2f}s"
        for r in call_log.iter_rows(named=True)
    )
    + "\n"
)
print(f"\nTrace written to: {trace_path}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert call_log["transport_ok"].all(), "Every call should complete the round trip"
assert not call_log["is_error"][:3].any(), (
    "The three valid calls must succeed — an analyse_passage error usually "
    "means Ollama is not running in the server process (ollama serve)"
)
assert call_log["is_error"][3], "The out-of-enum analysis_type must be rejected"
analysis = json.loads(call_log["content"][2])
assert analysis["key_entities"], "The structural specialist should return entities"
print("\n✓ Checkpoint 4 passed — 3 valid calls served, 1 invalid call rejected\n")

# INTERPRETATION: the rejected call never reached analyse_passage — the
# server validated the arguments against the schema it published in
# Task 3.  That is what a typed tool buys you: bad input from ANY client
# is stopped at the boundary, not inside your business logic.


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Measured tool-call latency over MCP
# ════════════════════════════════════════════════════════════════════════
# Every bar is a call you just made.  The stdio client starts a fresh
# server process per call, so each bar includes process start-up; the
# analyse_passage bar additionally includes a specialist LLM call.  A
# long-running server (HTTP/SSE) pays start-up once.

labels = [
    f"{r['tool']}\n({'rejected' if r['is_error'] else 'ok'})"
    for r in call_log.iter_rows(named=True)
]
fig, ax = plt.subplots(figsize=(9, 4))
colors = ["#e74c3c" if e else "#3498db" for e in call_log["is_error"].to_list()]
bars = ax.bar(labels, call_log["seconds"].to_list(), color=colors)
ax.set_ylabel("Round-trip seconds (measured)")
ax.set_title("MCP tools/call Round Trips — This Run", fontweight="bold")
for bar, sec in zip(bars, call_log["seconds"].to_list()):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        sec,
        f"{sec:.1f}s",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fname = OUTPUT_DIR / "ex6_mcp_tool_frequency.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {fname}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Singapore scenario: shared analysis tools across bank AI teams
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative figures): A Singapore financial institution
# runs three separate AI teams — retail credit, corporate credit, and
# AML/transaction monitoring. Each team has built its own ad-hoc
# "analyse document" and "search knowledge base" helpers, each wired to
# a different LLM framework. Reusing one team's tool from another team
# costs ~3 person-weeks of glue code per integration.
#
# MCP replaces the glue: retail credit exposes its tools on an
# MCPServer once. AML points its Kaizen supervisor at the server and
# lists tools at runtime. Corporate credit points a desktop assistant
# at the same server. One registration, three consumers, zero custom
# glue.
#
# IMPACT (per new tool every team wants):
#   Point-to-point integrations: 3 teams, each needs the other two's
#                                version = 6 directed integrations
#   Legacy glue cost:            6 x ~3 person-weeks = ~18 person-weeks
#   MCP cost:                    ~1 person-week (one registration)
#   Savings per shared tool:     ~17 person-weeks x S$3,500/week
#                                ≈ S$60K
#   Audit bonus: ONE set of tool schemas, ONE call log, ONE access-control
#   surface to review — not six.

print("=" * 70)
print("  SINGAPORE APPLICATION: Shared MCP Tools Across Bank AI Teams")
print("=" * 70)
print(
    """
  Teams: retail credit, corporate credit, AML monitoring (illustrative)
  Legacy integration cost per new tool:  ~18 person-weeks
  MCP integration cost per new tool:     ~1 person-week
  Savings per shared tool:                ~S$60K
  Audit surface:                          1 (schemas, logs, ACLs) — not 6
"""
)


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT
# ══════════════════════════════════════════════════════════════════
# The diagnostic for an MCP server is its protocol behaviour, which you
# measured: the published schemas (Task 3), the per-call outcomes, and
# the rejected out-of-enum call (Task 4).  Watch for (a) an empty
# discovery list — the server process crashed or printed to stdout;
# (b) a valid call returning an error — the handler's dependencies
# (here, Ollama) are not available inside the server process.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] MCP as the "USB for AI agents": one server, many consumers
  [x] @server.tool registration: type hints + docstring = published schema
  [x] A real MCP client round trip over stdio: tools/list, tools/call,
      and server-side rejection of invalid input
  [x] A specialist agent exposed as an MCP tool
  [x] Transport trade-off: stdio (local) vs HTTP/SSE (remote)
  [x] Singapore bank scenario: MCP collapses point-to-point glue into
      one registration per tool

  KEY INSIGHT: The moment you expose a tool via MCP, its capability
  card becomes discoverable by EVERY MCP-compatible agent — that is
  the leverage point. Don't write tools that only your own supervisor
  can call. Write tools that any agent can call.

  Next: 05_memory_and_security.py — wiring agent memory (short-term,
  long-term, entity) and guarding against multi-agent attack patterns.
"""
)
