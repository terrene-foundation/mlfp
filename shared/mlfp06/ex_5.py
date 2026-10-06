# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP06 Exercise 5 — AI Agents (ReAct, Structured,
Critic, Cost-Bounded).

Contains:
  - HotpotQA multi-hop QA dataset loading (cached parquet)
  - Agent tools: data_summary, search_documents, run_query, answer_question
  - build_tool_registry(): wraps the Python tools in a Kaizen ToolRegistry
    so an Ollama-backed Delegate can actually call them
  - require_llm_trace() / require_agent_result(): turn a failed LLM call
    into a loud, actionable error instead of an empty "result"
  - Model resolution from environment
  - Output directory setup

Technique-specific agent classes and signatures live in the per-technique
files under modules/mlfp06/solutions/ex_5/.
"""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any, Callable

import polars as pl

from shared.kailash_helpers import setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()

from shared.mlfp06._ollama_bootstrap import DEFAULT_CHAT_MODEL

MODEL = DEFAULT_CHAT_MODEL

OLLAMA_HINT = (
    "Is the local Ollama daemon running? Start it with `ollama serve` and "
    f"pull the chat model with `ollama pull {MODEL}`."
)

OUTPUT_DIR = Path("outputs") / "ex5_agents"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# DATASET — HotpotQA (multi-hop QA, 500 cached examples)
# ════════════════════════════════════════════════════════════════════════

CACHE_DIR = Path("data") / "mlfp06" / "hotpotqa"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "hotpotqa_500.parquet"


def load_hotpotqa() -> pl.DataFrame:
    """Load HotpotQA distractor split (500 shuffled examples, cached).

    Returns:
        Polars DataFrame with columns: text, question, answer, level, type.
    """
    if CACHE_FILE.exists():
        print(f"Loading cached HotpotQA from {CACHE_FILE}")
        return pl.read_parquet(CACHE_FILE)

    print("Downloading hotpotqa/hotpot_qa from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
    )
    ds = ds.shuffle(seed=42).select(range(min(500, len(ds))))
    rows = []
    for row in ds:
        context = row["context"]
        titles = context["title"]
        sentences = context["sentences"]
        joined = "\n".join(f"[{t}] " + " ".join(s) for t, s in zip(titles, sentences))
        rows.append(
            {
                "text": joined[:4000],
                "question": row["question"],
                "answer": row["answer"],
                "level": row["level"],
                "type": row["type"],
            }
        )
    qa_data = pl.DataFrame(rows)
    qa_data.write_parquet(CACHE_FILE)
    print(f"Cached {qa_data.height} HotpotQA examples at {CACHE_FILE}")
    return qa_data


# ════════════════════════════════════════════════════════════════════════
# AGENT TOOLS — docstrings ARE the agent's API documentation
# ════════════════════════════════════════════════════════════════════════
#
# Tool docstrings are what the LLM reads to choose WHICH tool to call and
# WHAT arguments to pass. Precise docstrings -> accurate tool selection.
# The tools close over a module-level `_qa_data` DataFrame populated by
# `make_tools()`.

_qa_data: pl.DataFrame | None = None


def _require_data() -> pl.DataFrame:
    if _qa_data is None:
        raise RuntimeError(
            "Agent tools used before make_tools() was called — "
            "call shared.mlfp06.ex_5.make_tools(qa_data) first."
        )
    return _qa_data


def data_summary(dataset_name: str = "qa_data") -> str:
    """Get a statistical summary of the QA dataset.

    Args:
        dataset_name: Which dataset to summarise.  Currently 'qa_data'.

    Returns:
        Text summary including shape, columns, type distribution, and
        average text lengths.
    """
    df = _require_data()
    parts = [
        f"Dataset: {dataset_name}",
        f"Shape: {df.height} rows x {df.width} columns",
        f"Columns: {', '.join(df.columns)}",
    ]
    for col in df.columns:
        dtype = str(df.schema[col])
        if "Utf8" in dtype or "String" in dtype:
            n_unique = df.select(pl.col(col).n_unique()).item()
            avg_len = df.select(pl.col(col).str.len_chars().mean()).item()
            parts.append(f"  {col} ({dtype}): {n_unique} unique, avg_len={avg_len:.0f}")
        elif "Int" in dtype or "Float" in dtype:
            stats = df.select(
                pl.col(col).mean().alias("mean"),
                pl.col(col).min().alias("min"),
                pl.col(col).max().alias("max"),
            ).row(0)
            parts.append(
                f"  {col} ({dtype}): mean={stats[0]}, range=[{stats[1]}, {stats[2]}]"
            )
    return "\n".join(parts)


def search_documents(query: str, top_k: int = 3) -> str:
    """Search the QA corpus for documents matching a keyword query.

    Args:
        query: Keywords to search for in the document texts.
        top_k:  Maximum number of matching documents to return.

    Returns:
        Matching document excerpts with their questions and answers.
    """
    df = _require_data()
    query_lower = query.lower()
    scored = []
    for i, row in enumerate(df.iter_rows(named=True)):
        text = row["text"].lower()
        score = sum(1 for word in query_lower.split() if word in text)
        if score > 0:
            scored.append((score, i, row))
    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, idx, row in scored[:top_k]:
        results.append(
            f"[Doc {idx}] Q: {row['question']}\n"
            f"  A: {row['answer']}\n"
            f"  Context (excerpt): {row['text'][:300]}..."
        )
    return "\n\n".join(results) if results else f"No documents matching '{query}'"


def run_query(query_description: str) -> str:
    """Run a descriptive query against the QA dataset.

    Args:
        query_description: Natural language description of the query
            (e.g., 'count comparison questions', 'find bridge-type questions').

    Returns:
        Query results as formatted text.
    """
    df = _require_data()
    desc = query_description.lower()

    if "count" in desc and "type" in desc:
        counts = df.group_by("type").len().sort("len", descending=True)
        return f"Question types:\n{counts}"
    elif "count" in desc and "level" in desc:
        counts = df.group_by("level").len().sort("len", descending=True)
        return f"Difficulty levels:\n{counts}"
    elif "comparison" in desc:
        comparison = df.filter(pl.col("type") == "comparison")
        return (
            f"Comparison questions: {comparison.height}\n"
            f"Sample: {comparison['question'][0]}"
        )
    elif "bridge" in desc:
        bridge = df.filter(pl.col("type") == "bridge")
        return f"Bridge questions: {bridge.height}\nSample: {bridge['question'][0]}"
    elif "top" in desc or "longest" in desc:
        df_with_len = df.with_columns(pl.col("text").str.len_chars().alias("text_len"))
        top = df_with_len.sort("text_len", descending=True).head(5)
        return f"Top 5 by text length:\n{top.select('question', 'text_len')}"
    else:
        return f"Dataset has {df.height} rows. Columns: {df.columns}"


def answer_question(question: str) -> str:
    """Extract answer evidence for a question from the corpus text itself.

    Args:
        question: The natural-language question to answer.

    Returns:
        The highest-overlap evidence sentences from the most relevant
        documents, with provenance. This tool NEVER reads the dataset's
        stored answer labels — the agent must synthesise the final answer
        from the returned evidence, exactly like a production RAG tool
        that only sees raw documents.
    """
    df = _require_data()
    q_terms = _content_words(question)
    if not q_terms:
        return "Could not extract content words from that question — rephrase it."

    # Rank documents by content-word overlap with the question.
    scored_docs = []
    for i, row in enumerate(df.iter_rows(named=True)):
        text_l = row["text"].lower()
        doc_score = sum(text_l.count(term) for term in q_terms)
        if doc_score > 0:
            scored_docs.append((doc_score, i, row))
    if not scored_docs:
        return "No documents in the corpus overlap with that question."
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    # Inside each top document, extract the most relevant SENTENCES.
    sections = []
    for doc_score, idx, row in scored_docs[:2]:
        sentences = re.split(r"(?<=[.!?])\s+", row["text"])
        scored_sents = sorted(
            (
                (sum(s.lower().count(term) for term in q_terms), s.strip())
                for s in sentences
                if s.strip()
            ),
            key=lambda x: x[0],
            reverse=True,
        )
        evidence = [s for sc, s in scored_sents[:3] if sc > 0]
        if not evidence:
            continue
        excerpt = "\n".join(f"  - {sent}" for sent in evidence)
        sections.append(f"[Doc {idx}] (overlap={doc_score})\n{excerpt}")

    if not sections:
        return "Relevant documents found but no sentence matched the question terms."
    return (
        "\n\n".join(sections)
        + "\n\n(Evidence extracted from the corpus text. Synthesise the final "
        "answer from these sentences — this tool has no access to answer labels.)"
    )


_STOPWORDS = frozenset(
    "the a an is are was were of in on at to and or for with what which who "
    "whom whose when where why how did does do by as that this it its from "
    "be been has have had not no yes".split()
)


def _content_words(text: str) -> list[str]:
    """Lower-cased, punctuation-stripped question terms minus stopwords."""
    return [
        w
        for w in (t.strip("?,.\"'():;").lower() for t in text.split())
        if w and w not in _STOPWORDS
    ]


def make_tools(qa_data: pl.DataFrame) -> list:
    """Bind the qa_data DataFrame to the tool closures and return the list.

    Args:
        qa_data: The HotpotQA DataFrame from load_hotpotqa().

    Returns:
        List of 4 tool callables. Pass them through build_tool_registry()
        before handing them to a Delegate.
    """
    global _qa_data
    _qa_data = qa_data
    return [data_summary, search_documents, run_query, answer_question]


def tool_schemas(tools: list) -> list[dict]:
    """Build JSON Schema descriptors for a list of tool callables.

    Produces the function-calling shape every tool-capable chat model
    expects (name, description, parameters.properties + required).
    build_tool_registry() registers exactly these schemas with Kaizen.
    """
    schemas = []
    for tool in tools:
        sig = inspect.signature(tool)
        params = {}
        required = []
        for name, param in sig.parameters.items():
            # `from __future__ import annotations` keeps hints as strings.
            annotation = param.annotation
            param_type = "string"
            if annotation in (int, "int"):
                param_type = "integer"
            elif annotation in (float, "float"):
                param_type = "number"
            params[name] = {
                "type": param_type,
                "description": f"Parameter: {name}",
            }
            if param.default is inspect.Parameter.empty:
                required.append(name)
        schemas.append(
            {
                "name": tool.__name__,
                "description": (tool.__doc__ or "").strip().split("\n")[0],
                "parameters": {
                    "type": "object",
                    "properties": params,
                    "required": required,
                },
            }
        )
    return schemas


def build_tool_registry(tools: list[Callable[..., str]]) -> Any:
    """Register plain Python tools on a Kaizen ``ToolRegistry``.

    A Kaizen ``Delegate`` only calls tools that live in its registry: each
    entry is (name, description, JSON-schema parameters, async executor).
    Handing a Delegate a bare list of Python callables registers NOTHING,
    so this helper is the bridge between "functions with docstrings" and
    "tools the model can call".

    Args:
        tools: Callables returning ``str`` (e.g. from :func:`make_tools`).

    Returns:
        A populated ``kaizen_agents.delegate.loop.ToolRegistry``.
    """
    from kaizen_agents.delegate.loop import ToolRegistry

    registry = ToolRegistry()
    for tool, schema in zip(tools, tool_schemas(tools)):

        async def _executor(_fn: Callable[..., str] = tool, **kwargs: Any) -> str:
            return str(_fn(**kwargs))

        registry.register(
            name=schema["name"],
            description=schema["description"],
            parameters=schema["parameters"],
            executor=_executor,
        )
    return registry


def require_llm_trace(trace: Any) -> Any:
    """Raise if a captured Delegate run never produced a real LLM answer.

    ``Delegate.run`` converts exceptions (daemon down, model missing) into an
    ``ErrorEvent`` instead of raising, so a failed run would otherwise look
    like an empty but "successful" result. This check makes it loud.

    Args:
        trace: The ``AgentTrace`` returned by ``obs.agent.capture_run``.

    Returns:
        The same trace, for chaining.
    """
    llm_errors = [ev.error for ev in trace.events if ev.kind == "error" and not ev.tool]
    if llm_errors:
        raise RuntimeError(f"Agent run failed: {llm_errors[0]}. {OLLAMA_HINT}")
    if not trace.filter_kind("complete"):
        raise RuntimeError(f"Agent run produced no completion event. {OLLAMA_HINT}")
    return trace


def require_agent_result(result: dict, agent_name: str) -> dict:
    """Raise if a BaseAgent ``run_async`` call returned an error dict.

    ``BaseAgent.run_async`` reports provider failures as
    ``{"error": ..., "success": False}`` rather than raising.

    Args:
        result: The dict returned by ``await agent.run_async(...)``.
        agent_name: Label used in the error message.

    Returns:
        The same result dict when the call succeeded.
    """
    if not isinstance(result, dict) or "error" in result or result.get("success") is False:
        detail = result.get("error") if isinstance(result, dict) else repr(result)
        raise RuntimeError(f"{agent_name} LLM call failed: {detail}. {OLLAMA_HINT}")
    return result


def print_tool_registry(tools: list) -> None:
    """Human-readable dump of tool names, schemas, and first-line docs."""
    print("Registered tools:")
    for tool in tools:
        doc_first = (tool.__doc__ or "").strip().split("\n")[0]
        print(f"  {tool.__name__}: {doc_first}")
    schemas = tool_schemas(tools)
    print(f"\nGenerated {len(schemas)} JSON Schema descriptors " f"(first shown):")
    print(json.dumps(schemas[0], indent=2)[:400])


__all__ = [
    "MODEL",
    "OLLAMA_HINT",
    "OUTPUT_DIR",
    "load_hotpotqa",
    "make_tools",
    "tool_schemas",
    "build_tool_registry",
    "require_llm_trace",
    "require_agent_result",
    "print_tool_registry",
    "data_summary",
    "search_documents",
    "run_query",
    "answer_question",
]
