# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP06 — Assessment Task 3: Tool-Using Agent (Reference Solution)

Reference implementation. Withheld from students. Verified to pass grader.py.
A Kaizen Delegate (Ollama, temperature 0) is given four deterministic tools
over the real SST-2 dataset. For each fixed question the agent must SELECT the
correct tool and produce the deterministic answer the tool computes.
"""
from __future__ import annotations

import asyncio

import polars as pl
from kaizen_agents.delegate.loop import ToolRegistry

from shared import MLFPDataLoader
from shared.mlfp06._ollama_bootstrap import make_delegate

# Each question is single-hop: exactly one correct tool answers it.
QUESTIONS: list[str] = [
    "How many reviews are in the dataset in total?",
    "How many reviews have the positive label?",
    "How many reviews have the negative label?",
    "What is the average review length in characters?",
    "What is the sentiment label of the review at index 0?",
]


def _make_tools(df: pl.DataFrame, call_log: list[tuple[str, dict]]):
    """Build four deterministic SST-2 tools that append to ``call_log``."""

    async def dataset_size() -> str:
        call_log.append(("dataset_size", {}))
        return f"The dataset has {df.height} reviews."

    async def count_by_label(label: str) -> str:
        key = str(label).strip().lower()
        n = df.filter(pl.col("label") == key).height
        call_log.append(("count_by_label", {"label": key}))
        return f"There are {n} reviews with label '{key}'."

    async def average_review_length() -> str:
        avg = df.select(pl.col("text").str.len_chars().mean()).item()
        call_log.append(("average_review_length", {}))
        return f"The average review length is {avg:.2f} characters."

    async def get_review_by_index(index: int) -> str:
        try:
            i = int(index)
        except (TypeError, ValueError):
            i = -1
        call_log.append(("get_review_by_index", {"index": i}))
        if 0 <= i < df.height:
            row = df.row(i, named=True)
            return f"Review {i}: label='{row['label']}', text={row['text'][:80]!r}"
        return f"Index {index} is out of range."

    reg = ToolRegistry()
    reg.register(
        name="dataset_size",
        description="Return the total number of reviews in the dataset.",
        parameters={"type": "object", "properties": {}},
        executor=dataset_size,
    )
    reg.register(
        name="count_by_label",
        description="Return how many reviews have a given sentiment label "
        "('positive' or 'negative').",
        parameters={
            "type": "object",
            "properties": {
                "label": {"type": "string", "description": "positive or negative"}
            },
            "required": ["label"],
        },
        executor=count_by_label,
    )
    reg.register(
        name="average_review_length",
        description="Return the average review length in characters across the dataset.",
        parameters={"type": "object", "properties": {}},
        executor=average_review_length,
    )
    reg.register(
        name="get_review_by_index",
        description="Return the sentiment label and text of the review at a given "
        "integer row index.",
        parameters={
            "type": "object",
            "properties": {"index": {"type": "integer", "description": "Row index"}},
            "required": ["index"],
        },
        executor=get_review_by_index,
    )
    return reg


async def _run() -> dict:
    df = MLFPDataLoader().load("mlfp06", "sst2/sst2_200.parquet")
    transcripts: list[dict] = []
    tool_names: list[str] = []
    for question in QUESTIONS:
        call_log: list[tuple[str, dict]] = []
        reg = _make_tools(df, call_log)
        tool_names = reg.tool_names
        delegate = make_delegate(
            model="llama3.2:3b", temperature=0.0, max_tokens=512, tools=reg
        )
        final = ""
        async for event in delegate.run(question):
            if getattr(event, "event_type", None) == "turn_complete":
                final = getattr(event, "text", "") or final
        transcripts.append(
            {
                "question": question,
                "tools_called": [[name, args] for name, args in call_log],
                "answer": final.strip(),
            }
        )
    return {"tool_names": tool_names, "transcripts": transcripts}


def solve() -> dict:
    """Run the tool-using agent over the five fixed questions.

    Returns {"tool_names": [str], "transcripts": [{question, tools_called,
    answer}]} with one transcript per question.
    """
    return asyncio.run(_run())


if __name__ == "__main__":
    out = solve()
    print("tools:", out["tool_names"])
    for t in out["transcripts"]:
        print(
            f"  Q={t['question'][:48]!r:50} tools={t['tools_called']} ans={t['answer'][:60]!r}"
        )
