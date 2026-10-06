# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP06 — Assessment Task 2: A Governed Agent's Tools and Operating Config
(Reference Solution)

Withheld from students. Verified to pass grader.py across seeds. No LLM calls.

A tool exists for the model only once it is registered with name +
description + JSON-schema parameters + an async executor; and the governed
config is set at construction — budget and clearance live on the envelope,
not on a separate tracker. "internal" is kaizen-agents' alias for pact's
RESTRICTED rung.
"""
from __future__ import annotations

import json

from shared.mlfp06._ollama_bootstrap import DEFAULT_CHAT_MODEL

TOOL_NAMES = ["compute_statistics", "normalise_text", "convert_currency"]


def build_tools():
    from kaizen_agents.delegate.loop import ToolRegistry

    async def compute_statistics(numbers: list[float]) -> str:
        vals = [float(v) for v in numbers]
        return json.dumps(
            {
                "count": len(vals),
                "mean": round(sum(vals) / len(vals), 4) if vals else 0.0,
                "min": min(vals) if vals else 0.0,
                "max": max(vals) if vals else 0.0,
            }
        )

    async def normalise_text(text: str) -> str:
        return json.dumps({"normalised": " ".join(str(text).split()).lower()})

    async def convert_currency(amount: float, rate: float) -> str:
        return json.dumps({"converted": round(float(amount) * float(rate), 2)})

    registry = ToolRegistry()
    registry.register(
        name="compute_statistics",
        description="Count, mean, min and max of a list of numbers.",
        parameters={
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "the numbers to summarise",
                }
            },
            "required": ["numbers"],
        },
        executor=compute_statistics,
    )
    registry.register(
        name="normalise_text",
        description="Lowercase a string and collapse its whitespace.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "the text to normalise"}
            },
            "required": ["text"],
        },
        executor=normalise_text,
    )
    registry.register(
        name="convert_currency",
        description="Convert an amount by an exchange rate (2 dp).",
        parameters={
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "source amount"},
                "rate": {"type": "number", "description": "exchange rate"},
            },
            "required": ["amount", "rate"],
        },
        executor=convert_currency,
    )
    return registry


def build_agent(registry):
    from kaizen_agents import GovernedSupervisor

    return GovernedSupervisor(
        model=DEFAULT_CHAT_MODEL,
        budget_usd=0.25,
        tools=list(TOOL_NAMES),
        data_clearance="internal",
    )


if __name__ == "__main__":
    import asyncio

    reg = build_tools()
    print("registered:", reg.tool_names)
    print(asyncio.run(reg.execute("compute_statistics", {"numbers": [1, 2, 3, 4]})))
    print(asyncio.run(reg.execute("normalise_text", {"text": "  Hello   WORLD "})))
    print(asyncio.run(reg.execute("convert_currency", {"amount": 10.0, "rate": 0.74})))
    agent = build_agent(reg)
    print("clearance:", agent.envelope.confidentiality_clearance)
    print("budget:", agent.envelope.financial.max_spend_usd)
    print("tools:", agent.envelope.operational.allowed_actions)
