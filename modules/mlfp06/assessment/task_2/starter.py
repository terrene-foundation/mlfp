# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP06 — Assessment Task 2: A Governed Agent's Tools and Operating Config

Implement `build_tools()` and `build_agent()`. problem.md holds the three
tool contracts and the agent config. The grader calls your executors with its
own inputs and reads your agent's envelope — canned outputs fail.

    python starter.py               # (you) build + smoke-test
    python grader.py starter.py     # (instructor) grade an attempt

No LLM is involved in this task.
"""
from __future__ import annotations

TOOL_NAMES = ["compute_statistics", "normalise_text", "convert_currency"]


def build_tools():
    """Return a kaizen ToolRegistry with the three tools from problem.md
    registered (name, description, JSON-schema parameters, async executor)."""
    from kaizen_agents.delegate.loop import ToolRegistry

    registry = ToolRegistry()
    raise NotImplementedError("Implement build_tools() — see problem.md")


def build_agent(registry):
    """Return a GovernedSupervisor with the config from problem.md
    (budget_usd=0.25, data_clearance="internal", the three tool names,
    model from the course Ollama bootstrap)."""
    raise NotImplementedError("Implement build_agent() — see problem.md")


if __name__ == "__main__":
    import asyncio

    reg = build_tools()
    print("registered:", reg.tool_names)
    print(asyncio.run(reg.execute("convert_currency", {"amount": 10.0, "rate": 0.74})))
    agent = build_agent(reg)
    env = agent.envelope
    print("clearance:", env.confidentiality_clearance)
    print("budget:", env.financial.max_spend_usd)
    print("tools:", env.operational.allowed_actions)
