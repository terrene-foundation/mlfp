# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP06 — Assessment Task 1: Schema-Constrained Extraction (Reference Solution)

Reference implementation. Withheld from students. Verified to pass grader.py.
Uses a Kaizen Signature + BaseAgent wired to the local Ollama daemon at
temperature 0 for deterministic, type-safe structured output.
"""
from __future__ import annotations

import asyncio

from kaizen import InputField, OutputField, Signature
from kaizen.core.base_agent import BaseAgent

from shared.mlfp06._ollama_bootstrap import DEFAULT_CHAT_MODEL, OLLAMA_BASE_URL

# ════════════════════════════════════════════════════════════════════════
# FIXED CORPUS — six SG last-mile logistics incident reports.
# Each report states the five fields explicitly; extraction is deterministic.
# ════════════════════════════════════════════════════════════════════════
INCIDENT_REPORTS: list[str] = [
    (
        "Incident Report INC-3001\n"
        "Severity: HIGH. Location: Tuas Checkpoint.\n"
        "A container truck overturned during transfer. 42 parcels affected. "
        "An insurance claim is required for the damaged goods."
    ),
    (
        "Incident Report INC-3002\n"
        "Severity: LOW. Location: Changi Airfreight Centre.\n"
        "A scanning belt jammed briefly. 3 parcels affected. "
        "No insurance claim is needed."
    ),
    (
        "Incident Report INC-3003\n"
        "Severity: MEDIUM. Location: Jurong Port.\n"
        "A forklift clipped a pallet stack. 17 parcels affected. "
        "An insurance claim is required."
    ),
    (
        "Incident Report INC-3004\n"
        "Severity: HIGH. Location: Woodlands Checkpoint.\n"
        "A refrigeration unit failed in transit. 58 parcels affected. "
        "An insurance claim is required for the spoiled shipment."
    ),
    (
        "Incident Report INC-3005\n"
        "Severity: LOW. Location: Pasir Panjang Terminal.\n"
        "A label printer ran out of ink. 1 parcel affected. "
        "No insurance claim is needed."
    ),
    (
        "Incident Report INC-3006\n"
        "Severity: MEDIUM. Location: Tampines Logistics Hub.\n"
        "A delivery rider was rerouted by road closures. 9 parcels affected. "
        "No insurance claim is needed."
    ),
]


class IncidentExtraction(Signature):
    """Extract structured fields from a last-mile logistics incident report."""

    report_text: str = InputField(description="Raw incident report text")

    incident_id: str = OutputField(
        description="The incident reference id, e.g. INC-3001"
    )
    severity: str = OutputField(description="Exactly one of: low, medium, high")
    location: str = OutputField(
        description="The location or facility named in the report"
    )
    parcels_affected: int = OutputField(
        description="Number of parcels affected (an integer)"
    )
    claim_required: bool = OutputField(
        description="True if an insurance claim is required, otherwise False"
    )


def _make_agent() -> BaseAgent:
    class Extractor(BaseAgent):
        def __init__(self) -> None:
            super().__init__(
                config={
                    "model": DEFAULT_CHAT_MODEL,
                    "llm_provider": "ollama",
                    "base_url": OLLAMA_BASE_URL,
                    "use_async_llm": True,
                    "temperature": 0.0,
                },
                signature=IncidentExtraction(),
            )

    return Extractor()


async def _extract_all() -> list[dict]:
    results: list[dict] = []
    for report in INCIDENT_REPORTS:
        agent = _make_agent()
        out = await agent.run_async(report_text=report)
        results.append(
            {
                "incident_id": out.get("incident_id"),
                "severity": out.get("severity"),
                "location": out.get("location"),
                "parcels_affected": out.get("parcels_affected"),
                "claim_required": out.get("claim_required"),
            }
        )
    return results


def solve() -> list[dict]:
    """Extract a structured record from each of the six incident reports.

    Returns a list of six dicts, each with keys: incident_id, severity,
    location, parcels_affected, claim_required.
    """
    return asyncio.run(_extract_all())


if __name__ == "__main__":
    for rec in solve():
        print(rec)
