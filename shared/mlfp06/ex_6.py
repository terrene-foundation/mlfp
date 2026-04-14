# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for Exercise 6 — Multi-Agent Orchestration and MCP.

Contains: SQuAD 2.0 corpus loading, specialist Signature definitions,
specialist BaseAgent classes, synthesis agent, output directory setup.
Technique-specific orchestration logic lives in the per-technique files.
"""
from __future__ import annotations

import os
from pathlib import Path

import polars as pl

from kaizen import InputField, OutputField, Signature
from kaizen.core import BaseAgent

from shared.kailash_helpers import setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()

MODEL = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
if not MODEL:
    raise EnvironmentError(
        "Set DEFAULT_LLM_MODEL or OPENAI_PROD_MODEL in .env before running "
        "MLFP06 Exercise 6 — every specialist agent needs an LLM model."
    )

# Output directory for all visualisation/trace artifacts
OUTPUT_DIR = Path("outputs") / "ex6_multi_agent"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — SQuAD 2.0 Multi-Domain Corpus
# ════════════════════════════════════════════════════════════════════════

CACHE_DIR = Path("data") / "mlfp06" / "squad"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "squad_v2_300.parquet"


def load_squad_corpus(n_rows: int = 300) -> pl.DataFrame:
    """Load (or download + cache) SQuAD 2.0 validation split.

    SQuAD 2.0 is a multi-domain reading-comprehension benchmark with
    100K+ questions across hundreds of Wikipedia titles. We take a
    shuffled slice of 300 rows so exercises run in bounded time while
    still exercising the "multi-domain" property.

    Returns a polars DataFrame with columns: title, text, question, answer.
    """
    if CACHE_FILE.exists():
        print(f"Loading cached SQuAD 2.0 from {CACHE_FILE}")
        return pl.read_parquet(CACHE_FILE)

    print("Downloading rajpurkar/squad_v2 from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    ds = ds.shuffle(seed=42).select(range(min(n_rows, len(ds))))
    rows = []
    for row in ds:
        answers = row["answers"]["text"]
        rows.append(
            {
                "title": row["title"],
                "text": row["context"],
                "question": row["question"],
                "answer": answers[0] if answers else "",
            }
        )
    passages = pl.DataFrame(rows)
    passages.write_parquet(CACHE_FILE)
    print(f"Cached {passages.height} SQuAD 2.0 rows to {CACHE_FILE}")
    return passages


# ════════════════════════════════════════════════════════════════════════
# SPECIALIST SIGNATURES — Domain-Specific Input/Output Contracts
# ════════════════════════════════════════════════════════════════════════


class FactualAnalysisSignature(Signature):
    """Extract factual claims and verify them against the passage."""

    document: str = InputField(description="Source passage text")
    question: str = InputField(description="Question to analyse")
    factual_claims: list[str] = OutputField(
        description="Key factual claims in the passage"
    )
    evidence_quality: str = OutputField(
        description="Quality of evidence: strong/moderate/weak"
    )
    answer_supported: bool = OutputField(
        description="Whether the passage supports an answer"
    )


class SemanticAnalysisSignature(Signature):
    """Analyse meaning, context, and implications beyond literal text."""

    document: str = InputField(description="Source passage text")
    question: str = InputField(description="Question to analyse")
    main_themes: list[str] = OutputField(description="Central themes in the passage")
    implicit_info: list[str] = OutputField(
        description="Implied but not stated information"
    )
    contextual_relevance: str = OutputField(
        description="How relevant context is to the question"
    )


class StructuralAnalysisSignature(Signature):
    """Analyse the structure, organisation, and argumentation pattern."""

    document: str = InputField(description="Source passage text")
    question: str = InputField(description="Question to analyse")
    structure_type: str = OutputField(
        description="Text structure: narrative/expository/argumentative"
    )
    key_entities: list[str] = OutputField(description="Named entities mentioned")
    relationships: list[str] = OutputField(description="Relationships between entities")


class SynthesisSignature(Signature):
    """Synthesise multiple specialist analyses into a unified answer."""

    document: str = InputField(description="Original passage")
    question: str = InputField(description="Original question")
    factual_analysis: str = InputField(description="Factual specialist output")
    semantic_analysis: str = InputField(description="Semantic specialist output")
    structural_analysis: str = InputField(description="Structural specialist output")
    unified_answer: str = OutputField(
        description="Comprehensive answer drawing on all analyses"
    )
    confidence: float = OutputField(description="Answer confidence 0-1")
    reasoning_chain: list[str] = OutputField(description="Step-by-step reasoning used")


class InterpretationSignature(Signature):
    """Interpret factual claims in context (sequential pipeline stage 2)."""

    factual_claims: str = InputField(description="Raw factual claims from prior stage")
    document: str = InputField(description="Original passage for context")
    question: str = InputField(description="Question being answered")
    interpreted_facts: list[str] = OutputField(
        description="Facts with contextual interpretation"
    )
    relevance_ranking: list[str] = OutputField(
        description="Facts ranked by relevance to question"
    )


# ════════════════════════════════════════════════════════════════════════
# SPECIALIST AGENT CLASSES
# ════════════════════════════════════════════════════════════════════════


class FactualAgent(BaseAgent):
    signature = FactualAnalysisSignature
    model = MODEL
    max_llm_cost_usd = 1.0
    description = "Specialist in factual analysis: claims, evidence, verification"


class SemanticAgent(BaseAgent):
    signature = SemanticAnalysisSignature
    model = MODEL
    max_llm_cost_usd = 1.0
    description = "Specialist in semantic analysis: themes, implications, context"


class StructuralAgent(BaseAgent):
    signature = StructuralAnalysisSignature
    model = MODEL
    max_llm_cost_usd = 1.0
    description = (
        "Specialist in structural analysis: entities, relationships, organisation"
    )


class SynthesisAgent(BaseAgent):
    signature = SynthesisSignature
    model = MODEL
    max_llm_cost_usd = 2.0
    description = (
        "Supervisor that synthesises specialist analyses into unified decisions"
    )


class InterpretationAgent(BaseAgent):
    signature = InterpretationSignature
    model = MODEL
    max_llm_cost_usd = 1.0
    description = "Stage-2 interpreter: contextualises raw factual claims for synthesis"


def build_specialists() -> tuple[FactualAgent, SemanticAgent, StructuralAgent]:
    """Return fresh instances of the three analysis specialists."""
    return FactualAgent(), SemanticAgent(), StructuralAgent()


def build_synthesis() -> SynthesisAgent:
    """Return a fresh synthesis (supervisor) agent."""
    return SynthesisAgent()
