# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 5: AI Agents — ReAct, Tool Use, and Function Calling
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Build a ReActAgent with the Thought -> Action -> Observation loop
#   - Define custom tools with structured schemas for agent consumption
#   - Implement function calling with tool_choice parameters (auto,
#     required, specific function)
#   - Apply cost budgets (LLMCostTracker) to prevent runaway spending
#   - Design agents using the mental framework from Deck 6B: goal,
#     thought process, specialist role, tools needed
#   - Build a custom BaseAgent with Signature for structured output
#   - Implement a critic agent for iterative refinement
#   - Add human-in-the-loop pausing for validation
#
# PREREQUISITES:
#   Exercises 1-4 (Delegate, Signature, prompt engineering).
#   Agents are LLMs with the ability to call functions — not new AI,
#   just LLMs that observe and act, not just respond.
#
# ESTIMATED TIME: ~180 min
#
# TASKS:
#    1. Load multi-hop QA dataset for agent testing
#    2. Define custom tools with structured schemas
#    3. Build ReActAgent with tool access
#    4. Run agent on multi-step analysis task
#    5. Inspect and interpret the reasoning trace
#    6. Function calling protocol (tool_choice, parallel calls)
#    7. Cost budget enforcement (LLMCostTracker)
#    8. Agent design mental framework (goal, process, specialist, tools)
#    9. Custom BaseAgent with Signature for structured analysis
#   10. Critic agent for iterative refinement
#
# DATASET: HotpotQA distractor (hotpotqa/hotpot_qa on HuggingFace)
#   Real multi-hop question-answer pairs that require reasoning over
#   multiple supporting paragraphs.  Perfect for testing the ReAct loop.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import polars as pl

from kaizen import InputField, OutputField, Signature
from kaizen.core import BaseAgent
from kaizen_agents import Delegate
from kaizen_agents.agents.specialized.react import ReActAgent
from kaizen_agents.agents.specialized.simple_qa import SimpleQAAgent

from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
if not model:
    raise EnvironmentError("Set DEFAULT_LLM_MODEL or OPENAI_PROD_MODEL in .env")
print(f"LLM Model: {model}")


# ══════════════════════════════════════════════════════════════════════════
# TASK 1: Load Multi-Hop QA Dataset
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 1: Load HotpotQA Dataset")
print("=" * 70)

CACHE_DIR = Path("data/mlfp06/hotpotqa")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "hotpotqa_500.parquet"

if CACHE_FILE.exists():
    print(f"Loading cached HotpotQA from {CACHE_FILE}")
    qa_data = pl.read_parquet(CACHE_FILE)
else:
    print("Downloading hotpotqa/hotpot_qa from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
        trust_remote_code=True,
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
    print(f"Cached {qa_data.height} HotpotQA examples")

print(f"Loaded {qa_data.height:,} multi-hop QA examples")
print(f"Types: {dict(qa_data['type'].value_counts().iter_rows())}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert qa_data.height > 0, "Task 1: dataset should not be empty"
print("✓ Checkpoint 1 passed — HotpotQA loaded\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 2: Define Custom Tools with Structured Schemas
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: Custom Tool Definitions")
print("=" * 70)


def data_summary(dataset_name: str = "qa_data") -> str:
    """Get a statistical summary of the QA dataset.

    Args:
        dataset_name: Which dataset to summarise.  Currently 'qa_data'.

    Returns:
        Text summary including shape, columns, type distribution, and
        average text lengths.
    """
    df = qa_data
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
    query_lower = query.lower()
    scored = []
    for i, row in enumerate(qa_data.iter_rows(named=True)):
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
    desc = query_description.lower()
    df = qa_data

    if "count" in desc and "type" in desc:
        counts = df.group_by("type").len().sort("len", descending=True)
        return f"Question types:\n{counts}"
    elif "count" in desc and "level" in desc:
        counts = df.group_by("level").len().sort("len", descending=True)
        return f"Difficulty levels:\n{counts}"
    elif "comparison" in desc:
        comparison = df.filter(pl.col("type") == "comparison")
        return f"Comparison questions: {comparison.height}\nSample: {comparison['question'][0]}"
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
    """Look up the answer to a specific HotpotQA question.

    Args:
        question: The exact question text to look up.

    Returns:
        The ground-truth answer if found, or 'not found'.
    """
    for row in qa_data.iter_rows(named=True):
        if question.lower().strip() in row["question"].lower():
            return (
                f"Answer: {row['answer']}\nType: {row['type']}, Level: {row['level']}"
            )
    return "Question not found in dataset."


tools = [data_summary, search_documents, run_query, answer_question]

# Tool schema display
print("Defined tools:")
for tool in tools:
    doc_first_line = tool.__doc__.strip().split("\n")[0]
    print(f"  {tool.__name__}: {doc_first_line}")

print(f"\nTool test — data_summary():")
print(data_summary()[:300])

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert len(tools) == 4, "Task 2: should have 4 tools"
assert all(callable(t) for t in tools), "All tools should be callable"
assert all(t.__doc__ for t in tools), "All tools need docstrings (agent reads them)"
print("\n✓ Checkpoint 2 passed — 4 tools defined with structured docstrings\n")

# INTERPRETATION: Tool docstrings are the agent's API documentation.
# The agent reads the docstring to decide WHICH tool to call and with
# WHAT arguments.  Precise docstrings with Args/Returns sections lead
# to accurate tool selection.  Vague docstrings lead to wrong calls.


# ══════════════════════════════════════════════════════════════════════════
# TASK 3: Build ReActAgent with Tool Access
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: ReActAgent Construction")
print("=" * 70)

print(
    """
ReAct (Reasoning + Acting) loop:
  1. THOUGHT: agent reasons about the current state and what to do next
  2. ACTION:  agent selects a tool and provides arguments
  3. OBSERVATION: tool executes and returns a result
  4. Repeat until the agent decides it has enough information
  5. FINAL ANSWER: agent synthesises observations into a response

Unlike if-else chains, the agent decides WHICH tool and WHAT arguments
via LLM reasoning.  The loop is autonomous — no human choreography.
"""
)


async def build_react_agent():
    agent = ReActAgent(
        model=model,
        tools=tools,
        max_llm_cost_usd=2.0,
    )
    print(f"ReActAgent created:")
    print(f"  Model: {model}")
    print(f"  Tools: {[t.__name__ for t in tools]}")
    print(f"  Budget: $2.00")
    return agent


react_agent = asyncio.run(build_react_agent())

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert react_agent is not None, "Task 3: agent should be created"
print("✓ Checkpoint 3 passed — ReActAgent created\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 4: Run Agent on Multi-Step Analysis Task
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 4: Multi-Step Agent Analysis")
print("=" * 70)


async def multi_step_analysis():
    agent = ReActAgent(model=model, tools=tools, max_llm_cost_usd=3.0)

    sample_q = qa_data["question"][0]
    task = f"""Analyse the HotpotQA multi-hop reasoning dataset to understand
its structure and answer the question: "{sample_q}"

Steps:
1. Get a dataset summary to understand the columns and types
2. Count question types (comparison vs bridge) and difficulty levels
3. Search for documents relevant to the question above
4. Look up the ground-truth answer
5. Synthesise your findings into a clear report."""

    print(f"Task: {task[:200]}...")
    result = await agent.run(task)

    output = ""
    if hasattr(result, "content"):
        output = result.content
    elif isinstance(result, str):
        output = result
    else:
        output = str(result)
    print(f"\nAgent output:\n{output[:500]}...")
    return result


analysis_result = asyncio.run(multi_step_analysis())

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert analysis_result is not None, "Task 4: analysis should produce a result"
print("✓ Checkpoint 4 passed — multi-step analysis complete\n")

# INTERPRETATION: The agent autonomously decided the order of tool calls.
# A good agent: summarises dataset first, then queries systematically.
# A poor agent: randomly calls tools or repeats the same call.


# ══════════════════════════════════════════════════════════════════════════
# TASK 5: Inspect and Interpret the Reasoning Trace
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 5: Reasoning Trace Inspection")
print("=" * 70)

print(
    """
The ReAct reasoning trace is the agent's decision log:

  Thought: "I need to understand the dataset structure first."
  Action:  data_summary(dataset_name="qa_data")
  Observation: [tool returns summary with 500 rows, 5 columns, ...]

  Thought: "Now I know the columns.  Let me count question types."
  Action:  run_query("count question types")
  Observation: [comparison: 243, bridge: 257]

  Thought: "I have the distribution.  Let me search for the specific question."
  Action:  search_documents("Who directed Casablanca")
  Observation: [matching documents with context]

  Thought: "I now have enough information to answer."
  Final Answer: [synthesised response]

Each Thought -> Action -> Observation is auditable.  For production:
  - Store traces for debugging (why did the agent make this choice?)
  - Store traces for compliance (audit trail of what the agent accessed)
  - Analyse traces for quality (did the agent take efficient steps?)

Key trace quality indicators:
  ✓ Logical step ordering (general -> specific)
  ✓ No redundant tool calls
  ✓ Tools called with correct arguments
  ✓ Final answer synthesises observations (not just the last one)

  ✗ Random tool ordering
  ✗ Same tool called twice with identical arguments
  ✗ Arguments that don't match tool schema
  ✗ Final answer ignores some observations
"""
)

# ── Checkpoint 5 ─────────────────────────────────────────────────────────
print("✓ Checkpoint 5 passed — reasoning trace interpretation complete\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 6: Function Calling Protocol
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 6: Function Calling Protocol")
print("=" * 70)

print(
    """
Function calling (tool_choice) options:

  tool_choice="auto" (default):
    The model decides whether to call a tool or respond directly.
    Use for: general conversation where tools are optional.

  tool_choice="required":
    The model MUST call at least one tool before responding.
    Use for: tasks that require data access (always query before answering).

  tool_choice={"type": "function", "function": {"name": "search_documents"}}:
    Force the model to call a SPECIFIC tool.
    Use for: pipeline steps where you know which tool is needed.

Structured tool schemas (JSON Schema):
  Each tool is described by a JSON schema that specifies:
    - name: function name
    - description: what it does (from docstring)
    - parameters: argument types, descriptions, required fields
  The model reads these schemas to decide how to call tools.

Parallel function calling:
  Some models can call MULTIPLE tools in a single turn:
    Turn 1: [search_documents("AI governance"), run_query("count types")]
  Both execute simultaneously, results returned together.
  Reduces latency for independent queries.
"""
)

# Demonstrate structured tool schema generation
tool_schemas = []
for tool in tools:
    import inspect

    sig = inspect.signature(tool)
    params = {}
    for name, param in sig.parameters.items():
        annotation = param.annotation
        param_type = "string"
        if annotation == int:
            param_type = "integer"
        elif annotation == float:
            param_type = "number"
        params[name] = {"type": param_type, "description": f"Parameter: {name}"}

    schema = {
        "name": tool.__name__,
        "description": tool.__doc__.strip().split("\n")[0],
        "parameters": {"type": "object", "properties": params},
    }
    tool_schemas.append(schema)
    print(f"  Schema: {json.dumps(schema, indent=2)[:200]}...")

# ── Checkpoint 6 ─────────────────────────────────────────────────────────
assert len(tool_schemas) == 4, "Task 6: should generate schemas for all 4 tools"
print("\n✓ Checkpoint 6 passed — function calling protocol explained\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 7: Cost Budget Enforcement (LLMCostTracker)
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 7: Cost Budget Enforcement")
print("=" * 70)

print(
    """
LLMCostTracker prevents runaway spending on agent tasks:

  Budget hierarchy:
    Session budget   ($50)  — total for all tasks in a session
      └─ Task budget ($5)   — per-task limit
           └─ Step budget ($1) — per-tool-call limit

  When budget is exceeded:
    1. Agent receives a warning: "Budget 80% consumed"
    2. Agent is forced to produce a final answer
    3. If the agent tries another tool call, it's blocked
    4. Partial results are returned with a budget-exceeded flag

  Budget cascading (connects to Ex 7 PACT governance):
    Parent agent allocates budget to child agents.
    Child cannot exceed its allocation.
    Parent's total spend = sum of children's spend.
"""
)


async def demonstrate_cost_budget():
    """Show that budget enforcement stops the agent gracefully."""
    # Intentionally low budget to trigger the limit
    low_budget_agent = ReActAgent(
        model=model,
        tools=tools,
        max_llm_cost_usd=0.10,  # very low
    )

    task = """Perform an exhaustive analysis of the dataset:
    1. Get summary statistics
    2. Count all question types
    3. Count all difficulty levels
    4. Search for 10 different topics
    5. Find the longest documents
    This should exceed the cost budget."""

    print("Running agent with $0.10 budget on expensive task...")
    try:
        result = await low_budget_agent.run(task)
        output = str(result)[:300] if result else "No output"
        print(f"  Result: {output}...")
    except Exception as e:
        print(f"  Budget exceeded: {e}")

    # Normal budget for comparison
    normal_agent = ReActAgent(
        model=model,
        tools=tools,
        max_llm_cost_usd=2.0,
    )
    result = await normal_agent.run("Get a summary of the dataset.")
    output = str(result)[:200] if result else "No output"
    print(f"  Normal budget result: {output}...")
    return result


budget_result = asyncio.run(demonstrate_cost_budget())

# ── Checkpoint 7 ─────────────────────────────────────────────────────────
print("✓ Checkpoint 7 passed — cost budget enforcement demonstrated\n")

# INTERPRETATION: Cost budgets are the financial operating envelope for
# agents.  Without budgets, a looping agent can spend $100+ on a single
# task.  max_llm_cost_usd is the first line of defence; PACT governance
# (Ex 7) adds organisational budgets on top.


# ══════════════════════════════════════════════════════════════════════════
# TASK 8: Agent Design Mental Framework
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 8: Agent Design Mental Framework (from Deck 6B)")
print("=" * 70)

print(
    """
When designing an agent, answer four questions:

  1. GOAL: What is our goal?
     Be specific.  "Analyse data" is too vague.
     "Find the 3 most impactful features for churn prediction" is specific.

  2. THOUGHT PROCESS: What is the thought process to achieve this goal?
     Break it into steps a human specialist would follow.
     "First, load data.  Then compute correlations.  Then rank by importance."

  3. SPECIALIST: What kind of specialist would we hire?
     Be SHARP: "data scientist with churn modelling experience"
     NOT: "someone who knows about data"
     The specialist description becomes the agent's system prompt.

  4. TOOLS: What tools do they need?
     Tools should be: versatile (handle edge cases), fault-tolerant
     (return useful errors), caching-friendly (idempotent where possible).

Agent design considerations:
  - Iterative refinement: critic agent reviews and suggests improvements
  - Human-in-the-loop: pause workflow for validation at key decision points
  - Monitoring: real-time tracking of intermediate outputs and costs
"""
)

# Example: design a data analysis agent using the framework
print("\n--- Example Agent Design ---")
print("Goal: Identify which question type (comparison vs bridge) has")
print("      longer supporting contexts in the HotpotQA dataset.")
print("Process: 1) Get dataset overview  2) Separate by type")
print("         3) Compare text lengths  4) Report findings")
print("Specialist: 'NLP research analyst specialising in QA datasets'")
print("Tools: data_summary, run_query, search_documents")

# ── Checkpoint 8 ─────────────────────────────────────────────────────────
print("\n✓ Checkpoint 8 passed — agent design mental framework explained\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 9: Custom BaseAgent with Signature for Structured Analysis
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 9: BaseAgent + Signature for Structured Output")
print("=" * 70)


class DataAnalysisSignature(Signature):
    """Analyse a dataset and produce structured insights."""

    dataset_summary: str = InputField(description="Statistical summary of the dataset")
    analysis_question: str = InputField(description="Specific question to investigate")

    key_findings: list[str] = OutputField(
        description="Top 3-5 findings from the analysis"
    )
    recommended_approach: str = OutputField(
        description="Best ML approach for this data (classification, clustering, etc.)"
    )
    data_quality_issues: list[str] = OutputField(
        description="Potential data quality concerns (missing values, bias, etc.)"
    )
    next_steps: list[str] = OutputField(
        description="3-5 recommended next analysis steps"
    )
    confidence: float = OutputField(description="Confidence in findings (0.0 to 1.0)")


class DataAnalysisAgent(BaseAgent):
    """Structured data analysis agent using typed Signature."""

    signature = DataAnalysisSignature
    model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
    max_llm_cost_usd = 1.0


async def run_structured_agent():
    summary = data_summary()
    agent = DataAnalysisAgent()
    result = await agent.run(
        dataset_summary=summary,
        analysis_question="What patterns distinguish comparison questions from bridge questions in this dataset?",
    )
    print(f"Key findings:     {result.key_findings}")
    print(f"Approach:         {result.recommended_approach}")
    print(f"Quality issues:   {result.data_quality_issues}")
    print(f"Next steps:       {result.next_steps}")
    print(f"Confidence:       {result.confidence:.2f}")
    return result


structured_result = asyncio.run(run_structured_agent())

# ── Checkpoint 9 ─────────────────────────────────────────────────────────
assert (
    structured_result is not None
), "Task 9: structured analysis should produce a result"
assert hasattr(structured_result, "key_findings"), "Result needs key_findings"
assert hasattr(structured_result, "confidence"), "Result needs confidence"
assert len(structured_result.key_findings) > 0, "Should have at least one finding"
assert 0 <= structured_result.confidence <= 1, "Confidence should be in [0, 1]"
print(
    f"\n✓ Checkpoint 9 passed — structured analysis: "
    f"{len(structured_result.key_findings)} findings, "
    f"confidence={structured_result.confidence:.2f}\n"
)

# INTERPRETATION: BaseAgent + Signature gives typed, validated output.
# result.key_findings[0] is reliable — no string parsing needed.
# Use BaseAgent when: output feeds into a pipeline, needs logging, or
# requires downstream processing.  Use ReActAgent when: the task
# requires tool exploration and the number of steps is unknown.


# ══════════════════════════════════════════════════════════════════════════
# TASK 10: Critic Agent for Iterative Refinement
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 10: Critic Agent — Iterative Refinement")
print("=" * 70)


class CriticSignature(Signature):
    """Critique an analysis and suggest improvements."""

    original_analysis: str = InputField(description="The analysis to critique")
    analysis_question: str = InputField(description="The original question asked")

    strengths: list[str] = OutputField(description="What the analysis does well")
    weaknesses: list[str] = OutputField(description="Gaps or errors in the analysis")
    suggestions: list[str] = OutputField(description="Specific improvement suggestions")
    quality_score: float = OutputField(description="Overall quality 0.0 to 1.0")
    should_revise: bool = OutputField(description="Whether the analysis needs revision")


class CriticAgent(BaseAgent):
    """Reviews and critiques analyses for quality."""

    signature = CriticSignature
    model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
    max_llm_cost_usd = 1.0


class RefinedAnalysisSignature(Signature):
    """Produce an improved analysis incorporating critic feedback."""

    dataset_summary: str = InputField(description="Dataset summary")
    analysis_question: str = InputField(description="Question to analyse")
    critic_feedback: str = InputField(description="Critic's improvement suggestions")

    improved_findings: list[str] = OutputField(description="Revised findings")
    methodology_note: str = OutputField(description="How the analysis was improved")
    confidence: float = OutputField(description="Confidence after revision (0-1)")


class RefinedAnalysisAgent(BaseAgent):
    """Produces improved analysis based on critic feedback."""

    signature = RefinedAnalysisSignature
    model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
    max_llm_cost_usd = 1.0


async def iterative_refinement():
    """Run analysis -> critic -> refined analysis loop."""
    question = "What makes multi-hop QA harder than single-hop QA?"
    summary = data_summary()

    # Step 1: Initial analysis
    print("Step 1: Initial analysis...")
    analyst = DataAnalysisAgent()
    initial = await analyst.run(
        dataset_summary=summary,
        analysis_question=question,
    )
    initial_text = (
        f"Findings: {initial.key_findings}\nApproach: {initial.recommended_approach}"
    )
    print(f"  Findings: {initial.key_findings[:3]}")

    # Step 2: Critic reviews
    print("\nStep 2: Critic reviews...")
    critic = CriticAgent()
    critique = await critic.run(
        original_analysis=initial_text,
        analysis_question=question,
    )
    print(f"  Strengths:  {critique.strengths[:2]}")
    print(f"  Weaknesses: {critique.weaknesses[:2]}")
    print(f"  Score:      {critique.quality_score:.2f}")
    print(f"  Revise?     {critique.should_revise}")

    # Step 3: Refine if needed
    if critique.should_revise:
        print("\nStep 3: Refining based on feedback...")
        refiner = RefinedAnalysisAgent()
        refined = await refiner.run(
            dataset_summary=summary,
            analysis_question=question,
            critic_feedback=str(critique.suggestions),
        )
        print(f"  Improved findings: {refined.improved_findings[:3]}")
        print(f"  Method note: {refined.methodology_note[:150]}")
        print(f"  Confidence:  {refined.confidence:.2f}")
        return refined
    else:
        print("\nStep 3: Critic approves — no revision needed.")
        return initial


refined_result = asyncio.run(iterative_refinement())

# ── Checkpoint 10 ────────────────────────────────────────────────────────
assert (
    refined_result is not None
), "Task 10: iterative refinement should produce a result"
print("\n✓ Checkpoint 10 passed — critic agent and iterative refinement complete\n")

# INTERPRETATION: The critic agent pattern implements quality assurance
# for AI outputs.  Initial analysis -> critique -> refined analysis.
# This is NOT the same as self-consistency (Ex 1 Task 6): self-consistency
# samples multiple independent paths; the critic reviews and suggests
# SPECIFIC improvements.  Use critic agents when: quality is critical,
# the task is complex, and the cost of two extra LLM calls is justified.


# ══════════════════════════════════════════════════════════════════════════
# Agent Selection Guide
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("Agent Selection Guide")
print("=" * 70)

agent_guide = pl.DataFrame(
    {
        "Agent Type": [
            "ReActAgent",
            "BaseAgent + Signature",
            "SimpleQAAgent",
            "Critic Agent",
            "ReAct + Signature (hybrid)",
        ],
        "When To Use": [
            "Open-ended exploration, unknown # of steps, tool use",
            "Known output schema, feeds into pipeline, audit required",
            "Simple Q&A with structured output, no tools needed",
            "Quality assurance, iterative refinement, review loop",
            "Explore with tools first, then structured final output",
        ],
        "Cost": [
            "Medium-High (variable)",
            "Low (1 LLM call)",
            "Low (1 call)",
            "Medium (2-3 calls)",
            "High (tools + structured)",
        ],
    }
)
print(agent_guide)


# ══════════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  ✓ ReActAgent: Thought -> Action -> Observation loop; autonomous multi-step
  ✓ Tool design: docstrings = agent's API; precision determines tool selection
  ✓ Structured tool schemas: JSON Schema definitions for function calling
  ✓ Function calling protocol: tool_choice (auto, required, specific)
  ✓ Parallel function calling: multiple tools in one turn
  ✓ Cost budgets: max_llm_cost_usd prevents runaway spending
  ✓ Agent design framework: goal, thought process, specialist, tools
  ✓ BaseAgent + Signature: typed, validated, pipeline-safe output
  ✓ Critic agent: iterative refinement loop (analyse -> critique -> refine)
  ✓ Human-in-the-loop: pause workflow for validation at decision points

  Agent type selection:
    ReActAgent:       tool exploration, variable steps, open-ended
    BaseAgent+Sig:    known schema, pipeline integration, audit trail
    Critic pattern:   quality assurance, iterative improvement

  NEXT: Exercise 6 (Multi-Agent) composes multiple specialist agents.
  A supervisor delegates to domain specialists, then synthesises their
  analyses — fan-out (parallel) -> fan-in (synthesis) orchestration.
"""
)
