# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT5 — Exercise 4: RAGResearchAgent
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build RAGResearchAgent over Kailash SDK documentation and
#   Singapore regulatory docs. Evaluate retrieval quality.
#
# TASKS:
#   1. Prepare document corpus (SDK docs + regulatory)
#   2. Build RAGResearchAgent with document retrieval
#   3. Evaluate retrieval quality (faithfulness, relevance)
#   4. Compare dense vs hybrid retrieval
#   5. Demonstrate RAGAS evaluation metrics
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl
from dotenv import load_dotenv

from kaizen import Signature, InputField, OutputField
from kaizen_agents.agents.specialized.rag_research import RAGResearchAgent

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Prepare document corpus
# ══════════════════════════════════════════════════════════════════════

# Simulated document corpus (in production, load from ascent05/ data)
sdk_docs = [
    {
        "title": "DataExplorer — Automated Data Profiling",
        "content": "DataExplorer provides async profiling of polars DataFrames. Key methods: profile() returns DataProfile with column statistics, correlation matrices (Pearson, Spearman, Cramer's V), and data quality alerts. AlertConfig lets you configure thresholds for 8 alert types: high_nulls, constant, high_skewness, high_zeros, high_cardinality, high_correlation, duplicates, imbalanced.",
        "source": "kailash-ml docs",
    },
    {
        "title": "TrainingPipeline — Model Training Lifecycle",
        "content": "TrainingPipeline orchestrates model training with ModelSpec (model class + hyperparameters), EvalSpec (metrics, split strategy), and FeatureSchema. Supports sklearn, LightGBM, and PyTorch Lightning frameworks. Integrated with ExperimentTracker for logging and ModelRegistry for artifact storage.",
        "source": "kailash-ml docs",
    },
    {
        "title": "DriftMonitor — Production Model Monitoring",
        "content": "DriftMonitor detects data drift using PSI (Population Stability Index) and KS (Kolmogorov-Smirnov) tests. PSI > 0.1 indicates moderate drift, > 0.2 indicates severe drift. DriftSpec configures per-feature thresholds. Integrates with ModelRegistry for model lifecycle management.",
        "source": "kailash-ml docs",
    },
    {
        "title": "Singapore AI Verify Framework",
        "content": "AI Verify is Singapore's AI governance testing framework. It assesses AI systems across 11 principles: transparency, explainability, fairness, human agency, robustness, safety, accountability, data governance, inclusive growth, environmental well-being, and interoperability. ISAGO 2.0 provides process-based governance assessment.",
        "source": "IMDA Singapore",
    },
    {
        "title": "MAS Guidelines on AI in Financial Services",
        "content": "The Monetary Authority of Singapore requires financial institutions to ensure AI models are fair, ethical, accountable, and transparent (FEAT). Key requirements: model risk management, ongoing monitoring, explainability for customer-facing decisions, and regular model validation. Enhanced scrutiny for credit scoring and anti-money laundering models.",
        "source": "MAS Guidelines",
    },
    {
        "title": "PACT Governance Framework",
        "content": "PACT (Principles for Accountable Compute Trust) uses D/T/R (Department/Team/Role) addressing for access control. GovernanceEngine enforces operating envelopes with monotonic tightening — child contexts cannot exceed parent permissions. GovernanceContext is a frozen dataclass: agents receive but cannot modify their governance.",
        "source": "kailash-pact docs",
    },
]

documents = [d["content"] for d in sdk_docs]
titles = [d["title"] for d in sdk_docs]

print(f"=== Document Corpus ===")
print(f"Documents: {len(documents)}")
for d in sdk_docs:
    print(f"  [{d['source']}] {d['title']}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Build RAGResearchAgent
# ══════════════════════════════════════════════════════════════════════


class RAGAnswer(Signature):
    """RAG-based answer with source attribution."""

    question: str = InputField(description="User question")

    answer: str = OutputField(description="Answer grounded in retrieved documents")
    sources: list[str] = OutputField(description="Source document titles used")
    confidence: float = OutputField(description="Confidence 0-1 in answer quality")
    retrieval_quality: str = OutputField(
        description="Assessment of retrieval relevance"
    )


async def build_rag_agent():
    agent = RAGResearchAgent(
        signature=RAGAnswer,
        model=model,
        documents=documents,
        max_llm_cost_usd=3.0,
    )

    questions = [
        "How does Kailash detect data drift in production models?",
        "What does Singapore's AI Verify framework require for financial AI?",
        "How does PACT ensure agents cannot exceed their governance boundaries?",
    ]

    results = []
    for q in questions:
        result = await agent.run(question=q)

        print(f"\n=== Q: {q} ===")
        print(f"A: {result.answer[:300]}...")
        print(f"Sources: {result.sources}")
        print(f"Confidence: {result.confidence}")
        print(f"Retrieval quality: {result.retrieval_quality}")
        results.append(result)

    return results


rag_results = asyncio.run(build_rag_agent())


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Evaluate retrieval quality (RAGAS-style)
# ══════════════════════════════════════════════════════════════════════


# RAGAS metrics (simplified implementation)
def evaluate_faithfulness(answer: str, context: str) -> float:
    """Are all claims in the answer supported by the context?"""
    answer_sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not answer_sentences:
        return 1.0
    supported = sum(
        1
        for s in answer_sentences
        if any(word.lower() in context.lower() for word in s.split() if len(word) > 4)
    )
    return supported / len(answer_sentences)


def evaluate_relevance(question: str, answer: str) -> float:
    """Does the answer address the question?"""
    q_words = set(question.lower().split())
    a_words = set(answer.lower().split())
    overlap = len(q_words & a_words) / max(len(q_words), 1)
    return min(overlap * 2, 1.0)  # Scale up


print(f"\n=== RAGAS-style Evaluation ===")
questions = [
    "How does Kailash detect data drift in production models?",
    "What does Singapore's AI Verify framework require for financial AI?",
    "How does PACT ensure agents cannot exceed their governance boundaries?",
]

for i, (q, r) in enumerate(zip(questions, rag_results)):
    relevant_docs = " ".join(
        d["content"] for d in sdk_docs if any(t in d["title"] for t in r.sources)
    )
    faith = evaluate_faithfulness(r.answer, relevant_docs) if relevant_docs else 0.0
    relev = evaluate_relevance(q, r.answer)

    print(f"\nQ{i+1}: {q[:60]}...")
    print(f"  Faithfulness: {faith:.3f}")
    print(f"  Relevance:    {relev:.3f}")
    print(f"  Self-rated confidence: {r.confidence:.3f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Retrieval strategy comparison
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Retrieval Strategy Comparison ===")
print(
    """
Dense retrieval (embedding similarity):
  + Captures semantic meaning ("data drift" matches "distribution shift")
  - Expensive to compute, requires embedding model
  - Can hallucinate relevance for semantically similar but factually unrelated docs

Sparse retrieval (BM25/TF-IDF):
  + Fast, exact keyword matching
  + No embedding model needed
  - Misses synonyms ("drift" won't match "shift")
  - No semantic understanding

Hybrid (dense + sparse):
  + Best of both worlds
  + Re-ranking with cross-encoder for precision
  - Most complex, highest latency
  → This is what production RAG systems use
"""
)


# ══════════════════════════════════════════════════════════════════════
# TASK 5: RAGAS metrics explained
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== RAGAS Framework (Industry Standard) ===")
print(
    """
1. Faithfulness: Are claims in the answer grounded in retrieved context?
   - Low faithfulness = model is hallucinating beyond the documents
   - Fix: better retrieval, stricter prompting, Corrective RAG

2. Answer Relevance: Does the answer actually address the question?
   - Low relevance = model is regurgitating context without answering
   - Fix: better question-answer alignment in the prompt

3. Context Precision: Is the retrieved context relevant to the question?
   - Low precision = retriever is pulling irrelevant documents
   - Fix: better embeddings, re-ranking, hybrid retrieval

4. Context Recall: Does the retrieved context cover all needed info?
   - Low recall = important documents are being missed
   - Fix: more documents, better chunking, parent-document retrieval

Your RAG system is only as good as its weakest metric.
High relevance + low faithfulness = confident hallucination (dangerous!)
"""
)

print("\n✓ Exercise 4 complete — RAGResearchAgent with retrieval evaluation")
