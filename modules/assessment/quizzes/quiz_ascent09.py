# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""ASCENT Module 9 — AI-Resilient Assessment Questions

LLMs, AI Agents & RAG Systems
Covers: decoder-only transformers, tokenization, KV cache, prompting,
        RAG, Kaizen Delegate/BaseAgent/Signature, ReActAgent,
        multi-agent, MCP, Nexus deployment
"""

QUIZ = {
    "module": "ASCENT09",
    "title": "LLMs, AI Agents & RAG Systems",
    "questions": [
        # ── Section A: LLM Architecture ─────────────────────────────────
        {
            "id": "9.A.1",
            "lesson": "9.A",
            "type": "context_apply",
            "difficulty": "advanced",
            "question": (
                "Exercise 1 calculates KV cache memory for a 7B parameter model with "
                "d_model=4096, 32 layers, 32 heads. At sequence length 4096, your calculation "
                "shows 4 GB for the KV cache alone. A colleague wants to serve 16 concurrent "
                "users with 8K context. How much KV cache memory is needed?"
            ),
            "options": [
                "A) 4 GB × 16 = 64 GB — KV cache scales linearly with concurrent users",
                "B) The 4 GB figure is for 4K context; the precise calculation gives ~2.15 GB at 4K. At 8K context: ~4.3 GB per user. With 16 users: 4.3 × 16 ≈ 69 GB for KV cache alone. This exceeds most single-GPU memory. Solutions: KV cache quantization (fp16 → int8 halves to ~34 GB), paged attention (only allocate used slots), or multi-GPU deployment.",
                "C) 4 GB total — KV cache is shared across all users",
                "D) 4 GB × 2 = 8 GB — each user adds one K and one V, so 2× per user",
            ],
            "answer": "B",
            "explanation": (
                "KV cache per token = 2 (K+V) × n_layers × d_model × dtype_bytes. "
                "For fp16: 2 × 32 × 4096 × 2 = 524,288 bytes per token. "
                "At 4K tokens: 524,288 × 4096 = 2.15 GB. At 8K: 4.3 GB per user. "
                "16 concurrent users × 4.3 GB = 68.7 GB just for KV cache. "
                "Plus model weights: 7B × 2 bytes = 14 GB. Total: ~83 GB. "
                "This is why production LLM serving uses paged attention (vLLM) to share "
                "KV cache pages across sequences and int8 quantization to halve memory."
            ),
            "learning_outcome": "Calculate KV cache memory requirements for concurrent LLM serving",
        },
        {
            "id": "9.A.2",
            "lesson": "9.A",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "Exercise 1 makes the first Delegate call. The student's code runs but "
                "costs $2.50 on a single question. They expected ~$0.01. What is wrong?"
            ),
            "code": (
                "delegate = Delegate(model=model)\n"
                "async for event in delegate.run(\n"
                "    'Here is the entire 50-page annual report: '\n"
                "    + full_report_text  # Bug: sending entire document\n"
                "    + '\\n\\nQuestion: What is the revenue?'\n"
                "):\n"
                "    print(event.content)\n"
            ),
            "options": [
                "A) The model parameter is wrong — use a cheaper model",
                "B) The student is sending the entire 50-page report as input (likely 100K+ tokens). LLM cost scales with token count. Fix: (1) Add max_llm_cost_usd to set a budget cap, (2) summarize or chunk the document first, (3) use RAG (Exercise 3-4) to retrieve only relevant passages instead of sending everything.",
                "C) Delegate should be called with run_async() not run()",
                "D) The async for loop is consuming tokens — use a single response instead",
            ],
            "answer": "B",
            "explanation": (
                "LLM API costs are per-token. A 50-page report is ~50,000-100,000 tokens. "
                "At typical rates ($0.01-0.03/1K tokens), that's $0.50-3.00 per request. "
                "Three fixes: (1) max_llm_cost_usd=0.05 on Delegate prevents overspending. "
                "(2) Document summarization reduces context. (3) RAG retrieves only the "
                "3-5 most relevant paragraphs (~500 tokens) instead of the full document. "
                "This is exactly why Exercises 3-4 exist — RAG is the production solution "
                "to the 'send everything to the LLM' anti-pattern."
            ),
            "learning_outcome": "Apply cost governance with max_llm_cost_usd and use RAG for large documents",
        },
        # ── Section B: Prompting ────────────────────────────────────────
        {
            "id": "9.B.1",
            "lesson": "9.B",
            "type": "context_apply",
            "difficulty": "intermediate",
            "question": (
                "Exercise 2 compares prompting strategies for classifying company reports "
                "into 5 categories. Zero-shot achieves 65%, few-shot (3 examples) achieves "
                "78%, and chain-of-thought achieves 82%. But chain-of-thought costs 3× more "
                "than few-shot. When is chain-of-thought worth the cost?"
            ),
            "options": [
                "A) Always — accuracy is always more important than cost",
                "B) When the task requires multi-step reasoning (like Exercise 2's category decisions involving both content AND tone analysis). For simple classification where the answer is obvious from keywords, few-shot is sufficient. Chain-of-thought pays off when the model needs to reason through ambiguous cases — the 'thinking out loud' helps it resolve contradictions.",
                "C) Never — few-shot at 78% is good enough for any use case",
                "D) Only when max_llm_cost_usd is set high enough to allow the extra tokens",
            ],
            "answer": "B",
            "explanation": (
                "Chain-of-thought (CoT) generates intermediate reasoning steps before the "
                "final answer. This costs 3× more tokens but improves accuracy on tasks "
                "requiring multi-step reasoning: 'This report discusses renewable energy "
                "(→ Technology?) but focuses on investment returns (→ Finance?) and was "
                "published by a regulatory body (→ Regulatory). Final: Regulatory.' "
                "For unambiguous cases ('quarterly earnings call' → Finance), CoT adds cost "
                "without benefit. Exercise 2 shows CoT's gain is concentrated on ambiguous "
                "reports — 95% accuracy on clear cases vs 60% → 82% on ambiguous ones."
            ),
            "learning_outcome": "Choose prompting strategy based on task complexity and cost constraints",
        },
        {
            "id": "9.B.2",
            "lesson": "9.B",
            "type": "architecture_decision",
            "difficulty": "intermediate",
            "question": (
                "Exercise 2 builds a custom Signature for structured extraction. Why use "
                "a Signature with typed OutputFields instead of asking the Delegate for "
                "free-form text and parsing it?"
            ),
            "options": [
                "A) Signatures are faster because they use fewer tokens",
                "B) Signatures define a contract: OutputField types (str, float, list[str]) are guaranteed in the response. Free-form text requires fragile parsing (regex, string splitting) that breaks on format variations. In production pipelines, downstream code depends on structured data — a Signature ensures the agent's output always matches the expected schema.",
                "C) Delegate cannot produce structured output — only Signatures can",
                "D) Signatures automatically validate the factual accuracy of outputs",
            ],
            "answer": "B",
            "explanation": (
                "Signature = API contract between agent and consumer. "
                "OutputField(description='Confidence score 0-1') with type float guarantees "
                "the downstream code receives a float, not 'I am 85% confident'. "
                "Without Signatures, parsing 'The confidence is approximately 0.85 or so' "
                "requires regex that breaks when the LLM varies its phrasing. "
                "In production: database writes, API responses, and UI rendering all need "
                "typed, predictable data. Signatures make agent output machine-consumable. "
                "This is analogous to ModelSignature in ML — specifying input/output schemas."
            ),
            "learning_outcome": "Use Kaizen Signatures for typed, predictable agent output in production",
        },
        # ── Section C: RAG ──────────────────────────────────────────────
        {
            "id": "9.C.1",
            "lesson": "9.C",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "Exercise 3's RAG pipeline retrieves regulation chunks for answering questions. "
                "For 'What is the penalty for non-compliance with AI Act Article 5?', the "
                "retriever returns chunks about Article 3 (definitions), Article 7 (conformity), "
                "and Article 12 (record-keeping) — but NOT Article 5. What is likely wrong?"
            ),
            "options": [
                "A) The embedding model doesn't understand legal text",
                "B) Chunk size is likely too large, causing Article 5 content to be diluted within a chunk that also covers Articles 3-8. When a chunk spans multiple articles, its embedding represents the average meaning, not any specific article. Fix: reduce chunk size and add overlap to ensure Article 5 has its own dedicated chunk.",
                "C) The cosine similarity threshold is set too high — lower it to include more results",
                "D) Vector search cannot handle queries with numbers like 'Article 5'",
            ],
            "answer": "B",
            "explanation": (
                "RAG retrieval quality depends critically on chunk size. If chunks are too large "
                "(e.g., entire chapters), the embedding represents the average of many topics. "
                "A query about Article 5 penalties may match a chunk covering Articles 1-10 "
                "with low specificity. Chunks about other articles with stronger keyword overlap "
                "(Article 12 mentions 'record-keeping penalties') may score higher. "
                "Fix: chunk_size=200-500 tokens with overlap=50 tokens ensures each article "
                "gets at least one dedicated chunk. Exercise 3 demonstrates this by comparing "
                "retrieval quality at different chunk sizes."
            ),
            "learning_outcome": "Diagnose RAG retrieval failures from chunk size misconfiguration",
        },
        {
            "id": "9.C.2",
            "lesson": "9.C",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 4 implements hybrid search (BM25 + vector) and re-ranking. "
                "On the regulation corpus: BM25 alone gets recall@5=0.72, vector alone gets "
                "recall@5=0.68, hybrid gets recall@5=0.85, hybrid+reranking gets recall@5=0.91. "
                "Why does hybrid outperform either method alone?"
            ),
            "options": [
                "A) Hybrid uses more computation, so it naturally performs better",
                "B) BM25 excels at exact keyword matching ('Article 5', 'MAS TRM 7.5') while vector search captures semantic similarity ('punishment for breaking rules' → 'penalties for non-compliance'). They fail on DIFFERENT queries. Hybrid combines both: BM25 catches exact terms vector search misses, and vector search catches paraphrases BM25 misses. The union covers more relevant documents.",
                "C) BM25 and vector search produce identical rankings — hybrid just averages them",
                "D) Re-ranking is responsible for the entire improvement; the hybrid combination adds nothing",
            ],
            "answer": "B",
            "explanation": (
                "BM25 is lexical: it matches exact terms. Strong for: specific references "
                "('Article 5', 'Section 12(b)'), technical terms, names. Weak for: paraphrases, "
                "synonyms ('fine' vs 'penalty'). Vector search is semantic: it matches meaning. "
                "Strong for: paraphrased queries, synonyms. Weak for: exact references (embeddings "
                "may not distinguish 'Article 5' from 'Article 7'). "
                "Hybrid: score = α × BM25_score + (1-α) × vector_score. "
                "Recall jumps from 0.72/0.68 to 0.85 because each method rescues documents the "
                "other misses. Re-ranking (cross-encoder) then reorders the combined set for "
                "precision, pushing recall to 0.91."
            ),
            "learning_outcome": "Combine BM25 and vector search for higher retrieval coverage",
        },
        # ── Section D: Agents ───────────────────────────────────────────
        {
            "id": "9.D.1",
            "lesson": "9.D",
            "type": "architecture_decision",
            "difficulty": "intermediate",
            "question": (
                "You need an agent to answer customer questions about your product. "
                "Questions are straightforward ('What is the return policy?'). "
                "Should you use Delegate, SimpleQAAgent, or ReActAgent?"
            ),
            "options": [
                "A) ReActAgent — always use the most capable agent",
                "B) Delegate for open-ended exploration, SimpleQAAgent for this use case. SimpleQAAgent with a Signature guarantees structured output (answer, confidence, sources) that the customer UI needs. Delegate's free-form output requires post-processing. ReActAgent adds tool overhead that isn't needed for simple Q&A without data lookups.",
                "C) Delegate — it's the simplest to configure",
                "D) Build a custom BaseAgent from scratch for maximum control",
            ],
            "answer": "B",
            "explanation": (
                "Agent selection should match task complexity: "
                "Delegate: autonomous, open-ended tasks. Free-form output. Best for exploration. "
                "SimpleQAAgent: structured Q&A with typed output. Best when downstream code "
                "needs predictable fields (answer, confidence, sources). "
                "ReActAgent: multi-step reasoning with tool use. Best when the agent needs to "
                "search, compute, or interact with external systems. "
                "For simple customer Q&A, SimpleQAAgent gives structured output without the "
                "overhead of ReAct's think-act-observe loop. Cost is also lower — no tool calls."
            ),
            "learning_outcome": "Select appropriate Kaizen agent type for task complexity level",
        },
        {
            "id": "9.D.2",
            "lesson": "9.D",
            "type": "code_debug",
            "difficulty": "advanced",
            "question": (
                "Exercise 6 uses Pipeline.router() to dispatch queries to specialist agents. "
                "A colleague proposes replacing it with keyword matching for 'faster routing'. "
                "Why is this an anti-pattern?"
            ),
            "code": (
                "# Colleague's proposal (WRONG)\n"
                "def route(message):\n"
                "    if 'financial' in message.lower():\n"
                "        return financial_agent\n"
                "    elif 'legal' in message.lower():\n"
                "        return legal_agent\n"
                "    else:\n"
                "        return technical_agent\n"
            ),
            "options": [
                "A) Keyword matching is fine for routing — it's faster and cheaper than LLM routing",
                "B) Keyword matching fails on: paraphrases ('money matters' → financial), implicit intent ('Is this contract valid?' → legal, no keyword match), multi-domain queries ('What are the legal risks of this financial decision?'). Pipeline.router() uses LLM reasoning to read agent capability cards and match query intent, handling all these cases. Keyword routing silently drops queries that don't match any hardcoded pattern.",
                "C) The only issue is missing the 'else' case — add more keywords to cover all cases",
                "D) Keyword matching works but is slower than Pipeline.router() due to string scanning",
            ],
            "answer": "B",
            "explanation": (
                "This is the LLM-First Rule: agent decisions MUST go through LLM reasoning, "
                "not code conditionals. Pipeline.router() examines each agent's description "
                "(capability card) and reasons about which specialist best handles the query. "
                "It handles: synonyms, paraphrases, implicit intent, multi-domain queries, "
                "and novel phrasings. Keyword matching creates a brittle dispatch table that "
                "fails silently on any input the developer didn't anticipate. "
                "In production, queries that don't match ANY keyword get routed to the default "
                "agent — often the wrong one — with no error signal."
            ),
            "learning_outcome": "Use Pipeline.router() for LLM-based routing instead of keyword dispatch",
        },
        # ── Section E: MCP ──────────────────────────────────────────────
        {
            "id": "9.E.1",
            "lesson": "9.E",
            "type": "architecture_decision",
            "difficulty": "intermediate",
            "question": (
                "Exercise 7 builds an MCP server exposing DataExplorer and TrainingPipeline "
                "as tools. Why define them as MCP tools instead of directly passing Python "
                "functions to the ReActAgent?"
            ),
            "options": [
                "A) MCP tools run faster because they use a binary protocol",
                "B) MCP decouples tool providers from consumers via a standard protocol. Direct Python functions tightly couple the agent to the tool implementation. MCP tools: (1) are discoverable at runtime, (2) can be shared across multiple agents, (3) can run on different machines, (4) can be versioned and updated independently. This is the same benefit as REST APIs vs function calls.",
                "C) ReActAgent cannot use Python functions — it only supports MCP tools",
                "D) MCP provides automatic input validation that Python functions lack",
            ],
            "answer": "B",
            "explanation": (
                "MCP (Model Context Protocol) standardizes the tool interface: "
                "any MCP-compatible agent can discover and call any MCP server's tools. "
                "Direct Python functions work but create tight coupling — changing a tool "
                "signature requires updating every agent that uses it. "
                "MCP benefits: (1) Tool discovery: agents learn available tools at runtime. "
                "(2) Reusability: one MCP server serves many agents. "
                "(3) Distribution: tools can run on a different machine. "
                "(4) Security: access can be gated per agent. "
                "Exercise 7 demonstrates this by connecting a ReActAgent to MCP tools — "
                "the agent doesn't import or know about the tool implementations."
            ),
            "learning_outcome": "Justify MCP tool architecture over direct function passing",
        },
        {
            "id": "9.E.2",
            "lesson": "9.E",
            "type": "output_interpretation",
            "difficulty": "intermediate",
            "question": (
                "Exercise 7's MCP server startup logs show: 'Registered 8 tools: "
                "[data_explore, data_preprocess, feature_engineer, train_model, "
                "evaluate_model, register_model, predict, monitor_drift]. "
                "Agent session: discovered 8 tools, used 3 (data_explore, train_model, "
                "predict).' The agent answered the user's question correctly using only 3 "
                "of 8 available tools. What does this tell you about MCP tool design?"
            ),
            "options": [
                "A) The other 5 tools are unused and should be removed to reduce overhead",
                "B) MCP tools should be fine-grained and composable. The agent discovered all 8 tools via MCP's tool listing protocol, then autonomously selected the 3 relevant to the user's query (explore data → train → predict). The unused tools (preprocess, feature_engineer, evaluate, register, monitor_drift) exist for OTHER queries. This is the MCP advantage: a single server exposes a toolkit; each agent session uses only what it needs.",
                "C) The agent should have used all 8 tools for a complete pipeline — it took a shortcut",
                "D) 3 out of 8 tools used means the tool descriptions are poorly written — the agent could not understand the other 5",
            ],
            "answer": "B",
            "explanation": (
                "MCP's tool discovery protocol lets agents dynamically select from available "
                "tools based on the task. A 'predict house prices' query needs: data_explore "
                "(understand the dataset), train_model (fit a model), predict (generate predictions). "
                "It does NOT need: preprocess (data was clean), feature_engineer (features were "
                "sufficient), evaluate (user asked for predictions not metrics), register "
                "(one-off query), monitor_drift (no production deployment). "
                "The 3/8 usage ratio is healthy — it shows the agent reasons about tool relevance "
                "rather than blindly executing all available tools. Broad tool registration with "
                "selective use is the intended MCP pattern."
            ),
            "learning_outcome": "Interpret MCP tool discovery logs and understand selective tool usage",
        },
        # ── Section F: Nexus ───────────────────────────────────────────
        {
            "id": "9.F.1",
            "lesson": "9.F",
            "type": "context_apply",
            "difficulty": "intermediate",
            "question": (
                "Exercise 8 deploys a RAG agent via Nexus with session persistence. "
                "A user asks via API: 'What AI regulations apply in Singapore?' then "
                "follows up via CLI: 'What about financial services specifically?' "
                "The CLI response correctly narrows to MAS AI guidelines. How?"
            ),
            "options": [
                "A) The CLI sends both questions to the LLM — it re-reads the first query from the API log",
                "B) Nexus sessions persist state across channels. The same session ID links the API and CLI requests. The session stores the conversation context (Singapore AI regulations), so the follow-up 'What about financial services?' is understood as a refinement. Without sessions, the CLI would treat it as a standalone question about financial services in general.",
                "C) Nexus automatically prepends 'In the context of Singapore AI regulations' to every follow-up",
                "D) The RAG retriever caches the previous query's results and reuses them",
            ],
            "answer": "B",
            "explanation": (
                "Nexus session management is channel-agnostic: a session ID created via API "
                "can be continued via CLI or MCP. The session stores conversation history, "
                "enabling contextual follow-ups. Without sessions, 'What about financial services?' "
                "would retrieve generic financial services documents. With sessions, the agent "
                "has context that the user is asking about AI regulations in Singapore's "
                "financial sector. This is critical for conversational RAG where queries build "
                "on each other. Exercise 8 demonstrates this cross-channel continuity."
            ),
            "learning_outcome": "Use Nexus sessions for cross-channel conversational state persistence",
        },
        {
            "id": "9.F.2",
            "lesson": "9.F",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 8 measures: API latency = 2.1s, cost = $0.008/query. "
                "The team wants to reduce latency to < 500ms for production. "
                "What is the most effective approach?"
            ),
            "options": [
                "A) Use a faster LLM model — switch to a smaller model variant",
                "B) The 2.1s breaks down as: retrieval (~50ms) + LLM generation (~2000ms). LLM inference dominates. Approaches: (1) Cache frequent queries and their answers (cache hit = ~10ms), (2) Use streaming to show partial results immediately (perceived latency drops even though total time is the same), (3) Use a smaller model for simple queries and route complex ones to the full model.",
                "C) Increase Nexus worker threads — the bottleneck is HTTP handling",
                "D) Pre-compute all possible answers at startup and serve from memory",
            ],
            "answer": "B",
            "explanation": (
                "LLM generation is the latency bottleneck (95%+ of total time). "
                "Response caching: frequent questions ('What is MAS TRM?') get cached answers "
                "at ~10ms. Cache hit rates of 30-50% are common for FAQ-like workloads. "
                "Streaming: user sees first tokens in ~200ms even though full response takes 2s. "
                "Perceived latency drops dramatically. "
                "Model routing: simple queries use a smaller, faster model (e.g., 7B at 500ms) "
                "while complex queries use the full model (70B at 2s). "
                "In practice, combining all three gets median latency well under 500ms."
            ),
            "learning_outcome": "Optimize agent latency through caching, streaming, and model routing",
        },
        # ── Section G: LLM Evaluation ──────────────────────────────────
        {
            "id": "9.G.1",
            "lesson": "9.G",
            "type": "context_apply",
            "difficulty": "advanced",
            "question": (
                "You are evaluating two LLMs for a legal document analysis agent. "
                "Model A scores MMLU=82%, GPQA=45%. Model B scores MMLU=78%, GPQA=62%. "
                "MMLU tests broad academic knowledge; GPQA tests graduate-level expert "
                "reasoning (questions written by PhD domain experts). Which model do you "
                "deploy for legal analysis and why?"
            ),
            "options": [
                "A) Model A — higher MMLU means better overall intelligence, which transfers to legal analysis",
                "B) Model B. GPQA measures expert-level reasoning — the ability to analyze complex problems requiring domain depth, not surface knowledge. Legal document analysis demands multi-step reasoning over nuanced text (interpreting clauses, identifying contradictions, applying precedent). Model B's 17-point GPQA advantage indicates stronger expert reasoning, even though Model A has broader surface knowledge (4-point MMLU lead). MMLU's multiple-choice format tests recall; GPQA tests applied reasoning.",
                "C) Neither — benchmark scores do not predict real-world task performance at all",
                "D) Model A — the 4% MMLU difference is more significant than the 17% GPQA difference because MMLU has more questions",
            ],
            "answer": "B",
            "explanation": (
                "Benchmark selection must match the deployment task's cognitive demands. "
                "MMLU (Massive Multitask Language Understanding) covers 57 subjects at undergraduate "
                "level — it measures breadth of knowledge. High MMLU does not guarantee deep reasoning. "
                "GPQA (Graduate-level Professional QA) uses questions designed by PhD experts to be "
                "answerable only through genuine understanding, not pattern matching. "
                "For legal analysis, the task requires: reading complex contracts, identifying implicit "
                "obligations, reasoning about clause interactions — all GPQA-style expert reasoning. "
                "Model B's 62% GPQA vs 45% is a substantial gap (38% relative improvement) indicating "
                "materially better reasoning capability for expert-level tasks."
            ),
            "learning_outcome": "Select LLMs based on task-relevant benchmarks, not headline scores",
        },
        {
            "id": "9.G.2",
            "lesson": "9.G",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Your RAG agent generates financial summaries. You need to evaluate output "
                "quality at scale (1,000 summaries/week). Three options: (1) Human expert "
                "review ($50/summary), (2) LLM-as-Judge (GPT-4 evaluating outputs, $0.05/summary), "
                "(3) Automated metrics (ROUGE + BERTScore, $0.001/summary). "
                "How should you combine these?"
            ),
            "options": [
                "A) Human review only — automated evaluation cannot assess financial accuracy",
                "B) Use all three in a tiered strategy. Tier 1: automated metrics on ALL 1,000 summaries to flag outliers (low ROUGE/BERTScore). Tier 2: LLM-as-Judge on flagged outputs + random 10% sample (~150 summaries) to assess reasoning quality. Tier 3: human expert review on LLM-flagged issues + random 2% sample (~20 summaries) to calibrate LLM-as-Judge accuracy and catch domain-specific errors. This gives comprehensive coverage at ~$12/week instead of $50,000/week.",
                "C) LLM-as-Judge only — it is as good as human evaluation and 1000× cheaper",
                "D) Automated metrics only — ROUGE and BERTScore are sufficient for financial text",
            ],
            "answer": "B",
            "explanation": (
                "Each evaluation method has complementary strengths and weaknesses: "
                "Automated metrics: fast and cheap, catch gross failures (empty outputs, wrong length, "
                "copied input) but cannot assess factual accuracy or reasoning quality. "
                "LLM-as-Judge: evaluates coherence, relevance, and reasoning quality at scale. "
                "Correlates 0.8-0.9 with human judgment on general quality. Weak on: domain-specific "
                "accuracy (may not catch subtle financial errors), novel failure modes. "
                "Human expert: catches domain errors (wrong financial interpretation, misleading "
                "summaries) but does not scale to 1,000/week. "
                "The tiered approach uses each method where it excels: metrics for coverage, "
                "LLM for quality, humans for calibration and domain accuracy."
            ),
            "learning_outcome": "Design tiered evaluation combining automated metrics, LLM-as-Judge, and human review",
        },
        # ── Section H: Agent Architecture ──────────────────────────────
        {
            "id": "9.H.1",
            "lesson": "9.H",
            "type": "code_debug",
            "difficulty": "advanced",
            "question": (
                "A student builds a multi-agent system where the orchestrator calls "
                "specialist agents. The orchestrator hangs indefinitely on the second "
                "query. The first query works fine. What is wrong?"
            ),
            "code": (
                "class Orchestrator(BaseAgent):\n"
                "    def __init__(self):\n"
                "        self.financial = FinancialAgent()\n"
                "        self.legal = LegalAgent()\n"
                "        # Bug: agents share a single session\n"
                "        self.session = create_session()\n"
                "\n"
                "    async def run(self, query):\n"
                "        if 'financial' in query:\n"
                "            return await self.financial.run(\n"
                "                query, session=self.session\n"
                "            )\n"
                "        return await self.legal.run(\n"
                "            query, session=self.session\n"
                "        )\n"
            ),
            "options": [
                "A) The if/else routing is too simplistic — use Pipeline.router() instead",
                "B) All agents share one session object. The first query acquires the session and completes. The second query tries to use the same session, which may still hold state (conversation history, locks) from the first. Fix: create a new session per query (session = create_session() inside run()), or use Kaizen's Pipeline which manages session lifecycle per request. Shared mutable state across agent invocations is the #1 multi-agent bug.",
                "C) BaseAgent does not support multiple specialist agents — use SupervisorAgent instead",
                "D) The async/await syntax is incorrect — use asyncio.gather() for parallel execution",
            ],
            "answer": "B",
            "explanation": (
                "Session-per-request is a fundamental pattern in agent architectures. "
                "A session holds: conversation history, tool call state, budget tracking, "
                "and potentially locks. Sharing a session across sequential agent calls means: "
                "(1) The second agent sees the first agent's conversation history (context bleed). "
                "(2) If the session has a lock or active state from the first call, the second "
                "call blocks waiting for it. (3) Budget tracking accumulates incorrectly. "
                "Fix: self.session = create_session() per request in run(), not in __init__(). "
                "Kaizen's Pipeline and SupervisorAgent handle this correctly by default — "
                "each agent invocation gets an isolated session."
            ),
            "learning_outcome": "Isolate agent sessions to prevent state leakage in multi-agent systems",
        },
        {
            "id": "9.H.2",
            "lesson": "9.H",
            "type": "output_interpretation",
            "difficulty": "intermediate",
            "question": (
                "Your deployed RAG agent via Nexus shows these metrics: "
                "Time to first token (TTFT): 450ms, Total response time: 3.2s, "
                "Token generation rate: 85 tokens/sec. Users report the agent feels "
                "'slow' despite the 450ms TTFT. The average response is 240 tokens. "
                "What is the actual user experience and how do you improve it?"
            ),
            "options": [
                "A) 450ms TTFT is fast — users are being unreasonable. No change needed.",
                "B) TTFT (450ms) is acceptable — users see the first word quickly. But total time (3.2s) is the perceived slowness. Breakdown: 450ms retrieval + LLM prefill, then 240 tokens at 85 tok/sec = 2.8s generation. Users wait 3.2s for a complete answer. With streaming enabled, users read as tokens arrive — perceived completion time drops because reading speed (~250 words/min ≈ 4 tok/sec) is much slower than generation (85 tok/sec). The answer appears complete before the user finishes reading.",
                "C) The token generation rate of 85 tok/sec is too slow — upgrade to a faster GPU",
                "D) 240 tokens is too long — reduce max_tokens to 50 for faster responses",
            ],
            "answer": "B",
            "explanation": (
                "Three latency metrics matter for agent UX: "
                "1. TTFT (Time To First Token): how long before the user sees anything. "
                "   450ms is good — under the 1s threshold for perceived responsiveness. "
                "2. Generation rate: 85 tok/sec means ~340 words/min of output. "
                "3. Total time: TTFT + tokens/rate = 450ms + 240/85 = 3.27s. "
                "The key insight: humans read at ~250 words/min (4 tok/sec), but tokens arrive "
                "at 85 tok/sec. With streaming, the user sees tokens 21× faster than they can read. "
                "By the time they read the first sentence (~15 words, 0.2s of reading), the model "
                "has generated ~17 more tokens. The response feels complete before they finish reading. "
                "Without streaming, they wait 3.2s for nothing, then read — doubling perceived latency."
            ),
            "learning_outcome": "Analyze agent latency metrics and apply streaming to improve perceived performance",
        },
    ],
}
