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
        # ── Section F-H: Nexus ──────────────────────────────────────────
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
    ],
}
