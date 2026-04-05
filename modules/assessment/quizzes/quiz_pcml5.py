# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""ASCENT Module 5 — AI-Resilient Assessment Questions

LLMs, AI Agents & RAG Systems
Covers: Delegate, CoT, ReActAgent, RAGResearchAgent, MCP, ML agents,
        multi-agent patterns, Nexus deployment
"""

QUIZ = {
    "module": "ASCENT5",
    "title": "LLMs, AI Agents & RAG Systems",
    "questions": [
        # ── Lesson 1: LLM fundamentals and Delegate ───────────────────────
        {
            "id": "5.1.1",
            "lesson": "5.1",
            "type": "code_debug",
            "difficulty": "foundation",
            "question": (
                "A student runs Delegate in Exercise 1 but ignores the mandatory cost budget. "
                "The script runs for 20 minutes and incurs a $47 API charge. "
                "What two things are wrong with this setup?"
            ),
            "code": (
                "from kaizen_agents import Delegate\n"
                "import os\n"
                "\n"
                "delegate = Delegate(\n"
                "    model='gpt-4o',  # Bug 1: hardcoded model name\n"
                "    # Bug 2: missing max_llm_cost_usd\n"
                ")\n"
                "async for event in delegate.run('Analyse this entire 500MB dataset in detail'):\n"
                "    print(event)"
            ),
            "options": [
                "A) delegate.run() should be awaited, not iterated; Delegate does not support async for",
                "B) The import path is wrong; Delegate should be imported from kaizen.agents",
                "C) (1) Model name is hardcoded; must use os.environ['DEFAULT_LLM_MODEL']; (2) max_llm_cost_usd is missing; this is mandatory for all Module 5 exercises to prevent runaway spending",
                "D) 500MB datasets cannot be processed by Delegate; use DataExplorer first",
            ],
            "answer": "C",
            "explanation": (
                "Two violations from the Module 5 mandatory setup: "
                "(1) Hardcoded model names are prohibited — use os.environ.get('DEFAULT_LLM_MODEL'). "
                "This is a zero-tolerance rule per env-models.md. "
                "(2) max_llm_cost_usd is the hard budget cap that prevents unbounded spending. "
                "Without it, a long-running Delegate task can consume unlimited API budget. "
                "Correct: Delegate(model=os.environ['DEFAULT_LLM_MODEL'], max_llm_cost_usd=2.0)"
            ),
            "learning_outcome": "Apply mandatory Delegate setup: env-based model name and cost budget",
        },
        {
            "id": "5.1.2",
            "lesson": "5.1",
            "type": "context_apply",
            "difficulty": "intermediate",
            "question": (
                "In Exercise 1, you compare Delegate (autonomous) vs SimpleQAAgent "
                "(custom Signature). Running both on the same data analysis question, "
                "Delegate takes 8 API calls and returns a comprehensive answer, "
                "while SimpleQAAgent takes 1 API call and returns a structured dict. "
                "For a production system that answers 10,000 customer queries per day about "
                "their account data, which approach is preferable and why?"
            ),
            "options": [
                "A) SimpleQAAgent — 1 API call per query vs 8 means ~8× lower cost at 10,000 queries/day. The structured Signature output (typed InputField/OutputField) also enables downstream processing and validation. Delegate's autonomous multi-step reasoning is valuable for complex one-off analysis, not high-volume standardised queries",
                "B) Delegate — more API calls means more thorough analysis",
                "C) Delegate — structured output is not important for customer queries",
                "D) Both are identical in cost; Delegate just shows its work",
            ],
            "answer": "A",
            "explanation": (
                "Delegate's TAOD loop (Think-Act-Observe-Decide) is designed for autonomous exploration "
                "of open-ended problems — it deliberately uses multiple calls to build context. "
                "For a standardised question with a known answer structure, SimpleQAAgent's "
                "fixed Signature is 8× cheaper and returns a typed, validated response. "
                "At 10,000 queries/day: Delegate ≈ 80,000 API calls vs SimpleQAAgent ≈ 10,000. "
                "This is the core trade-off between Delegate (autonomy) and Signature agents (efficiency)."
            ),
            "learning_outcome": "Choose between Delegate and Signature agents based on query volume and structure",
        },
        # ── Lesson 2: Chain-of-thought and structured reasoning ───────────
        {
            "id": "5.2.1",
            "lesson": "5.2",
            "type": "architecture_decision",
            "difficulty": "intermediate",
            "question": (
                "You are building a credit risk explanation agent that must output a "
                "structured JSON with fields: risk_level, primary_factors (list), "
                "recommended_action, and confidence_score. "
                "Should you use a raw Delegate.run() call or build a custom Signature "
                "with InputField/OutputField? Explain the key advantage of the Signature approach "
                "for a regulated financial service."
            ),
            "options": [
                "A) Custom Signature with OutputField(type=RiskExplanation) — the output is schema-validated at runtime so you can guarantee the agent returns exactly the required fields with correct types. For a regulated service, this means you never get malformed JSON causing a downstream crash, and the schema is auditable by the regulator",
                "B) Raw Delegate — more flexible and handles edge cases better",
                "C) Both are equivalent; parse the Delegate output with json.loads()",
                "D) Signature agents cannot produce JSON output; use a post-processing function",
            ],
            "answer": "A",
            "explanation": (
                "Kaizen Signatures define typed InputField and OutputField contracts. "
                "When the agent runs, Kaizen validates the output against the OutputField types — "
                "if the LLM returns an incomplete or malformed response, Kaizen retries or raises a clear error "
                "rather than passing bad data downstream. "
                "For a regulated credit decision, this schema enforcement is non-negotiable: "
                "you cannot allow an unstructured string to flow into a loan approval system."
            ),
            "learning_outcome": "Use Kaizen Signature for schema-validated agent outputs in regulated contexts",
        },
        # ── Lesson 3: ReAct agents with tools ────────────────────────────
        {
            "id": "5.3.1",
            "lesson": "5.3",
            "type": "output_interpret",
            "difficulty": "advanced",
            "question": (
                "Running ReActAgent in Exercise 3 on the credit dataset produces this trace:\n\n"
                "  Thought: I need to understand the data distribution first\n"
                "  Action: tool_profile_data('sg_credit_scoring')\n"
                "  Observation: 15,000 rows, default_rate=8.2%, 12 features, 3 alerts\n"
                "  Thought: I should check for highly correlated features before recommending engineering\n"
                "  Action: tool_check_correlations(threshold=0.8)\n"
                "  Observation: income x employment_income corr=0.91 (HIGH)\n"
                "  Thought: One of the pair should be removed or combined\n"
                "  Answer: Remove employment_income; it is 91% correlated with income...\n\n"
                "What does this trace demonstrate about ReAct vs a single-call Delegate query, "
                "and when would you choose ReAct over Delegate?"
            ),
            "options": [
                "A) ReAct is always slower; use Delegate for all production scenarios",
                "B) The trace shows ReAct made two unnecessary tool calls; one call would suffice",
                "C) ReAct cannot call DataExplorer tools; only Delegate can access ML tools",
                "D) The trace shows ReAct's Reason-Act-Observe loop: the agent produces an intermediate thought, calls a tool, observes the result, and reasons again before acting. This is preferable to Delegate when you want to see and audit the exact tool calls and intermediate observations — useful for regulated workflows where explainability of the agent's decision process is required",
            ],
            "answer": "D",
            "explanation": (
                "Delegate is a higher-level abstraction — it autonomously decides when to reason and act "
                "but does not expose a step-by-step trace by default. "
                "ReActAgent exposes every thought-action-observation cycle explicitly. "
                "In regulated ML workflows, being able to audit 'the agent decided to check correlations "
                "because it profiled the data first' provides a complete decision audit trail "
                "that a Delegate summary cannot."
            ),
            "learning_outcome": "Contrast ReAct's explicit trace with Delegate's opaque autonomy for auditability",
        },
        {
            "id": "5.3.2",
            "lesson": "5.3",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student's ReActAgent never uses the tool_profile_data function. "
                "The agent always responds with generic analysis. What is wrong?"
            ),
            "code": (
                "from kaizen_agents.agents.specialized.react import ReActAgent\n"
                "\n"
                "async def tool_profile_data(dataset_name: str) -> str:\n"
                '    """Profile a dataset using DataExplorer."""\n'
                "    # ... implementation ...\n"
                "    return profile_summary\n"
                "\n"
                "# Bug: tools not registered\n"
                "agent = ReActAgent(\n"
                "    model=os.environ['DEFAULT_LLM_MODEL'],\n"
                "    max_llm_cost_usd=3.0,\n"
                "    # tools parameter is missing\n"
                ")\n"
                "result = await agent.run('Analyse the credit dataset')"
            ),
            "options": [
                "A) tool_profile_data must be decorated with @agent.tool to be registered",
                "B) ReActAgent cannot use async tools; make tool_profile_data synchronous",
                "C) The tools parameter is missing from ReActAgent; pass tools=[tool_profile_data] to register the function; without it, the agent has no tools to call and falls back to pure LLM reasoning",
                "D) The function signature must include 'self' to be registered as an agent tool",
            ],
            "answer": "C",
            "explanation": (
                "ReActAgent discovers available tools from the tools= parameter in its constructor. "
                "Without this list, the agent operates in reasoning-only mode — "
                "it generates text responses without ever calling an external function. "
                "Correct: ReActAgent(model=..., max_llm_cost_usd=3.0, tools=[tool_profile_data, tool_check_correlations])"
            ),
            "learning_outcome": "Register tools with ReActAgent via the tools= constructor parameter",
        },
        # ── Lesson 4: RAG systems ─────────────────────────────────────────
        {
            "id": "5.4.1",
            "lesson": "5.4",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "You are building a RAG system to answer questions about Singapore financial "
                "regulations from a 2,000-page PDF corpus. Your colleague suggests "
                "chunking all documents into 512-token chunks with 50-token overlap. "
                "What problem does overlap solve, and what is the risk of chunks that are too small?"
            ),
            "options": [
                "A) Overlap reduces storage requirements; small chunks are always more precise",
                "B) Smaller chunks are always better because retrieval precision increases",
                "C) Overlap is only needed for code documents; regulatory PDFs do not need it",
                "D) Overlap prevents context loss at chunk boundaries — a sentence about 'the penalty' makes no sense if 'the regulation' was in the previous chunk. Chunks that are too small (e.g., 100 tokens) cause the retrieved context to lack enough information for the LLM to answer, leading to hallucination as the model fills in missing context",
            ],
            "answer": "D",
            "explanation": (
                "Chunking with overlap ensures that concepts split across chunk boundaries "
                "appear in at least one complete chunk. "
                "For regulatory text, a provision might reference a definition from the previous paragraph — "
                "without overlap, the retrieval system might return the provision but not its definition. "
                "Very small chunks (< 200 tokens) also reduce the information density per retrieved chunk, "
                "forcing the system to retrieve more chunks and increasing the chance of irrelevant context."
            ),
            "learning_outcome": "Explain chunk overlap trade-offs for RAG over long-form regulatory documents",
        },
        # ── Lesson 5: MCP servers ─────────────────────────────────────────
        {
            "id": "5.5.1",
            "lesson": "5.5",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student creates an MCP server but agents cannot discover the tools. "
                "What is wrong with this setup?"
            ),
            "code": (
                "from kailash.mcp_server import MCPServer, MCPTool, MCPToolResult\n"
                "\n"
                "server = MCPServer(name='ml_tools')\n"
                "\n"
                "# Define tool\n"
                "async def profile_data(dataset_name: str) -> MCPToolResult:\n"
                "    result = await _run_profiler(dataset_name)\n"
                "    return MCPToolResult(content=result)\n"
                "\n"
                "# Bug: tool is defined but never registered\n"
                "# server.register_tool() is missing\n"
                "\n"
                "await server.start(transport=StdioTransport())"
            ),
            "options": [
                "A) server.register_tool() is missing — the function must be wrapped in MCPTool and registered before server.start(); without registration, the server starts but advertises an empty tool list to clients",
                "B) MCPServer must be imported from kailash.mcp_server.server, not kailash.mcp_server",
                "C) StdioTransport cannot be used for multi-agent scenarios; use SSETransport",
                "D) MCPToolResult must include a status field; content alone is insufficient",
            ],
            "answer": "A",
            "explanation": (
                "MCP tool discovery works by clients calling list_tools — the server returns its registry. "
                "If no tools are registered, the server returns an empty list and agents have nothing to call. "
                "The correct pattern from Exercise 5:\n"
                "tool = MCPTool(name='profile_data', description='...', handler=profile_data)\n"
                "server.register_tool(tool)\n"
                "await server.start(transport=StdioTransport())"
            ),
            "learning_outcome": "Register tools with MCPServer before starting to enable agent discovery",
        },
        {
            "id": "5.5.2",
            "lesson": "5.5",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 5 covers two MCP transport options: StdioTransport and SSETransport. "
                "Your team wants to deploy the ML tools server so that multiple agents "
                "running in different processes can connect to it simultaneously. "
                "Which transport is required and why?"
            ),
            "options": [
                "A) StdioTransport — it supports multiple simultaneous connections via process forking",
                "B) MCPServer cannot serve multiple clients; deploy one server per agent",
                "C) Both transports support multi-client connections equally",
                "D) SSETransport (Server-Sent Events) — it runs as an HTTP server on a fixed port, allowing multiple clients to connect independently over a network. StdioTransport is subprocess-based (one client per pipe) and cannot serve multiple simultaneous clients",
            ],
            "answer": "D",
            "explanation": (
                "StdioTransport uses a subprocess pipe — one MCPServer process per client. "
                "It is appropriate for local development and single-agent testing. "
                "SSETransport starts an HTTP/SSE server on a configurable port. "
                "Multiple agents connect to the same URL (http://host:port) and make independent "
                "tool calls. This is the production deployment pattern for shared ML tool servers."
            ),
            "learning_outcome": "Select SSETransport for multi-client MCP server deployments",
        },
        # ── Lesson 6: ML agents ───────────────────────────────────────────
        {
            "id": "5.6.1",
            "lesson": "5.6",
            "type": "process_doc",
            "difficulty": "advanced",
            "question": (
                "You build an ML agent using Delegate that autonomously trains and evaluates "
                "a credit model. The agent's task prompt includes 'choose the best model'. "
                "After running, the agent reports AUC=0.91 on the training set. "
                "What governance violation has occurred and what should the prompt "
                "have specified to prevent it?"
            ),
            "options": [
                "A) The agent evaluated on training data, not a held-out test set — this is a data leakage violation. The prompt should specify: 'evaluate on the held-out test set only; training set performance is not a valid model selection criterion'. Without this constraint, the agent optimises in-sample fit, producing a model that is overfit and misleading",
                "B) No violation — the agent correctly maximised the AUC metric",
                "C) The violation is using AUC instead of F1-score; the prompt should specify F1",
                "D) The violation is using Delegate for model training; only TrainingPipeline may train models",
            ],
            "answer": "A",
            "explanation": (
                "An autonomous agent given a vague goal ('choose the best model') will optimise "
                "whatever metric it can measure most easily. Training set AUC is trivially improvable "
                "by overfitting. Agents need explicit constraints: 'use train/test split from PreprocessingPipeline', "
                "'report test set metrics only', 'do not tune hyperparameters on the test set'. "
                "This is a concrete example of why agent operating envelopes matter — "
                "covered in Module 6's PACT governance exercises."
            ),
            "learning_outcome": "Identify data leakage risk in autonomous ML agent prompts and write corrective constraints",
        },
        # ── Lesson 7: Multi-agent patterns ────────────────────────────────
        {
            "id": "5.7.1",
            "lesson": "5.7",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "You are designing a multi-agent ML pipeline with three stages: "
                "feature engineering, model training, and evaluation. "
                "Each stage should run concurrently where possible. "
                "Which Kaizen pattern is appropriate, and what does SupervisorWorkerPattern "
                "provide that three independent Delegate instances do not?"
            ),
            "options": [
                "A) Three independent Delegate instances — simpler and equivalent to SupervisorWorkerPattern",
                "B) SupervisorWorkerPattern only works for sequential pipelines; use asyncio.gather() for concurrent execution",
                "C) SupervisorWorkerPattern — the supervisor orchestrates task decomposition, monitors worker completion, aggregates results, and handles worker failures with retries. Three independent Delegates have no coordination mechanism: if the feature engineering agent fails, the training agent proceeds with stale features",
                "D) SupervisorWorkerPattern requires a shared database; use it only when persisting intermediate results",
            ],
            "answer": "C",
            "explanation": (
                "SupervisorWorkerPattern provides: "
                "(1) Task dependency management — supervisor knows feature engineering must complete before training starts; "
                "(2) Failure propagation — if a worker fails, the supervisor decides to retry or abort (not silently proceed); "
                "(3) Result aggregation — the supervisor collects worker outputs into a coherent final result. "
                "Three independent Delegates are fire-and-forget with no coordination. "
                "asyncio.gather() parallelises execution but has no semantic understanding of task dependencies."
            ),
            "learning_outcome": "Distinguish SupervisorWorkerPattern coordination from independent Delegate execution",
        },
        # ── Lesson 8: Nexus deployment ────────────────────────────────────
        {
            "id": "5.8.1",
            "lesson": "5.8",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student deploys an ML pipeline via Nexus but the /predict endpoint "
                "always returns 404. What is wrong with this setup?"
            ),
            "code": (
                "from kailash_nexus import Nexus\n"
                "\n"
                "app = Nexus()\n"
                "app.start()  # Bug: starting before registering\n"
                "\n"
                "# Student tries to register after start\n"
                "app.register(credit_scoring_workflow)"
            ),
            "options": [
                "A) Nexus must be imported from kailash.nexus, not kailash_nexus",
                "B) app.register() must be called before app.start() — Nexus builds the route table at startup from the registered workflows. Registering after start does not add routes to the already-running server",
                "C) workflow must be built with workflow.build() before registering with Nexus",
                "D) Nexus requires an explicit port number: Nexus(port=8080)",
            ],
            "answer": "B",
            "explanation": (
                "Nexus generates REST routes, CLI handlers, and MCP tool registrations at startup. "
                "The route table is fixed at the moment app.start() is called. "
                "Registering a workflow after start does nothing because the HTTP server is already "
                "running with its original route map. "
                "Correct pattern: register first, then start:\n"
                "app = Nexus()\n"
                "app.register(credit_scoring_workflow)\n"
                "app.start()"
            ),
            "learning_outcome": "Follow the correct Nexus register-before-start pattern",
        },
        {
            "id": "5.8.2",
            "lesson": "5.8",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "You need to expose a credit scoring workflow via REST API with state "
                "maintained across requests (e.g., a user's session history). "
                "Nexus provides app.create_session(). "
                "Why is a Nexus session preferable to storing state in a Python dict on "
                "the server, and what does the session provide that a dict cannot?"
            ),
            "options": [
                "A) A Python dict is preferable — simpler and faster than Nexus sessions",
                "B) Sessions and dicts are equivalent; use a dict for simplicity in production",
                "C) Nexus sessions are only needed when using MCP; REST can use a dict",
                "D) Nexus sessions provide cross-channel consistency (REST, CLI, and MCP share the same session state), automatic expiry, and thread-safe state management across concurrent requests. A Python dict is not thread-safe and is lost when the process restarts or when load-balanced across multiple instances",
            ],
            "answer": "D",
            "explanation": (
                "A Python dict is process-local and not thread-safe under concurrent API requests. "
                "Two simultaneous requests can corrupt the same dict key without locking. "
                "Nexus sessions are backed by the ConnectionManager's storage (persistent, thread-safe), "
                "expire automatically after inactivity, and are shared across REST/CLI/MCP channels "
                "so a user who starts on REST and continues on CLI maintains the same context. "
                "This is critical for stateful workflows like multi-turn credit assessments."
            ),
            "learning_outcome": "Justify Nexus sessions over in-memory state for production multi-channel deployments",
        },
        # ── Additional questions covering lessons 1–8 breadth ─────────────
        {
            "id": "5.1.3",
            "lesson": "5.1",
            "type": "output_interpret",
            "difficulty": "intermediate",
            "question": (
                "In Exercise 1, Delegate reports this at the end of the run:\n\n"
                "  LLM calls: 8\n"
                "  Total tokens: 12,847\n"
                "  Estimated cost: $0.19\n"
                "  Budget remaining: $1.81 / $2.00\n\n"
                "The analysis quality is good. A manager asks: 'Can we run this for all "
                "200 customer segments instead of 1?' "
                "What is the estimated total cost and what governance change is needed?"
            ),
            "options": [
                "A) Cost = $0.19 × 200 = $38; no governance change needed",
                "B) Estimated cost: $0.19 × 200 = $38 if each segment requires similar analysis. The current max_llm_cost_usd=2.0 is insufficient — a human operator must explicitly authorise a higher budget (e.g., max_llm_cost_usd=50.0) before scaling to 200 runs. The budget is a governance gate, not just a soft limit",
                "C) Delegate automatically scales; no cost estimation is possible before running",
                "D) Run all 200 segments in one call by passing all data in a single prompt",
            ],
            "answer": "B",
            "explanation": (
                "Linear cost scaling: $0.19 × 200 = $38. "
                "The current budget cap of $2.00 would terminate after ~10 segments ($2.00 / $0.19 ≈ 10). "
                "The max_llm_cost_usd parameter is a hard governance gate — "
                "Delegate stops as soon as the budget is reached, not gracefully scales back. "
                "A human must explicitly set the higher budget: the agent cannot self-authorise."
            ),
            "learning_outcome": "Estimate LLM cost at scale and identify when human budget authorisation is required",
        },
        {
            "id": "5.2.2",
            "lesson": "5.2",
            "type": "context_apply",
            "difficulty": "advanced",
            "question": (
                "You build a credit risk Signature agent with an OutputField "
                "of type RiskExplanation (a Pydantic model). "
                "The LLM occasionally returns a response where confidence_score > 1.0. "
                "Your Pydantic model has confidence_score: float. "
                "What change to the OutputField prevents this and why is field-level validation "
                "superior to post-processing the raw LLM output?"
            ),
            "options": [
                "A) Change confidence_score: float to confidence_score: str and parse manually",
                "B) Wrap the agent call in try/except and clamp the value: min(1.0, max(0.0, score))",
                "C) Add a Pydantic validator: confidence_score: float = Field(..., ge=0.0, le=1.0). When the LLM returns 1.2, Pydantic raises ValidationError before the response reaches downstream code — the agent retries rather than propagating an invalid value. Post-processing requires manually handling every edge case after the fact",
                "D) Instruct the LLM in the system prompt to always return values between 0 and 1",
            ],
            "answer": "C",
            "explanation": (
                "Pydantic field constraints (ge=0.0, le=1.0) are enforced at deserialisation time — "
                "if the LLM output fails validation, Kaizen catches the ValidationError and "
                "can retry the LLM call with the error as feedback. "
                "Post-processing via min/max clamping silently accepts the invalid value "
                "and passes it downstream — it hides the LLM's misbehaviour rather than correcting it. "
                "System prompt instructions are probabilistic; Pydantic validation is deterministic."
            ),
            "learning_outcome": "Use Pydantic field constraints in Kaizen OutputField for deterministic output validation",
        },
        {
            "id": "5.4.2",
            "lesson": "5.4",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student's RAGResearchAgent returns hallucinated answers for regulatory questions. "
                "The retrieval step finds relevant documents but the LLM ignores them. "
                "What common prompt construction error causes this?"
            ),
            "code": (
                "context_docs = retriever.search(query, top_k=5)\n"
                "context_text = '\\n'.join([doc.content for doc in context_docs])\n"
                "\n"
                "# Bug: context is appended after the question\n"
                'prompt = f"""\n'
                "Answer the following question about Singapore financial regulations:\n"
                "{query}\n"
                "\n"
                "Here are some relevant documents:\n"
                "{context_text}\n"
                '"""'
            ),
            "options": [
                "A) top_k=5 is too many documents; reduce to top_k=1",
                "B) The question appears before the context — LLMs generate the answer token-by-token and may begin answering before reading the context. Place the context before the question: 'Here are relevant documents:\\n{context_text}\\n\\nBased ONLY on the above, answer: {query}'",
                "C) context_text must be formatted as JSON, not plain text",
                "D) RAGResearchAgent handles prompt construction internally; never build prompts manually",
            ],
            "answer": "B",
            "explanation": (
                "Large language models generate autoregressively — the answer tokens are influenced "
                "by what came before them in the prompt. "
                "If the question appears first, the model may begin generating a plausible-sounding "
                "answer from its parametric memory before encountering the retrieved context. "
                "Placing context first and using an explicit grounding instruction "
                "('based ONLY on the above') reduces hallucination by anchoring the model "
                "to the retrieved documents."
            ),
            "learning_outcome": "Structure RAG prompts with context before question to reduce hallucination",
        },
        {
            "id": "5.6.2",
            "lesson": "5.6",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "You are building an ML agent that autonomously selects and trains models. "
                "The agent has access to TrainingPipeline, ModelRegistry.promote(), and DataExplorer. "
                "A security review flags: 'The agent can promote its own trained models to production.' "
                "Which ASCENT6 concept solves this, and what is the minimum change to the agent's "
                "configuration to prevent self-promotion?"
            ),
            "options": [
                "A) Both A and the PACT approach work, but the PACT approach is more robust: set can_deploy=False in the agent's role permissions. Even if ModelRegistry.promote() is in the tool list, GovernanceEngine.check_permission() blocks calls to that action for agents whose role lacks deploy permission. Removing the tool only prevents the agent from knowing about promotion; PACT prevents the action at the governance layer regardless of tool knowledge",
                "B) Remove ModelRegistry.promote() from the agent's tool list — the agent cannot promote what it cannot call",
                "C) Use DriftMonitor to detect if the agent is promoting models too frequently",
                "D) The agent should be trusted to make promotion decisions autonomously",
            ],
            "answer": "A",
            "explanation": (
                "Both options prevent self-promotion, but they defend at different layers. "
                "Removing the tool is a capability restriction — the agent cannot call the function at all. "
                "PACT's can_deploy=False is an authority restriction — the agent knows about promotion "
                "but is not authorised to execute it. "
                "The PACT approach is more auditable (every blocked attempt is logged to AuditChain) "
                "and more robust (governance cannot be bypassed by finding an alternate tool reference)."
            ),
            "learning_outcome": "Apply PACT permission restriction vs tool removal for agent deployment authority control",
        },
        {
            "id": "5.7.2",
            "lesson": "5.7",
            "type": "process_doc",
            "difficulty": "advanced",
            "question": (
                "In a multi-agent ML pipeline, the Feature Engineering Agent produces a feature "
                "set and the Training Agent consumes it. The training fails because the feature "
                "columns contain null values that the feature agent did not clean. "
                "How should the inter-agent handoff be structured to catch this before "
                "the training agent starts, and which Kailash class enforces the contract?"
            ),
            "options": [
                "A) The training agent should silently skip null columns",
                "B) Use asyncio.gather() to run both agents in parallel; the training agent will detect nulls",
                "C) Pass raw Polars DataFrames between agents; type checking is the training agent's responsibility",
                "D) The Feature Engineering Agent should return a typed result with a FeatureSchema that declares max_null_fraction=0.0. The supervisor uses ModelSignature or a custom Pydantic model to validate the handoff data before passing it to the Training Agent — validation failure triggers a Dereliction escalation rather than a silent training crash",
            ],
            "answer": "D",
            "explanation": (
                "In a Kaizen SupervisorWorkerPattern, the supervisor mediates inter-agent data flow. "
                "Typed handoff validation at the boundary (FeatureSchema or Pydantic model) catches "
                "data quality issues before the downstream agent starts. "
                "This converts a runtime training crash (hours in) into a fast validation failure "
                "at the handoff boundary (seconds in), and creates an AuditChain record "
                "that the feature agent delivered non-compliant output."
            ),
            "learning_outcome": "Design typed inter-agent handoffs with schema validation to prevent silent downstream failures",
        },
        {
            "id": "5.3.3",
            "lesson": "5.3",
            "type": "output_interpret",
            "difficulty": "intermediate",
            "question": (
                "Your ReActAgent is given a 15-feature credit dataset and the task "
                "'identify which features to keep'. The agent's trace shows:\n\n"
                "  Action: tool_check_correlations(threshold=0.8)\n"
                "  Observation: income x employment_income corr=0.91\n"
                "  Thought: employment_income is redundant; drop it\n"
                "  Answer: Drop employment_income\n\n"
                "The agent stopped after one tool call without profiling the data first. "
                "Why is this incomplete, and what did the agent miss by not calling "
                "tool_profile_data before tool_check_correlations?"
            ),
            "options": [
                "A) The agent is correct; correlation is the only relevant feature selection criterion",
                "B) The agent should have run more correlation checks with different thresholds",
                "C) Without profiling first, the agent does not know: (1) null rates — a feature with 40% nulls should be dropped regardless of correlation; (2) cardinality — a near-unique identifier column is harmful even if not correlated; (3) DataExplorer alerts that might flag other issues. The complete ReAct trace should profile_data → interpret alerts → check_correlations → synthesise",
                "D) tool_profile_data is optional; the agent chose the more efficient path",
            ],
            "answer": "C",
            "explanation": (
                "Feature selection requires holistic data understanding, not just correlation. "
                "A proper ReAct trace profiles first: DataExplorer identifies nulls, high-cardinality columns, "
                "constant features, and distribution alerts. "
                "Only after understanding the full data quality picture does correlation analysis "
                "make sense. The agent's shortcut might recommend keeping a feature that has "
                "60% nulls (unacceptable for training) because it appeared uncorrelated."
            ),
            "learning_outcome": "Design complete ReAct tool sequences that profile before performing feature selection",
        },
        {
            "id": "5.5.3",
            "lesson": "5.5",
            "type": "context_apply",
            "difficulty": "intermediate",
            "question": (
                "After deploying the MCP server in Exercise 5, you notice the profile_data tool "
                "is being called 500 times per minute by a runaway agent loop. "
                "The DataExplorer computation is expensive (2 seconds per call). "
                "Which two changes to the MCPServer configuration address this?"
            ),
            "options": [
                "A) Increase the server's max_workers to handle more concurrent calls",
                "B) Restart the MCP server to clear the runaway agent's connection",
                "C) (1) Add rate limiting: server.register_tool(tool, rate_limit=10) to cap calls per minute per client; (2) Add result caching: MCPTool(handler=profile_data, cache_ttl=300) returns cached results for repeated identical inputs within 5 minutes — the DataExplorer result for the same dataset does not change within seconds",
                "D) Replace SSETransport with StdioTransport to limit concurrent connections",
            ],
            "answer": "C",
            "explanation": (
                "Rate limiting (calls/minute per client) prevents a single misbehaving agent from "
                "monopolising the server. The client receives a rate limit error and must back off. "
                "Result caching returns the cached DataExplorer output for repeated calls on the "
                "same dataset within the TTL window — 500 identical calls cost the same as 1. "
                "Both changes are defensive measures: rate limiting addresses abuse, "
                "caching addresses inefficiency."
            ),
            "learning_outcome": "Apply MCPServer rate limiting and result caching to prevent runaway agent tool abuse",
        },
        {
            "id": "5.8.3",
            "lesson": "5.8",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "You deploy the credit scoring Kaizen agent via Nexus as a REST endpoint. "
                "During a load test, you observe that agent responses are inconsistent: "
                "the same credit application gets different risk scores on consecutive calls. "
                "Which of the following explains this, and how does Nexus session management "
                "address it?"
            ),
            "options": [
                "A) LLM temperature causes randomness; always set temperature=0.0 for deterministic agents",
                "B) Both A and session history can cause inconsistency: an agent with conversation history in its session context may reason differently on the second call based on prior context. Nexus sessions isolate per-user state — each credit application should use a fresh session (app.create_session()) so agents start from a clean context. Combine with temperature=0.0 for fully deterministic credit decisions",
                "C) Inconsistency only occurs with streaming responses; use blocking calls",
                "D) The Nexus load balancer routes requests to different instances; add sticky sessions",
            ],
            "answer": "B",
            "explanation": (
                "Two sources of non-determinism: "
                "(1) LLM sampling temperature > 0 adds randomness to token selection. "
                "(2) Session history: if the agent's session accumulates prior credit decisions, "
                "the second application is evaluated with the context of the first one. "
                "For credit scoring, each application must be evaluated independently: "
                "fresh session + temperature=0.0. "
                "Nexus create_session() creates an isolated context; the session should be discarded "
                "after each complete credit assessment."
            ),
            "learning_outcome": "Identify session history and LLM temperature as sources of inconsistency in credit scoring agents",
        },
        {
            "id": "5.2.3",
            "lesson": "5.2",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student builds a Kaizen agent with a custom Signature but the agent "
                "always returns the same answer regardless of the input. "
                "What is the structural bug?"
            ),
            "code": (
                "from kaizen import Signature, InputField, OutputField\n"
                "\n"
                "class CreditRiskSignature(Signature):\n"
                '    """Assess credit risk from customer features."""\n'
                "    # Bug: InputField defined but not used in the class body\n"
                "    customer_profile: InputField = InputField(description='Customer features')\n"
                "    risk_level: OutputField = OutputField(description='low/medium/high')\n"
                "    explanation: OutputField = OutputField(description='Reasoning')\n"
                "\n"
                "agent = SimpleQAAgent(signature=CreditRiskSignature, model=model)\n"
                "# Calling with customer_profile ignored:\n"
                "result = await agent.forward()  # No arguments passed"
            ),
            "options": [
                "A) OutputField must come before InputField in the class definition",
                "B) agent.forward() is called with no arguments — the customer_profile InputField is never populated. The correct call passes the input: result = await agent.forward(customer_profile=profile_dict). Without the input, the LLM receives no customer data and generates a generic response",
                "C) SimpleQAAgent requires model to be set via agent.configure(), not the constructor",
                "D) Signature classes must inherit from BaseSignature, not Signature",
            ],
            "answer": "B",
            "explanation": (
                "Kaizen's Signature-based agents receive inputs via the forward() method arguments. "
                "Each InputField in the Signature corresponds to a keyword argument. "
                "Calling forward() without arguments means the LLM prompt contains no customer data — "
                "the agent can only generate a generic answer. "
                "The fix: await agent.forward(customer_profile={'income': 45000, 'debt_ratio': 0.42})"
            ),
            "learning_outcome": "Pass InputField values via agent.forward() keyword arguments",
        },
        {
            "id": "5.4.3",
            "lesson": "5.4",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "You are building a RAG system for Singapore MAS regulations. "
                "After embedding and indexing 2,000 regulatory pages, "
                "retrieval returns 5 chunks per query. "
                "A compliance officer asks: 'How do you ensure the answer cites the exact "
                "regulation section, not just a paraphrase?' "
                "Which metadata field must be stored with each chunk, "
                "and how does the agent include it in the response?"
            ),
            "options": [
                "A) Store chunk_text only; regulation sections can be inferred from the content",
                "B) Use a reranker to put the most authoritative chunk first; the first chunk's source is the citation",
                "C) Citations are the compliance officer's responsibility; RAG only needs to answer correctly",
                "D) Store source_url, page_number, and section_heading with each chunk as metadata. Configure the RAG agent's Signature with an OutputField for citations: List[Citation] where Citation includes source, section, and excerpt. Instruct the agent to populate citations from the retrieved chunk metadata, not to generate them from parametric memory",
            ],
            "answer": "D",
            "explanation": (
                "Citations must be grounded in retrieved metadata, not LLM-generated guesses. "
                "If source metadata is not stored with each chunk, the agent cannot cite accurately — "
                "it would hallucinate plausible-sounding section references. "
                "A typed Citation OutputField in the Signature ensures every answer includes "
                "structured, validated citation data. "
                "The agent is instructed: 'cite only from the provided document metadata'."
            ),
            "learning_outcome": "Design RAG chunk metadata schema to enable accurate regulation section citations",
        },
        {
            "id": "5.6.3",
            "lesson": "5.6",
            "type": "output_interpret",
            "difficulty": "intermediate",
            "question": (
                "Your ML agent uses DataExplorer as a tool and receives this alert:\n\n"
                "  ALERT [CORRELATION] column_pair=('income', 'employment_income') r=0.91\n"
                "  ALERT [MISSING] column='credit_score' null_fraction=0.34\n"
                "  ALERT [CONSTANT] column='currency' unique_values=1\n\n"
                "The agent must recommend feature engineering actions for each alert. "
                "What is the correct recommendation for each alert type?"
            ),
            "options": [
                "A) Drop all flagged columns; alerts always mean the feature is useless",
                "B) CORRELATION (r=0.91): keep income, drop employment_income (or create ratio feature). MISSING (34%): add binary indicator is_credit_score_null; impute with median. CONSTANT (currency=1): drop immediately — a constant feature has zero information and will cause issues in some models",
                "C) CORRELATION: keep both columns. MISSING: drop the column. CONSTANT: encode as ordinal",
                "D) Only the CONSTANT alert requires action; the others are informational",
            ],
            "answer": "B",
            "explanation": (
                "CORRELATION: keeping both correlated features is redundant and inflates the feature space. "
                "For income/employment_income, keep the more complete/interpretable one or engineer a ratio. "
                "MISSING (34%): dropping would lose all information. A binary indicator preserves "
                "the MNAR signal (missing = no credit history). Median imputation handles the rest. "
                "CONSTANT: a feature with one unique value has zero variance — it contributes nothing "
                "to any model and may cause division-by-zero in normalisation."
            ),
            "learning_outcome": "Map DataExplorer alert types to correct feature engineering actions",
        },
        {
            "id": "5.7.3",
            "lesson": "5.7",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student implements SupervisorWorkerPattern but worker results are never "
                "collected — the supervisor exits immediately. What is the bug?"
            ),
            "code": (
                "from kaizen_agents import Delegate, SupervisorWorkerPattern\n"
                "\n"
                "supervisor = SupervisorWorkerPattern(\n"
                "    model=os.environ['DEFAULT_LLM_MODEL'],\n"
                "    max_llm_cost_usd=10.0,\n"
                ")\n"
                "\n"
                "# Workers dispatched but not awaited\n"
                "supervisor.dispatch(worker_fe, task='engineer features')\n"
                "supervisor.dispatch(worker_train, task='train model')\n"
                "\n"
                "# Bug: no await on results\n"
                "print('Done')  # exits before workers complete"
            ),
            "options": [
                "A) supervisor.dispatch() is non-blocking — it queues the tasks but does not wait. Add await supervisor.gather_results() or async for result in supervisor.results(): after dispatching to wait for and collect worker outputs before printing 'Done'",
                "B) dispatch() should be run(), not dispatch()",
                "C) Workers must be added to the supervisor before it is instantiated",
                "D) SupervisorWorkerPattern does not support multiple dispatch() calls; use a list",
            ],
            "answer": "A",
            "explanation": (
                "dispatch() is asynchronous fire-and-forget — it submits the task but returns immediately. "
                "Without awaiting completion, the main coroutine continues to 'print Done' and exits "
                "while workers are still running. "
                "gather_results() (or equivalent) blocks until all dispatched workers complete "
                "and aggregates their outputs. "
                "This is the standard producer-consumer pattern for async multi-agent workflows."
            ),
            "learning_outcome": "Await SupervisorWorkerPattern.gather_results() to collect worker outputs before proceeding",
        },
    ],
}

if __name__ == "__main__":
    for q in QUIZ["questions"]:
        print(f"\n{'=' * 60}")
        print(f"[{q['id']}] ({q['type']}) — Lesson {q['lesson']}  [{q['difficulty']}]")
        print(f"{'=' * 60}")
        print(q["question"])
        if q.get("code"):
            print(f"\n```python\n{q['code']}\n```")
        for opt in q["options"]:
            print(f"  {opt}")
        print(f"\nAnswer: {q['answer']}")
