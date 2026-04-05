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
                "B) (1) Model name is hardcoded — must use os.environ['DEFAULT_LLM_MODEL']; (2) max_llm_cost_usd is missing — this is mandatory for all Module 5 exercises to prevent runaway spending",
                "C) The import path is wrong; Delegate should be imported from kaizen.agents",
                "D) 500MB datasets cannot be processed by Delegate; use DataExplorer first",
            ],
            "answer": "B",
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
                "A) Delegate — more API calls means more thorough analysis",
                "B) SimpleQAAgent — 1 API call per query vs 8 means ~8× lower cost at 10,000 queries/day. The structured Signature output (typed InputField/OutputField) also enables downstream processing and validation. Delegate's autonomous multi-step reasoning is valuable for complex one-off analysis, not high-volume standardised queries",
                "C) Delegate — structured output is not important for customer queries",
                "D) Both are identical in cost; Delegate just shows its work",
            ],
            "answer": "B",
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
                "A) Raw Delegate — more flexible and handles edge cases better",
                "B) Custom Signature with OutputField(type=RiskExplanation) — the output is schema-validated at runtime so you can guarantee the agent returns exactly the required fields with correct types. For a regulated service, this means you never get malformed JSON causing a downstream crash, and the schema is auditable by the regulator",
                "C) Both are equivalent; parse the Delegate output with json.loads()",
                "D) Signature agents cannot produce JSON output; use a post-processing function",
            ],
            "answer": "B",
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
                "B) The trace shows ReAct's Reason-Act-Observe loop: the agent produces an intermediate thought, calls a tool, observes the result, and reasons again before acting. This is preferable to Delegate when you want to see and audit the exact tool calls and intermediate observations — useful for regulated workflows where explainability of the agent's decision process is required",
                "C) ReAct cannot call DataExplorer tools; only Delegate can access ML tools",
                "D) The trace shows ReAct made two unnecessary tool calls; one call would suffice",
            ],
            "answer": "B",
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
                "B) The tools parameter is missing from ReActAgent — pass tools=[tool_profile_data] to register the function; without it, the agent has no tools to call and falls back to pure LLM reasoning",
                "C) ReActAgent cannot use async tools; make tool_profile_data synchronous",
                "D) The function signature must include 'self' to be registered as an agent tool",
            ],
            "answer": "B",
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
                "B) Overlap prevents context loss at chunk boundaries — a sentence about 'the penalty' makes no sense if 'the regulation' was in the previous chunk. Chunks that are too small (e.g., 100 tokens) cause the retrieved context to lack enough information for the LLM to answer, leading to hallucination as the model fills in missing context",
                "C) Overlap is only needed for code documents; regulatory PDFs do not need it",
                "D) Smaller chunks are always better because retrieval precision increases",
            ],
            "answer": "B",
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
                "A) MCPServer must be imported from kailash.mcp_server.server, not kailash.mcp_server",
                "B) server.register_tool() is missing — the function must be wrapped in MCPTool and registered before server.start(); without registration, the server starts but advertises an empty tool list to clients",
                "C) StdioTransport cannot be used for multi-agent scenarios; use SSETransport",
                "D) MCPToolResult must include a status field; content alone is insufficient",
            ],
            "answer": "B",
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
                "B) SSETransport (Server-Sent Events) — it runs as an HTTP server on a fixed port, allowing multiple clients to connect independently over a network. StdioTransport is subprocess-based (one client per pipe) and cannot serve multiple simultaneous clients",
                "C) Both transports support multi-client connections equally",
                "D) MCPServer cannot serve multiple clients; deploy one server per agent",
            ],
            "answer": "B",
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
                "A) No violation — the agent correctly maximised the AUC metric",
                "B) The agent evaluated on training data, not a held-out test set — this is a data leakage violation. The prompt should specify: 'evaluate on the held-out test set only; training set performance is not a valid model selection criterion'. Without this constraint, the agent optimises in-sample fit, producing a model that is overfit and misleading",
                "C) The violation is using AUC instead of F1-score; the prompt should specify F1",
                "D) The violation is using Delegate for model training; only TrainingPipeline may train models",
            ],
            "answer": "B",
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
                "B) SupervisorWorkerPattern — the supervisor orchestrates task decomposition, monitors worker completion, aggregates results, and handles worker failures with retries. Three independent Delegates have no coordination mechanism: if the feature engineering agent fails, the training agent proceeds with stale features",
                "C) SupervisorWorkerPattern only works for sequential pipelines; use asyncio.gather() for concurrent execution",
                "D) SupervisorWorkerPattern requires a shared database; use it only when persisting intermediate results",
            ],
            "answer": "B",
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
                "B) Nexus sessions provide cross-channel consistency (REST, CLI, and MCP share the same session state), automatic expiry, and thread-safe state management across concurrent requests. A Python dict is not thread-safe and is lost when the process restarts or when load-balanced across multiple instances",
                "C) Nexus sessions are only needed when using MCP; REST can use a dict",
                "D) Sessions and dicts are equivalent; use a dict for simplicity in production",
            ],
            "answer": "B",
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
