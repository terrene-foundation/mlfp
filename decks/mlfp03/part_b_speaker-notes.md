# Module 5: LLMs, AI Agents & Production Deployment -- Speaker Notes

Total time: ~180 minutes (3 hours)

---

## Slide 1: Title Slide

**Time**: ~2 min
**Talking points**:

- Read the provocation: "Bloomberg's model was 10x smaller than GPT-3 -- but beat it on finance. Why?"
- Let it land. This sets up the entire module: domain-specific agents beat giant generalists.
- Today is the most practical module yet -- students will build agents that use tools, search databases, and deploy to production.
- If beginners look confused: "Today we give your ML models a brain that can reason, plan, and act."
- If experts look bored: "We will derive self-attention step by step, cover Flash Attention, and formalise the ReAct loop."
  **Transition**: "Let me show you where we are in the programme..."

---

## Slides 2-5: Recap (Vertical Stack)

**Time**: ~6 min total

### Slide 2: Where We Are

**Time**: ~2 min
**Talking points**:

- Quick table walkthrough. Do not re-teach M1-M4.
- The callout is the hook: "Today: We give your models a brain that can reason, plan, and act."
- If beginners look confused: "You can now train, evaluate, monitor, and serve ML models. But they can only predict. Today they learn to think."

### Slide 3: M4 Recap

**Time**: ~1 min
**Talking points**:

- List the four M4 engines. The callout is key: "They can predict. They cannot reason, search, or decide."

### Slide 4: What Changes Today

**Time**: ~2 min
**Talking points**:

- Two-column comparison: Prediction (M1-M4) vs Reasoning + Action (M5).
- "Input data to output number" vs "input question to plan to act to answer."
- If beginners look confused: "Today's models can ask follow-up questions and search for more information."
- If experts look bored: "This is the transition from function approximation to agentic systems."

### Slide 5: Module 5 Roadmap

**Time**: ~1 min
**Talking points**:

- Walk through the 8-lesson table. Students should note the progression.
- Lessons 5.1-5.4 build individual capabilities. 5.5-5.8 combine and deploy them.
  **Transition**: "Let me tell you about Bloomberg's secret weapon..."

---

## Slides 6-8: Opening Case -- BloombergGPT (Vertical Stack)

**Time**: ~7 min total

### Slide 6: Case Study -- BloombergGPT

**Time**: ~3 min
**Talking points**:

- 50B parameters beat 175B GPT-3 on every finance benchmark.
- The lesson callout: "A smaller model with domain knowledge + specialized tools beats a giant generalist. This is the agent thesis."
- If beginners look confused: "The smarter student who studies the right material beats the one who tries to memorize everything."
- If experts look bored: "The training data mixture (363B finance + 345B general) is the key design decision."

### Slide 7: Why Agents Beat Monoliths

**Time**: ~2 min
**Talking points**:

- Two-column comparison: Monolithic LLM (expensive, frozen) vs Agent System (tools, live data, cheaper).
- Kailash approach: "Kaizen agents wrap any LLM with tools, memory, and structured contracts."
- If beginners look confused: "Instead of building a bigger brain, give a smaller brain better tools."

### Slide 8: The Agent Stack

**Time**: ~2 min
**Talking points**:

- Flow diagram: LLM, Tools, Memory, Contracts.
- Map each to a Kailash primitive: Delegate, ReActAgent, RAGResearchAgent, Signature.
- This table is the roadmap for the entire module. Students should photograph or note it.
  **Transition**: "Let us start with the foundation: what is a language model?"

---

## Slides 9-20: 5.1 LLM Fundamentals (Vertical Stack)

**Time**: ~30 min total

### Slide 9: What Is a Language Model?

**Time**: ~3 min
**Talking points**:

- "A language model does exactly one thing: predict the next word."
- The step-box example: "The capital of Singapore is" predicts "Singapore" (77%).
- The autocomplete analogy: "Autocomplete on your phone, but trained on billions of pages."
- If beginners look confused: "It is a very sophisticated autocomplete. That is the entire trick."
- If experts look bored: "The philosophical question of whether next-word prediction yields understanding is one of the deepest in AI."
  **Transition**: "But models do not see words. They see tokens..."

### Slide 10: What Is a Token?

**Time**: ~3 min
**Talking points**:

- Walk through the examples: "unbelievable" = 3 tokens, "Singapore" = 2 tokens, "HDB" = 2 tokens.
- Cost: "You pay per token. A 1,000-word document is roughly 1,300 tokens."
- BPE callout: mention Byte Pair Encoding as the algorithm that decides splits.
- If beginners look confused: "A token is like a syllable -- sometimes a word, sometimes part of a word."
- If experts look bored: "Try tokenizing code versus natural language -- the token counts are very different."

### Slide 11: BPE -- How Tokenization Works

**Time**: ~3 min (THEORY)
**Talking points**:

- Walk through the iterative merging: characters, then "th", then "the", then "in".
- Stop when vocabulary reaches target size (50,000 typical).
- Key property: common words become single tokens, rare words are split.
- If beginners look confused: "BPE starts with individual letters and keeps gluing together pairs it sees a lot."
- If experts look bored: "SentencePiece, WordPiece, and BPE are all variants of the same idea."

### Slide 12: What Is an API Call to an LLM?

**Time**: ~2 min
**Talking points**:

- Show the Delegate code: send a message, get structured text back with cost tracking.
- Kailash Delegate wraps the API call with cost tracking, streaming, structured output.
- If beginners look confused: "You send text in, you get text back, and Kailash tells you how much it cost."
- SWITCH TO LIVE CODING if time allows: make a real Delegate call.

### Slide 13: The Transformer -- Bird's Eye

**Time**: ~3 min
**Talking points**:

- Two key ideas: self-attention and parallel processing.
- Encoder-Decoder split: BERT = encoder, GPT = decoder, T5 = both.
- If beginners look confused: "The Transformer is the architecture behind ChatGPT and every modern AI model."
- If experts look bored: "We are about to derive self-attention fully."
  **Transition**: "Let us look inside: how does self-attention work?"

### Slide 14: Self-Attention -- Step by Step

**Time**: ~4 min (THEORY)
**Talking points**:

- Introduce Q, K, V: "What am I looking for?", "What do I contain?", "What information do I carry?"
- Show the weight matrices: input X times learned W_Q, W_K, W_V.
- The intuition callout: "Q is a question, K is a label, V is the content. Match questions to labels, then read content."
- If beginners look confused: Use the intuition only. Skip the matrix notation.
- If experts look bored: "Note that d_k is typically d_model / h for h heads."

### Slide 15: Computing Attention Scores

**Time**: ~3 min (THEORY)
**Talking points**:

- The master equation: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V.
- Walk through the 4 steps: Score, Scale, Normalize, Aggregate.
- If beginners look confused: "Each word asks 'who should I pay attention to?' and gets a weighted answer."
- If experts look bored: "The softmax creates a distribution over positions -- it is a soft version of a dictionary lookup."

### Slide 16: Why Scale by sqrt(d_k)?

**Time**: ~3 min (THEORY)
**Talking points**:

- Without scaling, dot products grow with dimension, pushing softmax into saturation.
- Show the variance argument: Var(q dot k) = d_k, so scale by sqrt(d_k) to get variance 1.
- The numeric table makes it concrete: at d_k=64, softmax becomes one-hot without scaling.
- If beginners look confused: "Without this division, the model cannot learn. It is a simple fix that makes everything work."
- If experts look bored: "This is the same idea as Xavier/He initialisation -- keep variance stable across layers."

### Slide 17: Scaling -- Numerical Proof

**Time**: ~2 min (THEORY)
**Talking points**:

- Two columns: without scaling (near one-hot, gradient near 0) vs with scaling (smooth, healthy gradients).
- The warning callout: "Without this one division, Transformers cannot train."
- SKIPPABLE if running short -- the previous slide covers the concept.

### Slide 18: Self-Attention -- Complete Derivation

**Time**: ~2 min (THEORY)
**Talking points**:

- Full matrix equations for Q, K, V, A, Output.
- This is reference material. Do not re-derive; just show the complete picture.
- SKIPPABLE if running short.

### Slide 19: Multi-Head Attention

**Time**: ~2 min (THEORY)
**Talking points**:

- Multiple heads in parallel, each with different projections.
- Intuition: each head learns a different relationship (syntax, coreference, sentiment).
- Same total compute, richer representation.
- If beginners look confused: "Think of it as having 8 different pairs of glasses, each highlighting different patterns."

### Slide 20: Positional Encoding

**Time**: ~2 min (THEORY)
**Talking points**:

- Attention is permutation-invariant -- must inject position.
- Sinusoidal (original) vs RoPE (modern) vs ALiBi.
- If beginners look confused: "Without position encoding, the model does not know word order."
- If experts look bored: "RoPE's relative position encoding is what enables length generalisation in Llama."
  **Transition**: "Let me show some advanced optimisations for those interested..."

---

## Slides 21-26: 5.1 Advanced + Flash Attention (Vertical Stack)

**Time**: ~8 min total

### Slide 21: RoPE Deep Dive

**Time**: ~2 min (ADVANCED)
**Talking points**:

- Rotation matrices applied to 2D subspaces. Key property: dot product depends on relative position m-n.
- SKIPPABLE if running short.

### Slide 22: Pre-Training Objectives

**Time**: ~2 min (THEORY)
**Talking points**:

- Table: CLM (GPT), MLM (BERT), Span Corruption (T5).
- For agents: decoder-only (CLM) because agents need to generate.
- If beginners look confused: "GPT predicts the next word. BERT fills in blanks. For agents, we need GPT-style."

### Slide 23: Scaling Laws

**Time**: ~2 min (THEORY)
**Talking points**:

- Chinchilla: C = 6ND. Parameters and data should scale equally.
- The table: GPT-3 under-trained, Chinchilla optimal, Llama 3 intentionally over-trained.
- If beginners look confused: "Bigger is not always better. The right amount of data matters as much as model size."
- If experts look bored: "Llama 3 intentionally over-trains for inference savings -- a practical business decision."

### Slide 24: Emergent Abilities

**Time**: ~2 min (THEORY)
**Talking points**:

- Table of abilities and their scale thresholds.
- The debate callout: "Emergent abilities may be an artifact of discontinuous metrics." (Schaeffer 2023)
- SKIPPABLE if running short.

### Slide 25: Kailash Bridge -- Delegate & Signature

**Time**: ~2 min
**Talking points**:

- Show the code: AnalyzeHDB Signature with InputField and OutputField.
- Delegate wraps the API call. Result is typed and validated.
- SWITCH TO LIVE CODING: show a real Signature call.

### Slide 26: Signature Contracts -- Why Types Matter

**Time**: ~2 min (THEORY)
**Talking points**:

- Without contract: pray the LLM returns JSON. With Signature: enforced structure.
- "Signatures are contracts: the LLM must satisfy the output schema or the call is retried."
- If beginners look confused: "Types prevent the AI from giving you a random mess instead of structured data."
  **Transition**: "Sections 5.1 covered the foundation. Now: how do we make LLMs reason better?"

---

## Slides 21-26 (Flash Attention section -- part of vertical stack above)

### Slide 21 (FA): Flash Attention

**Time**: ~2 min (ADVANCED)
**Talking points**:

- Standard attention materialises n\*n matrix. Flash Attention tiles into SRAM blocks.
- Key insight: IO-aware tiling. Exact, not approximate. 2-4x speedup.
- SKIPPABLE if running short.

### Slide 22 (FA): GQA and MQA

**Time**: ~2 min (ADVANCED)
**Talking points**:

- Table: MHA (32/32), MQA (32/1), GQA (32/8).
- Smaller KV cache = longer context = agents can read more.
- SKIPPABLE if running short.

### Slide 23 (FA): Speculative Decoding

**Time**: ~2 min (ADVANCED)
**Talking points**:

- Draft model generates candidates, target model verifies in parallel.
- 2-3x speedup with exact same output distribution.
- SKIPPABLE if running short.
  **Transition**: "Now let us make LLMs think step by step..."

---

## Slides 27-31: 5.2 Chain-of-Thought (Vertical Stack)

**Time**: ~10 min total

### Slide 27: Chain-of-Thought Reasoning

**Time**: ~3 min
**Talking points**:

- Two columns: direct prompting (wrong) vs CoT (correct).
- The F1 example is great -- accuracy does not uniquely determine F1.
- Why it works: "Each reasoning step constrains the next step."
- If beginners look confused: "Asking the AI to show its work makes it much more accurate."
- If experts look bored: "The computational interpretation is that CoT gives intermediate scratch space."

### Slide 28: CoT -- Formal View

**Time**: ~2 min (THEORY)
**Talking points**:

- Without CoT: P(answer | question). With CoT: marginalise over reasoning chains.
- In practice: sample one chain, do not sum over all.
- If beginners look confused: "The math just says: reasoning helps because it narrows down the possible answers."
- If experts look bored: "The beam search approximation to the marginalisation is an interesting research direction."

### Slide 29: CoT in Kailash -- ChainOfThoughtAgent

**Time**: ~2 min
**Talking points**:

- Show the code: ChainOfThoughtAgent with reasoning_steps, answer, cost.
- SWITCH TO LIVE CODING if time allows.

### Slide 30: Beyond CoT -- Tree and Graph of Thought

**Time**: ~2 min (ADVANCED)
**Talking points**:

- ToT: branch into multiple paths, evaluate, prune. GoT: allow merging.
- Trade-off: CoT ~2x cost, ToT ~5-20x cost. For most tasks, CoT suffices.
- SKIPPABLE if running short.
  **Transition**: "CoT adds reasoning. Now let us add actions..."

---

## Slides 32-40: 5.3 ReAct Agents (Vertical Stack)

**Time**: ~18 min total

### Slide 31: What Is an Agent?

**Time**: ~3 min
**Talking points**:

- Two columns: Chatbot (no tools, might be wrong) vs Agent (with tools, verified from source).
- The HDB price example is local and relatable.
- If beginners look confused: "A chatbot guesses. An agent looks it up."
- If experts look bored: "This is the fundamental distinction between parametric and non-parametric knowledge."

### Slide 32: The ReAct Loop

**Time**: ~3 min
**Talking points**:

- Walk through the flow diagram: Thought, Action, Observation, Thought...
- The loop continues until the agent has enough information.
- If beginners look confused: "The agent thinks about what it needs, does something, looks at the result, and thinks again."
- If experts look bored: "ReAct interleaves reasoning and acting. Pure reasoning (CoT) or pure acting (tool-only) are both worse."

### Slide 33: ReAct -- Formal Definition

**Time**: ~3 min (THEORY)
**Talking points**:

- The policy pi_theta is the LLM. State s_t is the full history.
- Key insight callout: removing thoughts degrades tool-selection accuracy by 15-30%.
- If beginners look confused: Skip the formal definition. Use the previous slide's diagram.
- If experts look bored: "The ablation study on thought removal is one of the strongest arguments for ReAct over pure tool use."

### Slide 34: Tool Use Formalization

**Time**: ~2 min (THEORY)
**Talking points**:

- Each tool is a typed function. Agent must solve selection, argument generation, execution, and state update.
- If beginners look confused: "Tools are like apps on a phone. The agent picks the right app and uses it."
- If experts look bored: "The tool selection problem is itself a classification task within the generation."

### Slide 35: ReAct in Kailash

**Time**: ~3 min
**Talking points**:

- Show the code: ReActAgent with tools, max_steps, cost_budget.
- Cost budget callout: "enforced by Kailash, not by trust."
- SWITCH TO LIVE CODING: show a ReActAgent with real tools.
- If beginners look confused: "cost_budget is a hard limit. If the agent hits $0.50, it stops."

### Slide 36: Agent Safety -- Prompt Injection

**Time**: ~2 min (THEORY)
**Talking points**:

- Table: Direct, Indirect, Data exfiltration -- with attacks and defenses.
- Rule of thumb callout: "Never let user-provided data enter the system prompt without sanitization."
- If beginners look confused: "Bad actors can try to trick your agent. These are the defenses."
- If experts look bored: "Indirect prompt injection via retrieved documents is the hardest to defend against."
- PAUSE for questions. Agent safety concerns generate good discussion.

### Slide 37: Agent Safety in Kailash

**Time**: ~2 min
**Talking points**:

- Show the code: cost budgets, tool allow-lists, output validation.
- "Kailash enforces safety at the framework level, not the prompt level."
  **Transition**: "Agents can act. Now let us give them memory..."

---

## Slides 41-52: 5.4 RAG Systems (Vertical Stack)

**Time**: ~22 min total

### Slide 38: What Is RAG?

**Time**: ~3 min
**Talking points**:

- "Retrieval-Augmented Generation = look up, then answer. Open-book exam."
- Flow: Question, Search, Read, Answer.
- Three reasons RAG beats bigger models: live data, grounding, proprietary access.
- If beginners look confused: "Instead of memorizing everything, the AI looks up the answer."
- If experts look bored: "RAG is particularly powerful for long-tail knowledge and rapidly changing domains."

### Slide 39: RAG Pipeline Architecture

**Time**: ~3 min (THEORY)
**Talking points**:

- Walk through the pipeline: Chunk, Embed, Index, Retrieve, Re-rank, Generate.
- Walk through the table: key decisions at each stage.
- This is a reference slide. Students should note the pipeline.

### Slide 40: Chunking Strategies

**Time**: ~2 min (THEORY)
**Talking points**:

- Four strategies: fixed-size, semantic, document-structure, recursive.
- Each has pros and cons. Recursive is the most robust.
- If beginners look confused: "You need to cut documents into pieces small enough for the AI to use."
- If experts look bored: "Semantic chunking using embedding similarity between adjacent sentences is state-of-the-art."

### Slide 41: Embedding -- Text to Vectors

**Time**: ~2 min (THEORY)
**Talking points**:

- Cosine similarity formula. Table of embedding models with MTEB scores.
- "Match the model to your domain."
- If beginners look confused: "An embedding turns text into a list of numbers where similar meanings are close together."

### Slide 42: Retrieval -- Dense vs Sparse vs Hybrid

**Time**: ~3 min (THEORY)
**Talking points**:

- Table: Sparse (BM25), Dense (embeddings), Hybrid (combine with RRF).
- RRF formula for combining ranked lists.
- If beginners look confused: "Sparse search finds exact words. Dense search finds similar meanings. Hybrid does both."
- If experts look bored: "The RRF k=60 parameter is remarkably robust across tasks."

### Slide 43: Re-Ranking with Cross-Encoders

**Time**: ~2 min (THEORY)
**Talking points**:

- Bi-encoder (fast, separate) vs Cross-encoder (slow, together).
- Production pattern: bi-encoder retrieves top-100, cross-encoder re-ranks to top-5.
- If beginners look confused: "First pass is fast but rough. Second pass is slow but precise."

### Slide 44: RAG Evaluation -- RAGAS Framework

**Time**: ~3 min (THEORY)
**Talking points**:

- Table: Faithfulness, Answer Relevancy, Context Precision, Context Recall.
- Common failure: "High faithfulness but low recall = correct but incomplete."
- If beginners look confused: "RAGAS checks four things: is the answer supported? Is it relevant? Did we find the right documents? Did we miss anything?"
- If experts look bored: "LLM-based evaluation of faithfulness is itself subject to the biases we discuss in M6."
- PAUSE for questions. RAG evaluation is a critical concept.

### Slide 45: HyDE -- Hypothetical Document Embeddings

**Time**: ~2 min (THEORY)
**Talking points**:

- Problem: queries are short, documents are long. Embedding gap hurts retrieval.
- HyDE: generate a hypothetical answer, embed that instead of the query.
- "Retrieval precision improves 10-30% on domain-specific tasks."
- SKIPPABLE if running short.

### Slide 46: Self-RAG and Corrective RAG

**Time**: ~2 min (ADVANCED)
**Talking points**:

- Self-RAG: model decides when to retrieve. CRAG: evaluates retrieved docs, falls back to web search.
- Key insight: "Standard RAG always retrieves. Self-RAG/CRAG retrieve only when uncertain."
- SKIPPABLE if running short.

### Slide 47: Kailash Bridge -- RAGResearchAgent

**Time**: ~2 min
**Talking points**:

- Show the code: RAGResearchAgent with documents, chunk_size, retrieval_method, rerank.
- SWITCH TO LIVE CODING if time allows.
  **Transition**: "Tools are great, but every framework has its own format. MCP solves this..."

---

## Slides 53-58: 5.5 MCP Servers (Vertical Stack)

**Time**: ~8 min total

### Slide 48: MCP -- Model Context Protocol

**Time**: ~3 min
**Talking points**:

- Problem: every framework has different tool APIs. Solution: one universal protocol.
- USB analogy: "USB standardized device connections. MCP standardizes tool connections."
- If beginners look confused: "MCP means you write a tool once and any AI agent can use it."

### Slide 49: MCP Architecture

**Time**: ~2 min (THEORY)
**Talking points**:

- Table: Tools, Resources, Prompts, Transports.
- "Tools have JSON Schema definitions. Any MCP-compatible agent discovers and calls them automatically."

### Slide 50: Building an MCP Server in Kailash

**Time**: ~3 min
**Talking points**:

- Show the code: MCPServer with @server.tool() decorators.
- Two tools: profile_dataset and check_drift. Any MCP client can use them.
- SWITCH TO LIVE CODING if time allows.
  **Transition**: "Now let us see how LLMs can augment the entire ML lifecycle..."

---

## Slides 59-66: 5.6 ML Agent Pipeline (Vertical Stack)

**Time**: ~12 min total

### Slide 51: The 6 ML Agents

**Time**: ~3 min
**Talking points**:

- Walk through the table: DataScientist, FeatureEngineer, ModelSelector, ExperimentInterpreter, DriftAnalyst, RetrainingDecision.
- Each agent has a specific role and uses specific engines.
- If beginners look confused: "Think of these as six specialist AI assistants, each expert in one part of the ML process."

### Slide 52: ML Agent Pipeline Flow

**Time**: ~2 min
**Talking points**:

- Walk through the timeline: 1 through 6.
- The "double opt-in" callout is critical: "Each agent suggests but does not execute."
- If beginners look confused: "The agents advise. You decide."

### Slide 53: ML Agent Code Pattern

**Time**: ~3 min
**Talking points**:

- Show the code: DataScientistAgent.analyze() and ModelSelectorAgent.recommend().
- The recommendation includes reasoning (step-by-step justification).
- SWITCH TO LIVE CODING if time allows.

### Slide 54: Agent Confidence and Traceability

**Time**: ~2 min (THEORY)
**Talking points**:

- Every response includes: recommendation, confidence, reasoning_steps, evidence, cost.
- Low confidence triggers human review. High confidence can be auto-approved with governance.
- If beginners look confused: "The agent tells you how confident it is and why."
  **Transition**: "One agent is useful. Multiple agents working together are powerful..."

---

## Slides 67-75: 5.7 Multi-Agent Orchestration (Vertical Stack)

**Time**: ~18 min total

### Slide 55: Multi-Agent Patterns

**Time**: ~2 min
**Talking points**:

- Four patterns: Supervisor-Worker, Sequential, Parallel, Handoff.
- If beginners look confused: "Think of it as different ways to organise a team."

### Slide 56: Supervisor-Worker Pattern

**Time**: ~3 min (THEORY)
**Talking points**:

- Diagram: supervisor assigns to workers.
- Show the code: SupervisorWorkerPattern with workers dict.
- If beginners look confused: "One boss assigns tasks. Workers do the work and report back."

### Slide 57: A2A -- Agent-to-Agent Protocol

**Time**: ~2 min (THEORY)
**Talking points**:

- Table: Agent Card, Task, Message, Artifact.
- "MCP connects agents to tools. A2A connects agents to other agents."
- If beginners look confused: "A2A is like email between AI agents -- they can send work orders to each other."

### Slide 58: Handoff Pattern

**Time**: ~2 min (THEORY)
**Talking points**:

- Example: Triage, DriftAnalyst, RetrainingDecision.
- Show the HandoffPattern code.
- If beginners look confused: "The agent works until it hits something it cannot handle, then passes to a specialist."

### Slide 59: Debate Pattern

**Time**: ~2 min (THEORY)
**Talking points**:

- Two agents argue, a judge decides.
- Use case: conflicting model requirements (accuracy vs interpretability).
- If beginners look confused: "Two AIs argue and a third AI picks the winner."
- If experts look bored: "Irving's debate framework for AI safety is the theoretical foundation."

### Slide 60: Multi-Agent in Kailash

**Time**: ~3 min
**Talking points**:

- Show code for SequentialPattern and ParallelPattern.
- Emphasise: parallel agents run simultaneously for independent sub-tasks.
- SWITCH TO LIVE CODING if time allows.

### Slide 61: Evaluating Agent Systems

**Time**: ~2 min (ADVANCED)
**Talking points**:

- Table: AgentBench, WebArena, SWE-bench, GAIA.
- "Best agents solve ~30-50% of SWE-bench. Enormous room for improvement."
- SKIPPABLE if running short.
  **Transition**: "We have built agents. Now let us deploy them..."

---

## Slides 76-83: 5.8 Production Deployment with Nexus (Vertical Stack)

**Time**: ~15 min total

### Slide 62: Deploying with Nexus

**Time**: ~3 min
**Talking points**:

- Nexus deploys as API + CLI + MCP simultaneously from one codebase.
- "Why three channels? Developers use APIs. Data scientists use CLIs. AI agents use MCP."
- If beginners look confused: "One code, three ways to access it."
- If experts look bored: "The MCP channel means your deployed service is automatically available as an AI tool."

### Slide 63: Nexus Minimal Example

**Time**: ~3 min
**Talking points**:

- Show the code: @app.endpoint("/predict") creates all three channels.
- Walk through what each channel produces: REST endpoint, CLI command, MCP tool.
- SWITCH TO LIVE CODING: show a Nexus app starting up.

### Slide 64: Three Channels in Action

**Time**: ~2 min
**Talking points**:

- Show the curl command (API), nexus CLI command, and MCP discovery.
- Side-by-side comparison drives home the "one codebase, three audiences" message.

### Slide 65: Authentication and Middleware

**Time**: ~2 min (THEORY)
**Talking points**:

- Show jwt_auth, rbac, custom middleware code.
- "analyst" can predict. Only "admin" can retrain.
- If beginners look confused: "Authentication means only authorised people can use your AI."

### Slide 66: Monitoring in Production

**Time**: ~2 min (THEORY)
**Talking points**:

- Show DriftMonitor integrated with the Nexus endpoint.
- Every prediction is logged for drift monitoring.

### Slide 67: Production Architecture

**Time**: ~2 min (THEORY)
**Talking points**:

- Complete stack: Client, Nexus, Agent, ML Engine.
- Table: Gateway (Nexus), Intelligence (Kaizen), ML (InferenceServer), Monitoring (DriftMonitor), Storage (DataFlow + ModelRegistry).
- Warning callout: "M5 deploys working systems. M6 governs them."
  **Transition**: "Let us map all the engines we learned today..."

---

## Slides 84-88: Engine Deep Dive

**Time**: ~6 min total

### Slide 68: Kailash M5 Engine Map

**Time**: ~2 min
**Talking points**:

- Table of all 9 engines. This is reference material.
- SKIPPABLE if running short.

### Slide 69: Theory-to-Engine Mapping

**Time**: ~2 min
**Talking points**:

- Each theory concept maps to a Kailash engine and API pattern.
- "The math tells you why. The engine tells you how."
  **Transition**: "Time for the labs..."

---

## Slides 89-92: Lab Setup

**Time**: ~8 min total

### Slide 70: Lab Exercises

**Time**: ~3 min
**Talking points**:

- Walk through the 6 exercises. ~30% scaffolding.
- Students write agent configuration, tool definitions, orchestration logic.
- SWITCH TO LIVE CODING: show the file structure.

### Slide 71: Lab Setup

**Time**: ~2 min
**Talking points**:

- Show the environment setup: dotenv, ASCENTDataLoader, Kailash imports.
- Warning: API keys required. All exercises use gpt-4o-mini (lowest cost).

### Slide 72: Exercise Progression

**Time**: ~2 min
**Talking points**:

- Walk through the progression: ex_1 (single call), ex_2 (reasoning), ex_3 (tools), ex_4 (retrieval), ex_5 (full pipeline), ex_6 (orchestration).
- "Open modules/ascent05/local/ex_1.py."
  **Transition**: "Before the lab, let us discuss some scenarios..."

---

## Slides 93-95: Discussion Prompts

**Time**: ~12 min total

### Slide 73: Discussion -- Agent Trust

**Time**: ~5 min
**Talking points**:

- The scenario: ReActAgent recommends dropping an ethnicity feature.
- Three questions: trust, permission, audit.
- PAUSE for class discussion. This is a governance preview for M6.
- If beginners look confused: "Should an AI be allowed to make decisions about what data to use?"
- If experts look bored: "This is the operating envelope question that PACT addresses in M6."

### Slide 74: Discussion -- RAG vs Fine-Tuning

**Time**: ~4 min
**Talking points**:

- 10,000 policy documents. RAG vs fine-tuning.
- Three questions: accuracy, updates, cost.
- Let students debate. Common answer: RAG for most cases (easy to update, cheaper, more traceable).
- SKIPPABLE if running short.

### Slide 75: Discussion -- Cost and Scale

**Time**: ~3 min
**Talking points**:

- Cost calculation: 6 calls _ $0.002 _ 10,000 = $120/day = $3,600/month.
- "Is this reasonable? Which agents could be cached?"
- If beginners look confused: "AI has a running cost. You need to manage it."
  **Transition**: "Let us wrap up..."

---

## Slides 96-99: Synthesis

**Time**: ~8 min total

### Slide 76: Key Takeaways -- Everyone

**Time**: ~2 min
**Talking points**:

- Four core takeaways: next-token prediction, RAG = look up then answer, cost budgets, Nexus three channels.

### Slide 77: Key Takeaways -- Math

**Time**: ~2 min (THEORY)
**Talking points**:

- Self-attention formula, ReAct formalisation, hybrid retrieval, RAGAS metrics.
- SKIPPABLE if running short.

### Slide 78: Key Takeaways -- Experts

**Time**: ~2 min (ADVANCED)
**Talking points**:

- Flash Attention, GQA, speculative decoding, Self-RAG/CRAG, ToT/GoT.
- SKIPPABLE if running short.

### Slide 79: Cumulative Engine Map

**Time**: ~1 min
**Talking points**:

- Show the progression. M5 adds 8+ engines. M6 is next.

### Slide 80: Next -- Module 6

**Time**: ~1 min
**Talking points**:

- "You know how to build powerful agents. Now you need to learn how to control them."
- This provocation sets up M6 perfectly.

---

## Slide 81: Assessment Preview

**Time**: ~3 min
**Talking points**:

- Walk through quiz topics and assignment connections.
- AI-resilient: quizzes require own exercise outputs.
  **Transition**: "Open your laptops. Start with Exercise 5.1."

---

## Time Budget Summary

| Section                             | Slides | Time         |
| ----------------------------------- | ------ | ------------ |
| Title + Recap                       | 1-5    | ~8 min       |
| Opening Case (BloombergGPT)         | 6-8    | ~7 min       |
| 5.1 LLM Fundamentals                | 9-20   | ~30 min      |
| 5.1 Advanced (Flash Attention etc.) | 21-26  | ~8 min       |
| 5.2 Chain-of-Thought                | 27-30  | ~10 min      |
| Break                               | --     | ~10 min      |
| 5.3 ReAct Agents                    | 31-37  | ~18 min      |
| 5.4 RAG Systems                     | 38-47  | ~22 min      |
| 5.5 MCP Servers                     | 48-50  | ~8 min       |
| 5.6 ML Agent Pipeline               | 51-54  | ~12 min      |
| 5.7 Multi-Agent Orchestration       | 55-61  | ~18 min      |
| 5.8 Nexus Deployment                | 62-67  | ~15 min      |
| Engine Deep Dive                    | 68-69  | ~6 min       |
| Lab Setup                           | 70-72  | ~8 min       |
| Discussion                          | 73-75  | ~12 min      |
| Synthesis + Assessment              | 76-81  | ~11 min      |
| **Total**                           |        | **~203 min** |

**Note**: To fit 180 minutes:

- Skip all ADVANCED slides (21, 22, 23, 24, 30, 46, 61): saves ~14 min
- Compress the self-attention derivation (slides 14-18) to show only the final formula + sqrt(d_k) explanation: saves ~6 min
- Shorten discussion to one scenario: saves ~4 min
- Total savings: ~24 min, bringing it to ~179 min.

**Mark as skippable**: Slides 17 (Scaling Proof), 18 (Complete Derivation), 21 (RoPE), 22 (GQA/MQA), 23 (Speculative Decoding), 24 (Emergent Abilities), 30 (ToT/GoT), 45 (HyDE), 46 (Self-RAG/CRAG), 61 (Agent Evaluation), 68 (Engine Map table), 74 (RAG vs Fine-Tuning discussion), 77 (Math Summary), 78 (Expert Summary).

**Best break point**: After 5.2 Chain-of-Thought (~65 minutes in). Second option: after 5.4 RAG (~115 minutes in).
