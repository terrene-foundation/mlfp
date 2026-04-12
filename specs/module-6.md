### MODULE 6: Machine Learning with Language Models and Agentic Workflows

**Description**: Build LLM applications, fine-tune models, deploy governed agents. All engineering, all code. Following R5 Deck 6B (10 fine-tuning techniques, agentic design) — adapted from CrewAI to Kaizen.

**Module Learning Objectives**: By the end of M6, students can:
- Use LLMs effectively with prompt engineering and structured output
- Fine-tune LLMs using LoRA and adapter layers (from scratch implementation)
- Survey all 10+ fine-tuning techniques and know when to use each
- Align models using DPO and GRPO
- Build RAG systems with proper evaluation
- Build ReAct agents with tool use and cost budgets
- Orchestrate multi-agent systems and implement MCP servers
- Implement AI governance with PACT (access controls, operating envelopes)
- Deploy full production platforms with Nexus

**Kailash Engines**: Kaizen (Delegate, BaseAgent, Signature, agents), kailash-align (AlignmentPipeline, AdapterRegistry), kailash-pact (GovernanceEngine, PactGovernedAgent), kailash-nexus (Nexus, auth, middleware), kailash-mcp

---

#### Lesson 6.1: LLM Fundamentals, Prompt Engineering, and Structured Output

**Prerequisites**: M5 complete (DL architectures, transformers)
**Spectrum Position**: Semantic feature learning — language models as feature extractors at scale

**Topics**:
- **LLM Foundation** (from Deck 6B):
  - Transformer architecture recap (from M5.4)
  - Pre-training: next token prediction (GPT), masked language modelling (BERT)
  - Scaling laws: parameters, data, compute
  - Notable models: GPT, Claude, Gemini, Llama, Phi, Mistral, Gemma (from Deck 6B)
  - RLHF overview: how production LLMs are aligned (connects M5.8 RL to alignment)
- **Prompt Engineering** (new, from completeness audit):
  - Zero-shot prompting: task description only
  - Few-shot prompting: provide examples
  - Chain-of-thought (CoT): "Let's think step by step"
  - Zero-shot CoT: append "Let's think step by step" without examples
  - Self-consistency: sample multiple CoT paths, majority vote
  - Structured prompting: output format specification (JSON, tables)
  - Prompt engineering as the most immediately practical LLM skill
- **Kaizen Structured Output**:
  - Signature: InputField, OutputField
  - Delegate: streaming, events, cost tracking
  - Type-safe structured output (not free-form text)
- **Inference considerations** (brief): KV-cache, speculative decoding, continuous batching (from completeness audit)

**Learning Objectives**: Students can:
- Explain how LLMs are pre-trained and aligned
- Apply 5+ prompt engineering techniques effectively
- Use Kaizen Delegate for structured LLM output with cost tracking
- Understand basic inference optimisation concepts

**Exercise**: Build classification system using Kaizen Delegate. Compare zero-shot vs few-shot vs CoT prompting on same task. Measure cost and accuracy for each approach.

**Assessment Criteria**: All prompting techniques demonstrated. Accuracy comparison shows when each technique helps. Cost tracking working.

**R5 Source**: Deck 6B (LLM foundation models) + PCML6-12 (adapted CrewAI → Kaizen)

---

#### Lesson 6.2: LLM Fine-tuning — LoRA, Adapters, and the Technique Landscape

**Prerequisites**: 6.1 (LLM fundamentals), 5.4 (transformer architecture)
**Spectrum Position**: Customising language models — making them domain-specific

**Topics**:
- **LoRA (Deep Dive)** (from Deck 6B + PCML6-11):
  - Theory: reduce weight updates to low-rank matrices A x B
  - Pre-trained weights remain frozen, new weights stored separately
  - FROM-SCRATCH implementation: LoRALayer with reset_parameters
  - Connects to M4.3 SVD: LoRA IS low-rank factorisation
- **Adapter Layers (Deep Dive)** (from Deck 6B + PCML6-10):
  - Theory: bottleneck modules (FC → activation → FC) inserted between transformer layers
  - FROM-SCRATCH implementation: AdapterLayer, AdapterTransformerModel
  - Task-specific fine-tuning without changing original weights
- **LoRA vs Adapter comparison** (from Deck 6B slide 8):
  - Parameter update mechanism
  - Parameter efficiency
  - Implementation complexity
  - Flexibility and modularity
- **Fine-tuning Landscape Survey** (remaining 8 techniques from Deck 6B, 30-min lecture + reference table):
  - Prefix Tuning: task-specific vectors prepended to attention K/V
  - Prompt Tuning: learnable prompt tokens added to input
  - Task-specific Fine-tuning: full backprop with LR schedulers + gradient clipping + mixed precision
  - LLRD (Layer-wise Learning Rate Decay): lower LR for earlier layers
  - Progressive Layer Freezing: top-down unfreezing
  - Knowledge Distillation: teacher-student with soft labels
  - Differential Privacy: DPSGD, gradient noise injection
  - Elastic Weight Consolidation: Fisher Information Matrix, prevent catastrophic forgetting
- **Model Merging** (new, from completeness audit):
  - TIES: trim, elect sign, merge
  - DARE: drop and rescale
  - SLERP: spherical linear interpolation
  - Task arithmetic: add/subtract fine-tuned weights
  - Application: combine LoRA adapters from different tasks
- **Quantisation** (brief, from completeness audit):
  - GPTQ, AWQ, GGUF, bitsandbytes
  - QLoRA: quantise base model + LoRA on top
  - When to quantise: deployment on constrained hardware
- **kailash-align**: AlignmentPipeline, AlignmentConfig, AdapterRegistry

**Learning Objectives**: Students can:
- Implement LoRA from scratch (understand the mathematics)
- Implement adapter layers from scratch
- Compare LoRA vs adapters across 4 dimensions
- Survey the full fine-tuning landscape and select the right technique
- Explain model merging techniques and when to use them

**Exercise**: Implement LoRA from scratch on IMDB sentiment classification (from PCML6-11). Implement adapter layers from scratch (from PCML6-10). Compare performance, parameter count, training time. Merge two LoRA adapters with TIES.

**Assessment Criteria**: Both from-scratch implementations work. Comparison quantified. Merging produces functional combined model.

**R5 Source**: Deck 6B slides 3-8 (10 techniques) + PCML6-10 (adapter from scratch) + PCML6-11 (LoRA from scratch)

---

#### Lesson 6.3: Preference Alignment — DPO and GRPO

**Prerequisites**: 6.2 (fine-tuning), 5.8 (RL: PPO)
**Spectrum Position**: Aligning models with human preferences — engineering the training signal

**Topics**:
- **RLHF overview**: reward model + PPO. Why it's complex (reward model training, PPO instability).
- **DPO (Direct Preference Optimization)**:
  - Derive from RLHF: bypass the reward model entirely
  - Bradley-Terry preference model: P(y_w > y_l | x) = sigma(beta * (log pi(y_w|x) - log pi_ref(y_w|x)) - beta * (log pi(y_l|x) - log pi_ref(y_l|x)))
  - Implementation: training loop with preference pairs (chosen, rejected)
  - Hyperparameter beta controls deviation from reference policy
- **GRPO (Group Relative Policy Optimization)** (new, from completeness audit):
  - Used in DeepSeek-R1 (2025)
  - Sample multiple completions, score relative to group mean
  - No reward model needed (like DPO), but maintains policy gradient framework
  - Comparison with DPO: when to use each
- **LLM-as-Judge Evaluation**:
  - Use one LLM to evaluate another's outputs
  - Known biases: position bias, verbosity bias, self-enhancement bias
  - Mitigation strategies: swap positions, normalize lengths
- **Evaluation Benchmarks** (from completeness audit):
  - MMLU: multi-task language understanding
  - HellaSwag: commonsense reasoning
  - HumanEval: code generation
  - MT-Bench: multi-turn conversation quality
  - lm-eval-harness: unified evaluation framework
- **kailash-align**: AlignmentPipeline (method="dpo"), evaluator

**Key Formulas**:
- DPO loss: L_DPO = -E[log sigma(beta * log(pi(y_w|x)/pi_ref(y_w|x)) - beta * log(pi(y_l|x)/pi_ref(y_l|x)))]
- GRPO: advantage estimated relative to group mean reward

**Learning Objectives**: Students can:
- Derive DPO from the RLHF objective
- Implement DPO training with preference pairs
- Explain GRPO and when to prefer it over DPO
- Evaluate fine-tuned models using LLM-as-judge and standard benchmarks
- Use lm-eval-harness for systematic evaluation

**Exercise**: Fine-tune a model with DPO on preference data. Evaluate using LLM-as-judge (measure position and verbosity bias). Run lm-eval benchmarks before and after alignment.

**Assessment Criteria**: DPO training converges. LLM-as-judge biases measured and mitigated. Benchmarks show alignment impact.

**R5 Source**: ASCENT (new, not in R5)

---

#### Lesson 6.4: RAG Systems

**Prerequisites**: 6.1 (LLM fundamentals, prompt engineering), 4.6 (NLP, embeddings)
**Spectrum Position**: Knowledge-augmented generation — grounding LLMs in facts

**Topics**:
- RAG concept: Retrieval-Augmented Generation. External knowledge injected into LLM context.
- **Chunking strategies**: fixed size, sentence, paragraph, semantic. Overlap. Chunk size tradeoffs.
- **Retrieval**:
  - Dense retrieval: sentence embeddings, vector similarity (cosine, dot product)
  - Sparse retrieval: BM25 (from M4.6)
  - Hybrid retrieval: combine dense + sparse
  - Re-ranking: cross-encoder scoring
- **RAGAS evaluation framework**: faithfulness, answer relevance, context relevance, context recall
- **HyDE (Hypothetical Document Embeddings)**: generate hypothetical answer, use it for retrieval
- **Advanced RAG patterns**: multi-hop retrieval, document summarisation, metadata filtering
- **Kaizen RAG agents**: RAGResearchAgent, MemoryAgent

**Key Concepts**: Chunking, dense/sparse retrieval, hybrid retrieval, RAGAS evaluation, HyDE

**Learning Objectives**: Students can:
- Build a complete RAG pipeline from documents to answers
- Compare dense, sparse, and hybrid retrieval approaches
- Evaluate RAG quality using RAGAS metrics
- Implement HyDE for improved retrieval

**Exercise**: Build RAG system on Singapore policy documents. Compare BM25, dense, and hybrid retrieval. Evaluate with RAGAS. Implement HyDE and measure improvement.

**Assessment Criteria**: RAG pipeline end-to-end. Three retrieval methods compared. RAGAS metrics computed. HyDE measurably improves retrieval.

**R5 Source**: ASCENT (new, not in R5)

---

#### Lesson 6.5: AI Agents — ReAct, Tool Use, and Function Calling

**Prerequisites**: 6.1 (LLM fundamentals, Kaizen Delegate)
**Spectrum Position**: Autonomous ML — agents that reason and act

**Topics**:
- **Agent concept**: reason about a task, take actions, observe results, iterate
- **ReAct** (Reasoning + Acting): thought → action → observation loop
- **Chain-of-Thought agents**: step-by-step reasoning before acting
- **Tool use**:
  - Custom tools wrapping Kailash engines (DataExplorer, TrainingPipeline as agent tools)
  - Tool selection: agent decides which tool to use
  - Cost budget safety: LLMCostTracker prevents runaway spending
- **Function calling protocol** (from completeness audit):
  - Structured tool schemas (JSON schema definitions)
  - tool_choice parameter: auto, required, specific function
  - Parallel function calling: multiple tools invoked simultaneously
- **Mental framework for agent creation** (from Deck 6B):
  - What is our goal?
  - What is our thought process?
  - What kind of specialist would we hire? (Be sharp: "researcher" vs "HR research specialist")
  - What tools do they need? (Versatile, fault-tolerant, caching)
- **Agent design considerations** (from Deck 6B):
  - Iterative refinement: critic agent that recommends improvements
  - Human-in-the-loop: pause workflow for validation
  - Monitoring and logging: real-time tracking of intermediate outputs
- **Kaizen agents**: ReActAgent, ChainOfThoughtAgent, custom agents with BaseAgent

**Learning Objectives**: Students can:
- Build ReAct agents with custom tools
- Implement function calling with structured schemas
- Apply cost budgets to prevent runaway LLM spending
- Design agents using the mental framework from Deck 6B

**Exercise**: Build data analysis agent with ReAct: wraps DataExplorer, TrainingPipeline, and ModelVisualizer as tools. Agent autonomously explores data, selects model, trains, reports results. Cost budget enforced.

**Assessment Criteria**: Agent reasons through steps (not random tool calls). Tools correctly invoked. Cost budget respected. Results interpretable.

**R5 Source**: Deck 6B (agent design, mental framework, task definition) + PCML6-12 (adapted CrewAI → Kaizen)

---

#### Lesson 6.6: Multi-Agent Orchestration and MCP

**Prerequisites**: 6.5 (single agents, tool use)
**Spectrum Position**: Agent coordination — multiple specialists working together

**Topics**:
- **Multi-agent patterns**:
  - Supervisor-worker: one agent delegates to specialists
  - Sequential: output of one agent feeds into next
  - Parallel: multiple agents work simultaneously, results aggregated
  - Handoff: agent transfers to specialist when topic changes
- **A2A (Agent-to-Agent) protocol**: structured communication between agents
- **Agent memory** (from Deck 6B slide 12):
  - Short-term memory: current conversation context
  - Long-term memory: persistent knowledge across sessions
  - Entity memory: structured knowledge about people, places, concepts
- **MCP (Model Context Protocol)**:
  - Protocol for exposing tools to agents at scale
  - Tool registration: define tools with schemas
  - Transport: stdio, HTTP/SSE
  - Build an MCP server: expose Kailash engines as MCP tools
- **Agent design (from Deck 6B)**: architectural considerations (modularity, load balancing, dynamic agent creation), security (prevent data leakage between agents)

**Learning Objectives**: Students can:
- Implement supervisor-worker, sequential, and parallel multi-agent patterns
- Build an MCP server that exposes ML tools
- Configure agent memory (short-term, long-term, entity)
- Apply security considerations to agent architectures

**Exercise**: Build multi-agent ML pipeline: DataScientist agent → FeatureEngineer agent → ModelSelector agent → ReportWriter agent. Build MCP server exposing ML tools. Test cross-agent coordination.

**Assessment Criteria**: Multi-agent pipeline produces correct results. MCP server functional. Agents communicate structured outputs.

**R5 Source**: Deck 6B (agent architecture, memory, security) + ASCENT

---

#### Lesson 6.7: AI Governance Engineering

**Prerequisites**: 6.6 (multi-agent systems)
**Spectrum Position**: Governed AI — engineering safety and accountability into systems

**Topics**:
- **PACT framework** (engineering focus):
  - D/T/R addressing: Domain/Team/Role structure for access control
  - GovernanceEngine: `compile_org()` to create governance structure
  - `Address`: identify who is requesting access
  - `can_access()`, `explain_access()`: check and explain access decisions
  - Operating envelopes: define boundaries for what agents can do
    - Task envelopes: restrict agent to specific task types
    - Role envelopes: restrict based on role in organisation
    - Monotonic tightening: envelopes can only get stricter, never looser
  - Enforcement modes: warn, block, audit
  - Fail-closed: if governance check fails, deny access (not fail-open)
- **CostTracker**: budget allocation and cascading
  - Budget cascading: parent agent allocates budget to children
  - What happens when budget runs out: agent stops gracefully
- **PactGovernedAgent**: agent wrapper that enforces governance
- **Audit trails**: log every access decision for compliance
- **Clearance levels**: graduated access based on trust level
- **Governance testing**: test that governance WORKS (denied access stays denied)

**Design Note**: This is ENGINEERING. Students implement access controls, test them, and verify they work. No philosophical discussion of AI ethics frameworks. The code IS the governance.

**Learning Objectives**: Students can:
- Implement PACT governance with D/T/R addressing
- Define and enforce operating envelopes for agents
- Implement budget cascading across agent hierarchies
- Test governance rules (verify denied access stays denied)
- Create audit trails for compliance

**Exercise**: Build governed multi-agent system. Define D/T/R structure. Set operating envelopes (task and role). Implement budget cascading. Write governance tests that verify access controls. Generate audit trail.

**Assessment Criteria**: Governance correctly denies unauthorised access. Operating envelopes enforce boundaries. Budget cascading works. Tests verify governance.

**R5 Source**: ASCENT (new, not in R5)

---

#### Lesson 6.8: Capstone — Full Production Platform

**Prerequisites**: All of M6 (6.1-6.7)
**Spectrum Position**: Integration — ship a complete governed AI system

**Topics**:
- **Nexus deployment**: multi-channel (API + CLI + MCP simultaneously)
  - One codebase, three interfaces
  - Auth: RBAC (Role-Based Access Control), JWT tokens
  - Middleware: rate limiting, logging, CORS
  - Plugins: extend Nexus with custom functionality
- **Production monitoring**: DriftMonitor integration from M3.8
- **Full platform integration**: Core SDK → DataFlow → ML → Kaizen → PACT → Nexus → Align
  - Train model (TrainingPipeline)
  - Persist to DataFlow
  - Wrap in agent (Kaizen)
  - Govern agent (PACT)
  - Deploy (Nexus)
  - Monitor (DriftMonitor)
- **Debugging traces**: understanding agent reasoning chains
- **Testing agents**: automated testing for agentic systems
- **Inference optimisation** (brief, from completeness audit): KV-cache, flash attention, vLLM for production serving
- **Multimodal LLMs** (brief mention): vision-language models (GPT-4V, LLaVA, Gemini) as awareness

**Scaffolding**: ~40% (capstone tests integration, not from-scratch). Students connect existing components, not build everything from zero.

**Learning Objectives**: Students can:
- Deploy a complete AI system with Nexus (API + CLI + MCP)
- Implement authentication and authorization
- Integrate all Kailash packages into a production pipeline
- Debug agent reasoning chains
- Monitor deployed models for drift

**Exercise**: Deploy the M6 multi-agent system via Nexus. Add RBAC authentication. Integrate DriftMonitor. Test end-to-end: query via API, CLI, and MCP. Verify governance enforces access controls at deployment level.

**Assessment Criteria**: System deployed and accessible via all 3 channels. Auth works (unauthenticated requests rejected). Drift monitoring active. Governance enforced in production.

**R5 Source**: ASCENT (new, not in R5)

**End of Module Assessment**: Capstone project presentation + comprehensive quiz.
