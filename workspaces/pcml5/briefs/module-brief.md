# Module 5: LLMs, AI Agents & RAG Systems

**Duration**: 7 hours  
**Kailash**: Kaizen (Delegate, BaseAgent, Signature, specialized agents), kailash-ml (6 ML agents)  
**Scaffolding**: 30%  
**API Keys Required**: OPENAI_API_KEY (or GROQ_API_KEY) + DEFAULT_LLM_MODEL in .env  
**Prerequisites**: Modules 1-4 (polars, kailash-ml engines, workflows, deployment)

## Lecture Topics

### 5A: Transformer Architecture & LLMs (90 min)
- **Attention from information retrieval**: derive Q-K-V analogy, scaled dot-product (show √d_k prevents softmax saturation — compute Var(q·k) = d_k), multi-head attention formula
- **Flash Attention** (MUST): IO-aware exact attention — GPU memory hierarchy (HBM vs SRAM), tiling to avoid N×N materialization, mathematically identical result. Flash Attention 3 (2024) with H100 warp specialization. Production deployment literacy.
- **Grouped-Query Attention (GQA)** (MUST): KV-cache memory formula, spectrum MHA → GQA → MQA. Llama 3, Gemma 3, Mistral all use GQA. Essential for inference cost reasoning.
- Positional encodings: sinusoidal (intuition), **RoPE derivation** (MUST: 2D rotation matrix, q^T R_{n-m} k gives relative position — elegant and accessible), ALiBi (linear bias, when to use which)
- Tokenization: BPE (byte-level, GPT-2 style), WordPiece, Unigram (SentencePiece), vocabulary size trade-offs
- Pre-training objectives: masked LM (BERT), causal LM (GPT), span corruption (T5)
- Scaling laws: Chinchilla compute-optimal ratio (derive: tokens ≈ 20× parameters), emergent abilities (mention Schaeffer et al. 2023 counter-argument)
- Inference optimization: KV-cache (explain memory bottleneck), speculative decoding algorithm, continuous batching, PagedAttention (vLLM), quantization (GPTQ, AWQ, GGUF)

### 5B: RAG Architecture & Evaluation (60 min)
- RAG pipeline: chunking (fixed, semantic, recursive, **parent-document retrieval**), embedding models (SBERT, instructor embeddings), vector stores (framework-agnostic)
- Retrieval: dense, sparse (BM25), hybrid, re-ranking (**cross-encoder vs bi-encoder trade-off**)
- **HyDE** (Hypothetical Document Embeddings): generate hypothetical answer, embed it, retrieve similar real documents
- **RAGAS framework** (MUST): faithfulness, answer relevance, context precision, context recall — the industry-standard RAG evaluation
- **Self-RAG, Corrective RAG (CRAG)**: adaptive retrieval — when to retrieve vs when to trust the model
- Advanced RAG: multi-hop, **GraphRAG** (Microsoft), agentic RAG (iterative retrieval)

### 5C: Agent Architecture & Multi-Agent Systems (60 min)
- Agent paradigm: perception-reasoning-action loop, tool use, memory
- Signature-based programming: InputField/OutputField type contracts, structured output
- Specialized agents: CoT (step-by-step), ReAct (reasoning + action), RAG (retrieval-augmented)
- Multi-agent: A2A protocol, supervisor-worker, debate, consensus
- ML agents: LLMs augmenting ML lifecycle (feature suggestion, model selection, drift analysis)
- Agent safety: prompt injection protection, output validation, cost budgets, human-in-the-loop

## Key Import Paths

```python
# Engine-level (recommended — framework-first)
from kaizen_agents import Delegate

# Primitive-level (custom agents)
from kaizen import Signature, InputField, OutputField
from kaizen.core.base_agent import BaseAgent

# Specialized agents
from kaizen_agents.agents.specialized.simple_qa import SimpleQAAgent
from kaizen_agents.agents.specialized.chain_of_thought import ChainOfThoughtAgent
from kaizen_agents.agents.specialized.react import ReActAgent
from kaizen_agents.agents.specialized.rag_research import RAGResearchAgent

# All 6 ML agents
from kailash_ml.agents.data_scientist import DataScientistAgent
from kailash_ml.agents.model_selector import ModelSelectorAgent
from kailash_ml.agents.feature_engineer import FeatureEngineerAgent
from kailash_ml.agents.experiment_interpreter import ExperimentInterpreterAgent
from kailash_ml.agents.drift_analyst import DriftAnalystAgent
from kailash_ml.agents.retraining_decision import RetrainingDecisionAgent
```

## Lab Exercises (6)

### 5.1: Delegate + SimpleQAAgent
- Use `Delegate` for a data analysis Q&A task over e-commerce dataset
- Build a `SimpleQAAgent` with custom Signature for structured domain-specific answers
- Demonstrate InputField/OutputField type contracts
- **Governance from Exercise 1**: set `max_llm_cost_usd` on every agent — mandatory for all M5 exercises. CO methodology: human-on-the-loop, not in-the-loop.

### 5.2: ChainOfThoughtAgent
- Build a `ChainOfThoughtAgent` that reasons step-by-step about Module 4 clustering results
- Compare CoT output quality vs direct answering
- Agent explains WHY segments formed, not just WHAT they are

### 5.3: ReActAgent with Tools
- Build a `ReActAgent` with custom tools for autonomous data exploration
- Agent can call DataExplorer, FeatureEngineer, ModelVisualizer as tools
- Observe the reasoning-action trace and tool selection logic
- **Safety sub-task**: "What happens if you remove the cost budget? What if the agent calls DataExplorer on a 100GB dataset?" — make governance concrete

### 5.4: RAGResearchAgent
- Build `RAGResearchAgent` over Kailash SDK documentation + Singapore regulatory docs
- Evaluate retrieval quality with faithfulness and relevance metrics
- Compare dense vs hybrid retrieval strategies

### 5.5: ML Agent Pipeline (all 6 agents)
- Full ML agent chain: DataScientistAgent → FeatureEngineerAgent → ModelSelectorAgent → **ExperimentInterpreterAgent** → **DriftAnalystAgent** (using M4 drift data) → **RetrainingDecisionAgent**
- Compare LLM-augmented feature/model choices to manual Module 3 choices
- Demonstrate agent confidence scores, cost budget tracking, and the **double opt-in pattern** (AgentInfusionProtocol)

### 5.6: Multi-Agent A2A Coordination
- Full orchestration: research → analyze → engineer → review agents using A2A protocol
- End-to-end autonomous ML pipeline driven by agent coordination
- Compare autonomous result to manual pipeline from Module 3
- **GovernedSupervisor** preview: briefly show how governance wraps multi-agent systems (seeds M6)

## Datasets

- **Same e-commerce + credit datasets** from Modules 3-4 (agents reason over familiar data)
- **Kailash SDK documentation corpus**: For RAG exercise
- **Singapore regulatory corpus**: AI Verify framework, PDPA guidelines (for governance-aware RAG)

**Data source**: `ascent_data/ascent05/` on shared Google Drive.

## Quiz Topics
- Signature InputField/OutputField: "What's wrong with this Signature definition?"
- ReAct vs CoT: "When would you choose ReAct over CoT?"
- RAG evaluation: "Your RAG system has high relevance but low faithfulness. What does this mean?"
- Agent cost budgets: "How does Kailash enforce cost limits on agents?"
- ML agent pipeline: "Which ML agent would you use for feature suggestion vs model selection?"
- Multi-agent A2A: "Draw the message flow for a 3-agent pipeline"

## Deck Opening Case
**BloombergGPT** — Bloomberg built a domain-specific LLM that outperformed GPT-3 on financial tasks despite being 10x smaller. Lesson: specialized agents with domain tools beat general-purpose models. This is why Kaizen's signature-based agents with domain tools are the right architecture.
