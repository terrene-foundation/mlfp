# Module 5: LLMs, AI Agents & RAG Systems

**Kailash**: Kaizen (Delegate, BaseAgent, Signature), kailash-ml (6 ML agents) | **Scaffolding**: 30%

## Lecture (3h)
- **5A** Transformer Architecture & LLMs: encoder-decoder, self/cross-attention, positional encodings (RoPE, ALiBi), tokenization (BPE, WordPiece), scaling laws, inference optimization (KV-cache, speculative decoding, quantization)
- **5B** RAG Architecture: chunking strategies, embedding models, hybrid retrieval, re-ranking, evaluation (faithfulness, relevance, correctness), graph RAG, agentic RAG
- **5C** Agent Architecture: perception-reasoning-action loop, Signature-based contracts (InputField/OutputField), CoT/ReAct/RAG agents, multi-agent A2A, ML agents, agent safety (prompt injection, cost budgets)

## Lab (3h) — 6 Exercises
1. Delegate + SimpleQAAgent with typed Signature contracts
2. ChainOfThoughtAgent reasoning about clustering results from Module 4
3. ReActAgent with tools (DataExplorer, FeatureEngineer, ModelVisualizer)
4. RAGResearchAgent over SDK + Singapore regulatory docs with quality evaluation
5. ML Agent Pipeline: DataScientistAgent → FeatureEngineerAgent → ModelSelectorAgent
6. Multi-agent A2A orchestration: research → analyze → engineer → review

## Datasets
Same e-commerce + credit datasets (agents reason over familiar data), Kailash SDK docs, Singapore AI Verify corpus
