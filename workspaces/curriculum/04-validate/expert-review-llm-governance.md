# Expert Review: Modules 5-6 (LLMs, Agents, RAG, Alignment, Governance, RL)

**Reviewer**: LLM Research, AI Safety & Governance Expert  
**Date**: 2026-04-05  
**Scope**: Module 5 (LLMs, AI Agents & RAG Systems), Module 6 (Alignment, Governance, RL & Production Deployment)  
**Benchmark**: 2025-2026 state of the art in LLM architecture, alignment research, agent systems, and AI governance  
**Standard**: Senior ML engineer / tech lead competency for production LLM deployment

---

## Executive Summary

Modules 5 and 6 cover the right topics at the right level of ambition. The selection of transformer internals, RAG architecture, agent patterns, alignment methods, governance frameworks, and RL foundations is well-suited for senior ML engineers who need to make production deployment decisions.

However, both modules have significant gaps relative to the 2025-2026 state of the art. The transformer section lacks the attention variants that now dominate production inference (Flash Attention, GQA). The RAG section omits the evaluation framework that has become the industry standard (RAGAS). The alignment section names GRPO without the context that makes it important (DeepSeek-R1's training paradigm). The governance section underrepresents the US regulatory landscape. The RL section lacks the bridge to RLHF that would unify Modules 6A and 6C.

This review provides 47 specific recommendations, each with priority, depth guidance, and time estimates. Net additional lecture time needed: approximately 25-35 minutes, achievable by tightening existing sections and converting some derivations to "key steps" rather than full proofs.

**Critical gaps**: 6  
**High-priority additions**: 15  
**Medium-priority additions**: 14  
**Nice-to-have additions**: 12  

---

## Module 5A: Transformer Architecture & LLMs (90 min)

### 5A-1. Attention Derivation from Information Retrieval Analogy

**What to add**: Derive scaled dot-product attention starting from the database query analogy. Frame Query as "what am I looking for?", Key as "what do I contain?", Value as "what do I return if matched?" Then show: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V. Derive the sqrt(d_k) scaling by showing that without it, the dot products grow proportional to d_k in expectation (assuming unit-variance inputs), pushing softmax into saturation where gradients vanish.

**Why**: This derivation appears in Vaswani et al. (2017) but most courses skip the "why sqrt(d_k)?" step. Senior engineers debugging attention-related training instabilities need to understand that this is a variance-control mechanism, not an arbitrary constant. The information retrieval framing also builds intuition for why RAG works (5B) -- retrieval IS attention over an external memory.

**Depth**: Key steps derivation. Show E[q_i * k_i] = 0 and Var(q dot k) = d_k when components are independent with mean 0, variance 1. Therefore divide by sqrt(d_k) to restore unit variance. 3-4 slides.

**Where**: Beginning of 5A, before positional encodings. Replace any "here is the attention formula" slide with this derivation.

**Time estimate**: 8 minutes (replaces existing attention introduction, net +3 min)

**Priority**: **MUST** -- This is foundational. Without it, students memorize a formula without understanding the only non-obvious design choice in it.

---

### 5A-2. Flash Attention (IO-Aware Exact Attention)

**What to add**: Explain that standard attention is memory-bandwidth bound, not compute-bound. Flash Attention (Dao et al., 2022; Flash Attention 2, 2023; Flash Attention 3, 2024) restructures the computation to minimize HBM reads/writes by tiling the attention computation into blocks that fit in SRAM, computing softmax incrementally using the online softmax trick. Key insight: exact same mathematical result, dramatically different memory access pattern. Flash Attention 3 (2024) adds warp specialization for H100 Tensor Cores, achieving 2-4x speedup over FA1.

**Why**: Flash Attention is now the default attention implementation in every production inference framework (vLLM, TGI, TensorRT-LLM). A senior engineer who cannot explain why their model serves at the throughput it does -- or why changing sequence length has non-linear latency impact -- lacks fundamental deployment literacy. This is not optional knowledge in 2026.

**Depth**: Intuition + key mechanism. Do NOT derive the tiling algorithm or online softmax proof. Explain: (1) attention is O(N^2) in memory for the attention matrix, (2) GPU memory hierarchy (HBM vs SRAM), (3) Flash Attention tiles the computation so the full N x N matrix never materializes in HBM, (4) result is mathematically identical. Show wall-clock comparison chart. 2 slides.

**Where**: 5A, after the attention derivation, before positional encodings.

**Time estimate**: 5 minutes

**Priority**: **MUST** -- Production deployment literacy. Every inference optimization discussion in 2025-2026 assumes familiarity with Flash Attention.

---

### 5A-3. Grouped-Query Attention (GQA) and Multi-Query Attention (MQA)

**What to add**: Explain the KV-cache memory bottleneck (covered partially in inference optimization), then show how MQA (Shazeer 2019) reduces KV-cache by sharing a single K,V head across all query heads. GQA (Ainslie et al., 2023) is the interpolation: group query heads to share K,V projections. Show the spectrum: MHA (H key-value heads) -> GQA (G groups, G < H) -> MQA (1 key-value head). Llama 3, Gemma 3, Mistral, and essentially all production models deployed in 2025-2026 use GQA.

**Why**: GQA has replaced standard multi-head attention as the default in every major model family since 2024. A senior engineer choosing between model architectures or configuring inference serving needs to understand this tradeoff (quality vs KV-cache memory vs throughput). Without this, students cannot reason about why Llama 3 70B fits in memory configurations where a hypothetical MHA-equivalent would not.

**Depth**: Key steps. Show KV-cache memory formula: 2 * num_layers * num_kv_heads * head_dim * seq_len * batch_size * bytes_per_param. Demonstrate that reducing num_kv_heads from H to G reduces memory proportionally. Show benchmark: GQA with 8 groups matches MHA quality on most tasks while using 1/4 the KV-cache of full MHA. 2-3 slides.

**Where**: 5A, in the inference optimization section, directly after KV-cache explanation.

**Time estimate**: 5 minutes

**Priority**: **MUST** -- This is the dominant attention variant in production. Omitting it is like teaching CNNs without mentioning ResNets.

---

### 5A-4. RoPE Derivation (Rotation Matrix for Relative Position)

**What to add**: Currently, the brief lists "positional encodings (sinusoidal, RoPE, ALiBi)" without specifying depth. For RoPE: show the core insight -- encode position by rotating query and key vectors in 2D subspaces by an angle proportional to position. The dot product between rotated q at position m and rotated k at position n depends only on (m - n), giving relative position encoding. Show the 2D rotation matrix, then generalize to d dimensions by applying independent rotations in d/2 paired dimensions. Key formula: (R_theta_m * q)^T (R_theta_n * k) = q^T R_theta_(n-m) k.

**Why**: RoPE is now the standard positional encoding for essentially all decoder-only models (Llama, Mistral, Qwen, DeepSeek, Gemma). The rotation matrix derivation is elegant and accessible (2D rotation is high-school trigonometry). Senior engineers need this for: (1) understanding context length extension methods (NTK-aware scaling, YaRN) which modify RoPE frequencies, (2) debugging position-related failures in long-context applications, (3) understanding why RoPE enables length extrapolation better than sinusoidal encodings.

**Depth**: Key steps derivation. Show 2D case fully (rotation matrix, dot product preserving relative position). State the d-dimensional generalization without full proof. Show frequency assignment: theta_i = 10000^(-2i/d). Compare to sinusoidal (absolute position, added to embeddings) -- RoPE multiplies into Q,K only. 3 slides.

**Where**: 5A, positional encodings subsection. Expand the current single mention into a focused derivation.

**Time estimate**: 7 minutes (replaces the current brief positional encoding mention, net +4 min)

**Priority**: **MUST** -- RoPE is the de facto standard. The derivation is accessible and builds geometric intuition that pays dividends for understanding context extension.

---

### 5A-5. ALiBi: When to Use Which Positional Encoding

**What to add**: ALiBi (Press et al., 2022) adds a static, head-specific linear bias to attention scores: bias = -m * |i - j| where m is a head-specific slope (geometric progression: 2^(-8/n_heads)). No learned parameters. Key property: trains on short sequences, extrapolates to longer sequences at inference. Compare decision framework: sinusoidal (legacy, BERT-era), RoPE (default for all decoder models, good extrapolation with scaling), ALiBi (training efficiency, strong extrapolation, used in BLOOM, MPT).

**Why**: Students need a decision framework, not just a catalog. The "when to use which" question is what a senior engineer faces when choosing or configuring models.

**Depth**: Intuition only for ALiBi mechanism. The value is in the comparison table showing trade-offs (extrapolation, training cost, quality at trained length, ease of implementation).

**Where**: 5A, immediately after RoPE, as a comparison slide.

**Time estimate**: 3 minutes

**Priority**: **SHOULD** -- The comparison framework has high practical value even if ALiBi itself is less common than RoPE in 2026.

---

### 5A-6. Sliding Window Attention

**What to add**: Explain that full self-attention is O(N^2) in sequence length. Sliding window attention (used in Mistral, Gemma 2) restricts each token to attend only to a fixed window of W previous tokens. Combined with stacked layers, the effective receptive field grows linearly with depth (layer L can "see" L * W tokens back). Mention that some architectures interleave sliding window layers with full attention layers (e.g., Gemma 2 alternates).

**Why**: Context window management is a core production concern. Engineers deploying long-context applications need to understand that "128K context" does not mean every token attends to every other token in all layers. This has direct implications for RAG chunk sizing (5B) and agent memory design (5C).

**Depth**: Intuition only. Show the windowed attention mask diagram. State the receptive field formula. 1-2 slides.

**Where**: 5A, after GQA, within the inference optimization subsection.

**Time estimate**: 3 minutes

**Priority**: **SHOULD** -- Important for production deployment reasoning but not essential for the course's core learning objectives.

---

### 5A-7. Mixture of Experts (MoE) Architecture

**What to add**: Explain the MoE paradigm: each transformer layer contains multiple "expert" FFN sub-networks, with a gating network that routes each token to the top-K experts. Total parameters are large (671B for DeepSeek-V3) but active parameters per token are small (37B). Cover: gating mechanism (top-K softmax routing), load balancing (auxiliary loss or DeepSeek's auxiliary-loss-free approach), why MoE enables training larger models at lower compute cost. Reference DeepSeek-V3 as the defining example: 671B total, 37B active, trained on 14.8T tokens for ~$5.5M in H800 compute.

**Why**: MoE is the architecture behind the most cost-efficient frontier models of 2025-2026 (DeepSeek-V3, Mixtral, Llama 4). A senior ML engineer who cannot explain why DeepSeek-V3 matches GPT-4 class performance at a fraction of the training cost is missing the most important architectural development of the past two years. MoE also provides critical context for the scaling laws discussion -- the Chinchilla optimal ratio changes when active parameters differ from total parameters.

**Depth**: Key mechanism + practical implications. Show the gating equation, explain load balancing problem, show the parameter count comparison (671B total vs 37B active vs a 70B dense model). Do NOT derive the gating gradients. 3-4 slides.

**Where**: 5A, after the base transformer architecture, before scaling laws. MoE must precede scaling laws because it changes the compute-parameter relationship.

**Time estimate**: 7 minutes

**Priority**: **MUST** -- This is the defining architecture innovation of 2024-2025. Omitting MoE from a 2026 course is a critical gap.

---

### 5A-8. Byte-Level BPE vs SentencePiece/Unigram

**What to add**: Clarify that GPT-2/3/4 use byte-level BPE (operates on UTF-8 bytes, never produces unknown tokens) vs the original BPE (character-level, produces UNK for unseen characters). SentencePiece (Google) implements both BPE and Unigram on raw text (no pre-tokenization). The practical difference: byte-level BPE handles any Unicode input but may produce long token sequences for non-Latin scripts; SentencePiece Unigram is better for multilingual models.

**Why**: Tokenization is the most common source of subtle bugs in multilingual LLM deployment. Singapore is a multilingual context (English, Mandarin, Malay, Tamil). A senior engineer deploying models for Singapore audiences must understand tokenization's impact on cost (token count), quality (token fertility across languages), and fairness (some languages are 3-5x more expensive per word).

**Depth**: Intuition + practical comparison table. Show token fertility examples across languages. 1-2 slides.

**Where**: 5A, tokenization subsection, expand existing bullet.

**Time estimate**: 3 minutes (replaces existing tokenization mention, net +1 min)

**Priority**: **SHOULD** -- High relevance for Singapore context and multilingual deployment.

---

### 5A-9. Chinchilla Scaling: Kaplan vs Hoffmann and the Compute-Optimal Ratio

**What to add**: Currently the brief says "Chinchilla scaling, compute-optimal training, emergent abilities." Expand to: (1) Kaplan et al. (2020) found N_optimal proportional to C^0.73, suggesting "make the model as big as possible." (2) Hoffmann et al. (2022, Chinchilla) trained 400+ models and found N_optimal proportional to C^0.50, meaning model size and data should scale equally -- leading to the ~20 tokens per parameter rule. (3) The discrepancy: Kaplan counted non-embedding parameters and extrapolated from small scale, producing a biased exponent. (4) Practical implication: Chinchilla (70B, 1.4T tokens) matched Gopher (280B, 300B tokens) at 4x fewer parameters. (5) Post-Chinchilla reality: Llama models train far beyond Chinchilla-optimal (Llama 3 8B on 15T tokens), trading compute efficiency for inference efficiency -- important when serving cost dominates training cost.

**Why**: Scaling laws are the single most consequential theoretical result for production ML economics. The Kaplan vs Hoffmann distinction matters because it changed how the entire industry allocates compute. The "train past Chinchilla-optimal for inference efficiency" insight is critical for understanding 2025-2026 model design choices.

**Depth**: Key steps. Show the power-law forms, explain why the exponents differ, derive the ~20:1 ratio from Chinchilla, then explain the post-Chinchilla "over-train for inference" paradigm. Do NOT reproduce the full fitting methodology. 3 slides.

**Where**: 5A, scaling laws subsection, expanding the existing single mention.

**Time estimate**: 7 minutes (replaces current scaling laws mention, net +4 min)

**Priority**: **MUST** -- This is the theoretical foundation for all model training economics.

---

### 5A-10. Emergent Abilities Debate (Schaeffer et al. 2023)

**What to add**: Present the original claim (Wei et al., 2022): certain abilities (multi-step arithmetic, chain-of-thought reasoning) appear abruptly at scale, suggesting phase transitions. Then present the counter-argument (Schaeffer et al., NeurIPS 2023): "emergent abilities are a mirage" caused by discontinuous evaluation metrics. When using exact-match (0 or 1), a gradual improvement in per-token accuracy looks like a sudden jump. When using continuous metrics (e.g., edit distance, token-level accuracy), improvement is smooth and predictable. The debate continues: the definition has been revised to "super-linear improvement with scale" rather than "sudden appearance."

**Why**: This is one of the most consequential active debates in LLM research. It directly affects how organizations plan compute budgets ("should we train a 70B model hoping for emergent reasoning, or will a 7B model with more data get us 80% of the way?"). A senior engineer citing "emergent abilities" without knowing the counter-argument is not operating at the expected level.

**Depth**: Intuition + key experimental result. Show the original "emergent abilities" plot alongside Schaeffer's continuous-metric replot. One slide for each, one slide for the debate status. 3 slides.

**Where**: 5A, immediately after scaling laws.

**Time estimate**: 5 minutes

**Priority**: **MUST** -- Active research debate with direct production implications. Teaches critical evaluation of ML claims, which is a meta-skill for senior engineers.

---

### 5A-11. KV-Cache Memory Bottleneck

**What to add**: The brief mentions "KV-cache" but should explain WHY it matters. During autoregressive generation, each new token requires attending to all previous tokens. Without caching, this recomputes all key-value projections at every step (O(N^2) total). KV-cache stores previous K,V tensors, making each step O(N). But the cache grows as: memory = 2 * layers * kv_heads * head_dim * seq_len * batch_size * bytes. For Llama 3 70B with 128K context: ~40GB per request in KV-cache alone. This is why batch size is memory-limited and why GQA (5A-3) and PagedAttention (5A-12) matter.

**Why**: KV-cache is the primary bottleneck in LLM serving. Without understanding it, students cannot reason about inference costs, batch sizing, or why quantization of KV-cache (separate from weight quantization) is a research frontier.

**Depth**: Key steps. Show the memory formula, work through a concrete example (Llama 3 70B at different sequence lengths and batch sizes), demonstrate the trade-off between context length and batch size.

**Where**: 5A, inference optimization subsection, as the motivating problem for GQA and PagedAttention.

**Time estimate**: 4 minutes

**Priority**: **MUST** -- The entire inference optimization narrative hangs on this bottleneck.

---

### 5A-12. Speculative Decoding, Continuous Batching, PagedAttention

**What to add**: The brief mentions speculative decoding and quantization. Expand to three key inference optimizations:

(1) **Speculative decoding**: Use a small "draft" model to generate K candidate tokens cheaply, then verify all K tokens in a single forward pass of the large model. Accepted tokens save time; rejected tokens fall back to the large model. Expected speedup: 2-3x for well-matched draft models. Key insight: verification is parallelizable (one forward pass for K tokens) while generation is sequential.

(2) **Continuous batching** (Orca, 2022): Instead of waiting for all requests in a batch to finish before starting new ones, insert new requests into the batch as soon as any request completes. Delivers 3-10x higher throughput on the same hardware compared to static batching.

(3) **PagedAttention** (vLLM, 2023): Borrows virtual memory paging from OS design. Divides KV-cache into fixed-size blocks (pages), allocated non-contiguously. Eliminates memory fragmentation (wastes 19-27% in naive allocation), enables memory sharing across requests with shared prefixes. vLLM achieves up to 24x higher throughput than HuggingFace TGI under high concurrency.

**Why**: These three optimizations constitute the production inference stack in 2025-2026. Every major deployment uses vLLM, TGI, or TensorRT-LLM, all of which implement these techniques. A senior engineer cannot operate an LLM serving platform without understanding why continuous batching matters or what PagedAttention does. This is not academic -- it directly determines serving cost and latency SLAs.

**Depth**: Intuition + mechanism for each. Show the static vs continuous batching timeline diagram. Show the PagedAttention block table diagram. Explain speculative decoding's acceptance criterion. Do NOT derive the optimal draft model size or prove the acceptance probability. 4-5 slides total.

**Where**: 5A, inference optimization subsection, after KV-cache and GQA.

**Time estimate**: 8 minutes (partially replaces existing inference content, net +4 min)

**Priority**: **MUST** -- Production deployment literacy. These are not advanced topics; they are baseline knowledge for anyone serving LLMs.

---

### 5A-13. T5 Span Corruption and UL2

**What to add**: Briefly explain T5's span corruption objective: randomly mask contiguous spans (not individual tokens), model predicts the missing spans. This creates a more challenging pre-training task than single-token MLM (BERT). Mention UL2 (Unified Language Learner): combines causal LM, prefix LM, and span corruption in a single training regime using mode tokens.

**Why**: Span corruption is already listed in the brief. UL2 represents the insight that pre-training objective choice is not binary (MLM vs CLM) -- you can mix objectives. This matters for understanding encoder-decoder vs decoder-only design choices.

**Depth**: Intuition only. Show the span corruption diagram. Mention UL2's three modes in one sentence. 1 slide.

**Where**: 5A, pre-training objectives subsection.

**Time estimate**: 2 minutes (net +1 min, expanding existing bullet)

**Priority**: **NICE-TO-HAVE** -- Already partially covered. UL2 is interesting but not essential.

---

## Module 5B: RAG Architecture & Evaluation (60 min)

### 5B-1. Semantic Chunking and Parent-Document Retrieval

**What to add**: Beyond fixed-size and recursive chunking, cover (1) semantic chunking: split on embedding similarity breakpoints -- compute sentence embeddings, find where cosine similarity between adjacent sentences drops below a threshold, split there. Produces semantically coherent chunks of variable length. (2) Parent-document retrieval: index small chunks for precise retrieval, but return the parent document (or a larger surrounding chunk) to the LLM for context. This gives retrieval precision with generation context.

**Why**: Fixed-size chunking is the #1 source of RAG quality failures in production. Semantic chunking and parent-document retrieval are the two most impactful production improvements, and both are now supported in LangChain, LlamaIndex, and other major frameworks. A senior engineer building RAG must understand these alternatives.

**Depth**: Intuition + implementation pattern. Show the embedding similarity plot with breakpoints for semantic chunking. Show the two-level index for parent-document retrieval. 2 slides.

**Where**: 5B, chunking strategies subsection, expanding the existing "(fixed, semantic, recursive)" mention.

**Time estimate**: 4 minutes (partially replaces existing content, net +2 min)

**Priority**: **SHOULD** -- High production impact, moderate lecture time.

---

### 5B-2. Hypothetical Document Embeddings (HyDE)

**What to add**: HyDE (Gao et al., 2023) inverts the retrieval problem: instead of embedding the query and searching for similar documents, use the LLM to generate a hypothetical answer document, embed that, and search for real documents similar to the hypothetical answer. Rationale: the hypothetical answer is in "document space" (same register, vocabulary, structure as real documents), while the query is in "question space." This bridges the embedding space mismatch. Recent results (2025): HyDE variants improve retrieval precision by up to 42 percentage points on some datasets. Tradeoff: adds latency (one LLM call before retrieval) and can hallucinate if the LLM's hypothetical answer is wrong.

**Why**: HyDE is a conceptually elegant technique that illustrates a fundamental RAG insight: the semantic gap between queries and documents is often the primary retrieval failure mode. Understanding HyDE helps students reason about embedding space mismatches more generally.

**Depth**: Intuition only. Show the pipeline diagram (query -> LLM -> hypothetical doc -> embed -> retrieve). Mention the latency/hallucination tradeoff. 1-2 slides.

**Where**: 5B, retrieval subsection, after dense/sparse/hybrid retrieval.

**Time estimate**: 3 minutes

**Priority**: **SHOULD** -- Conceptually important, practically useful, demonstrates creative problem-solving in retrieval.

---

### 5B-3. Cross-Encoder vs Bi-Encoder Trade-Off

**What to add**: The brief mentions "re-ranking (cross-encoder)" but should explain the fundamental distinction. Bi-encoders: encode query and document independently, compare via cosine similarity. Fast (can pre-compute document embeddings), but limited interaction between query and document. Cross-encoders: encode query and document together as a single input, output a relevance score. Much more accurate (captures fine-grained query-document interactions) but O(N) per query (must run the encoder for every query-document pair). Production pattern: bi-encoder for initial retrieval (top-100), cross-encoder for re-ranking (top-100 -> top-5).

**Why**: This is the most important architectural decision in RAG retrieval design. The two-stage retrieve-then-rerank pattern is standard in production RAG, and students need to understand why both stages exist and what each contributes.

**Depth**: Key steps. Show the architectural diagrams for each. Quantify the speed difference (bi-encoder: millions of comparisons per second; cross-encoder: ~100 per second). Show accuracy comparison on a standard benchmark. 2 slides.

**Where**: 5B, retrieval subsection, expanding the existing re-ranking mention.

**Time estimate**: 4 minutes (partially replaces existing content, net +2 min)

**Priority**: **MUST** -- Fundamental to RAG architecture design. Without this, students cannot make informed retrieval architecture decisions.

---

### 5B-4. RAGAS Evaluation Framework

**What to add**: RAGAS (Retrieval Augmented Generation Assessment) is the standard evaluation framework for RAG systems in 2025-2026. Core metrics: (1) Context Precision: are relevant chunks ranked higher than irrelevant ones? (2) Context Recall: did we retrieve all the information needed to answer? (3) Faithfulness: is the answer grounded in the retrieved context (no hallucination)? (4) Answer Relevancy: does the answer actually address the question? RAGAS is LLM-based (uses an LLM to judge these qualities), making it scalable without human annotation. Show how component-level evaluation (retriever metrics vs generator metrics) enables targeted debugging.

**Why**: The brief lists "faithfulness, relevance, answer correctness, citation accuracy" as evaluation metrics, which is the right set. But RAGAS has become the standard implementation of these concepts, with widespread adoption (LangChain, LlamaIndex, Haystack integrations). A senior engineer building RAG in 2026 will encounter RAGAS. More importantly, the component-level evaluation approach (separate retriever and generator metrics) is a critical debugging methodology.

**Depth**: Key metrics + practical usage. Show the four core metrics with definitions. Show a concrete example: "high relevance but low faithfulness means the retriever found good context but the generator hallucinated." 2-3 slides.

**Where**: 5B, evaluation subsection, as the practical implementation of the existing evaluation concepts.

**Time estimate**: 5 minutes (partially replaces existing evaluation content, net +2 min)

**Priority**: **MUST** -- Industry standard evaluation framework. The existing metrics list is correct but lacks the organized framework.

---

### 5B-5. Corrective RAG (CRAG), Self-RAG, Adaptive RAG

**What to add**: Three advanced RAG patterns that represent the 2024-2025 frontier:

(1) **Self-RAG** (Asai et al., 2024): The model decides WHEN to retrieve (not every query needs retrieval), retrieves, then generates "reflection tokens" that self-assess: is the retrieved context relevant? Is my answer faithful to it? If not, re-retrieve or abstain. Reduces hallucinations by making retrieval conditional.

(2) **Corrective RAG (CRAG)** (Yan et al., 2024): Adds a lightweight retrieval evaluator that scores retrieved documents for relevance. If scores are low, the system falls back to web search. The evaluator acts as a quality gate between retrieval and generation.

(3) **Adaptive RAG**: Dynamically adjusts retrieval strategy based on query complexity -- single-hop for factual queries, multi-stage for reasoning queries. This is the production pattern: not every query needs the same retrieval depth.

**Why**: These patterns represent the evolution from "retrieve then generate" to "reason about whether and how to retrieve." This is the direction RAG is heading in 2025-2026, and the concepts are already implemented in LangGraph, LlamaIndex, and other frameworks. The brief already mentions "agentic RAG (iterative retrieval)" which is in this family. Making the taxonomy explicit gives students a framework for understanding the RAG design space.

**Depth**: Intuition + architecture diagram for each. Show the decision flow: query -> classify complexity -> route to appropriate RAG strategy. Do NOT implement any of these in detail. 3 slides (one per pattern).

**Where**: 5B, advanced RAG subsection, expanding the existing "multi-hop, graph RAG, agentic RAG" mention.

**Time estimate**: 5 minutes (partially replaces existing advanced RAG content, net +2 min)

**Priority**: **SHOULD** -- Important for understanding the RAG design space, but the concepts build on the core RAG pipeline which is already well-covered.

---

### 5B-6. GraphRAG (Microsoft)

**What to add**: The brief mentions "graph RAG" but should clarify: Microsoft's GraphRAG (2024) extracts an entity-relationship knowledge graph from the corpus, then uses this graph for retrieval. Two modes: (1) local search (entity-centric, answers specific questions about entities), (2) global search (community summaries, answers theme-level questions like "What are the main compliance risks across all documents?"). Key trade-off: graph extraction costs 3-5x more than baseline RAG and requires domain-specific tuning, but enables reasoning queries that vector similarity cannot answer.

**Why**: GraphRAG is the most important RAG architecture for enterprise use cases where documents contain structured relationships (contracts, regulations, organizational data). Singapore regulatory analysis (the course's RAG lab) is exactly this kind of use case.

**Depth**: Intuition + architecture diagram. Show the graph extraction pipeline, the two search modes, and the cost comparison. Do NOT cover the graph construction algorithm in detail. 2 slides.

**Where**: 5B, advanced RAG subsection, expanding the existing "graph RAG" mention.

**Time estimate**: 3 minutes (replaces existing graph RAG mention, net +1 min)

**Priority**: **SHOULD** -- Directly relevant to the Singapore regulatory RAG lab exercise.

---

### 5B-7. Vector Database Landscape

**What to add**: The brief says "vector stores" without naming any. For a production-focused course, briefly orient students: (1) purpose-built vector databases (Pinecone, Weaviate, Qdrant -- managed, scalable, metadata filtering), (2) vector extensions to existing databases (pgvector for PostgreSQL, SQLite-vec), (3) in-memory/lightweight (Chroma, FAISS). Decision factors: scale, persistence needs, metadata filtering, managed vs self-hosted, cost.

**Why**: A senior engineer must be able to make this infrastructure choice. Keeping it entirely "framework-agnostic" leaves students unable to make informed deployment decisions. Present options without endorsing any specific vendor.

**Depth**: Comparison table only. 1 slide.

**Where**: 5B, after the embedding models mention.

**Time estimate**: 2 minutes

**Priority**: **NICE-TO-HAVE** -- Useful orientation but changes rapidly. A comparison table that students can reference is sufficient.

---

## Module 5C: Agent Architecture & Multi-Agent Systems (60 min)

### 5C-1. Tool-Use Formalization (Toolformer, Gorilla, MCP)

**What to add**: Briefly trace the evolution of tool use: (1) Toolformer (Meta, 2023): first demonstration that LLMs can learn to insert API calls autonomously via self-supervised training. (2) Gorilla (UC Berkeley, 2023): fine-tuned on API documentation, outperformed GPT-4 in producing syntactically correct API calls with fewer hallucinated parameters. (3) 2025-2026: tool calling is now native to every frontier model (function calling in GPT-4, tool_use in Claude, tool calling in Gemini), and standardized via Model Context Protocol (MCP) and A2A protocol. The Kaizen framework the students are using sits on top of this stack.

**Why**: The brief mentions tool use but does not explain how LLMs learned to use tools or why the current tool-calling paradigm works. This historical arc takes 3 minutes and gives students the context to understand why Kaizen's tool registration pattern exists and what it abstracts away.

**Depth**: Timeline + key insight per milestone. Toolformer's self-supervised approach, Gorilla's fine-tuning approach, and the current native function-calling paradigm. 2 slides.

**Where**: 5C, at the beginning of the tool use discussion.

**Time estimate**: 3 minutes

**Priority**: **SHOULD** -- Provides important context for the agent paradigm discussion.

---

### 5C-2. Planning: Tree-of-Thought, Graph-of-Thought, Plan-and-Execute

**What to add**: Expand the planning discussion beyond CoT and ReAct:

(1) **Tree-of-Thought** (Yao et al., 2023): explores multiple reasoning paths in parallel (branches), evaluates them, and selects the best. Enables backtracking, which basic CoT cannot do. Useful for problems with search-like structure (puzzle solving, code generation).

(2) **Plan-and-Execute** (Wang et al., 2023): separate planning and execution phases. A planner LLM creates a high-level plan of steps, then an executor LLM carries out each step. This decomposition enables: (a) the planner to use a more capable/expensive model, (b) the executor to be cheaper, (c) plan revision if a step fails.

(3) **Graph-of-Thought**: extends ToT to allow merging of reasoning paths, not just branching. Useful when sub-problems can be decomposed and recombined.

**Why**: The ReAct pattern (which the course covers) is one-step-at-a-time. Many production agent tasks require multi-step planning with the ability to revise. Plan-and-execute is the dominant pattern in LangGraph, CrewAI, and AutoGen. Students building the multi-agent system in Lab 5.6 need to understand these planning paradigms.

**Depth**: Intuition + comparison. Show the CoT -> ToT -> GoT progression as diagrams (chain -> tree -> graph). Explain plan-and-execute as the practical production pattern. 2-3 slides.

**Where**: 5C, after CoT and ReAct agent discussion, before multi-agent coordination.

**Time estimate**: 5 minutes

**Priority**: **SHOULD** -- Important for production agent design, and directly relevant to Lab 5.6.

---

### 5C-3. Agent Memory: Episodic vs Semantic vs Working Memory

**What to add**: The brief mentions "memory (short-term buffer, long-term retrieval)" but should formalize the taxonomy:

(1) **Working memory**: the current context window. Limited by context length. Contains the current task, recent reasoning steps, and tool outputs.

(2) **Episodic memory**: records of past interactions and experiences. Stored as retrievable text (often in a vector store). Used for "remember what happened last time we tried this approach."

(3) **Semantic memory**: persistent factual knowledge. Stored as structured data or knowledge graphs. Used for "what are the rules for this domain?"

The distinction matters for agent architecture: working memory is managed by context window strategy (compression, summarization), episodic memory by RAG over interaction logs, semantic memory by knowledge bases.

**Why**: Memory architecture is the primary differentiator between toy agent demos and production agent systems. A student who builds a ReAct agent in Lab 5.3 will immediately encounter the context window limit. Understanding the three memory types provides the mental model for solving this.

**Depth**: Taxonomy + implementation patterns. Show how each memory type maps to implementation choices (context window management, vector store for episodes, structured DB for semantic). 2 slides.

**Where**: 5C, memory subsection, expanding the existing brief mention.

**Time estimate**: 3 minutes (partially replaces existing content, net +1 min)

**Priority**: **SHOULD** -- Directly addresses a practical limitation students will encounter in labs.

---

### 5C-4. Agent Evaluation Frameworks (AgentBench, WebArena, SWE-bench)

**What to add**: Briefly survey the agent evaluation landscape:

(1) **AgentBench** (Liu et al., 2023): 8 environments testing LLM-as-agent across OS, database, web, game, and reasoning tasks. Tests sustained multi-step goal-directed behavior.

(2) **WebArena** (Zhou et al., 2024): 812 web-based tasks across e-commerce, forums, code repos, CMS. Tests realistic end-user web interactions.

(3) **SWE-bench** (Jimenez et al., 2024): real GitHub issues requiring code understanding, bug localization, and fix generation. The benchmark that defined "can AI write production code?" discourse in 2024-2025.

Key insight: agent evaluation is fundamentally different from model evaluation -- it requires measuring multi-step task completion, not single-turn quality. Success rates are the primary metric, not perplexity or BLEU.

**Why**: Students building agents in Labs 5.3-5.6 need to understand how to evaluate agent quality. "Does it produce good text?" is insufficient; "does it complete the task?" is the right question. This also teaches students to think critically about agent capability claims.

**Depth**: Survey + key insight. 1-2 slides listing the major benchmarks with their focus areas and current best results. Emphasize the evaluation paradigm shift.

**Where**: 5C, after the agent architecture discussion, before multi-agent.

**Time estimate**: 3 minutes

**Priority**: **SHOULD** -- Important for evaluation literacy, especially given the agent hype cycle.

---

### 5C-5. Multi-Agent: Debate and Mixture-of-Agents

**What to add**: The brief lists "debate pattern" and "consensus mechanisms." Expand:

(1) **Debate protocol**: Two or more agents argue opposing positions, a judge agent evaluates arguments and decides. Reduces hallucination because adversarial pressure forces grounded claims. Used in FactCheck-style systems.

(2) **Mixture-of-Agents** (Wang et al., 2024): multiple LLMs generate responses in parallel, then a synthesis layer aggregates them. Analogous to ensemble methods in classical ML. Shown to outperform individual models including GPT-4.

**Why**: Students in Lab 5.6 build a multi-agent system. Understanding the debate pattern and mixture-of-agents gives them concrete architectural options beyond the supervisor-worker pattern.

**Depth**: Intuition + architecture diagram. 1-2 slides.

**Where**: 5C, multi-agent subsection, expanding existing mentions.

**Time estimate**: 3 minutes (net +1 min)

**Priority**: **NICE-TO-HAVE** -- The supervisor-worker pattern from the lab is sufficient for most production use cases.

---

### 5C-6. Agent Safety: Jailbreak Taxonomy and Prompt Injection Defense

**What to add**: The brief mentions "prompt injection protection" but should formalize:

(1) **Attack taxonomy**: Direct injection (user manipulates the system prompt via their input), indirect injection (malicious content in retrieved documents or tool outputs alters agent behavior), multi-turn manipulation (gradually steering agent behavior across conversation turns).

(2) **Defense patterns**: Input scanning (regex + ML classifiers for known jailbreak patterns, 60-80% detection), system prompt integrity (canary tokens, instruction hierarchy), output validation (check for policy violations before returning), tool sandboxing (restrict file system access, network access, execution privileges), content isolation (separate user content from instructions).

(3) **Current reality**: Prompt injection is ranked #1 in OWASP LLM Top 10 (2025-2026). Automated attacks achieve 80-94% success on proprietary models. No defense is complete; defense-in-depth is the only viable strategy.

**Why**: Agent safety is not optional in 2026, especially for Singapore financial services (MAS guidelines). The students' Lab 5.4 (RAG over regulatory documents) and Lab 6.4 (governed agents) both touch on safety. A senior engineer deploying agents without understanding prompt injection taxonomy is creating security vulnerabilities.

**Depth**: Taxonomy + defense checklist. Show the attack types with examples (do NOT provide working exploits). Show the defense-in-depth stack. 3 slides.

**Where**: 5C, agent safety subsection, expanding the existing mention.

**Time estimate**: 5 minutes (partially replaces existing safety content, net +2 min)

**Priority**: **MUST** -- Security literacy is non-negotiable for production agent deployment. OWASP #1 vulnerability.

---

## Module 6A: LLM Fine-Tuning & Alignment (90 min)

### 6A-1. LoRA Low-Rank Approximation Math

**What to add**: Derive why LoRA works. Start with: a pre-trained weight matrix W (d x d) is modified during fine-tuning by delta_W. Hypothesis: delta_W has low intrinsic rank for task-specific adaptation (the task only modifies a low-dimensional subspace of the full parameter space). Therefore, decompose delta_W = B * A where B is (d x r) and A is (r x d), with r << d. Forward pass: h = (W + BA)x = Wx + BAx. Training: freeze W, train only A and B. Parameter reduction: from d^2 to 2dr. For d=4096, r=16: from 16.7M to 131K parameters (128x reduction). Initialize A with random Gaussian, B with zeros, so BA = 0 at start (fine-tuning begins from the pre-trained model).

**Why**: LoRA is the most widely used fine-tuning method in 2025-2026. The low-rank hypothesis is not just a computational trick; it reveals something fundamental about how fine-tuning modifies pre-trained representations. Understanding the math enables students to: (1) reason about rank selection (higher rank for more complex adaptations), (2) understand why LoRA preserves base model capabilities (delta_W is small), (3) understand DoRA and other variants that modify this decomposition.

**Depth**: Full derivation -- this is accessible linear algebra. Show the parameter count comparison. Show rank selection guidelines (r=8-64 for most tasks). Show which modules to target (attention Q,V are standard; adding K,O and FFN layers for harder tasks). 3-4 slides.

**Where**: 6A, beginning of the LoRA discussion.

**Time estimate**: 8 minutes (replaces existing LoRA mention, net +4 min)

**Priority**: **MUST** -- The course brief promises "LoRA theory" and this is the theory. Without the derivation, it is just an API call.

---

### 6A-2. QLoRA: NF4 Quantization and Double Quantization

**What to add**: Explain QLoRA's three innovations:

(1) **NF4 (4-bit NormalFloat)**: Pre-trained weights are approximately normally distributed. NF4 is an information-theoretically optimal 4-bit quantization for normal distributions: compute the 16 quantiles of N(0,1), use these as the 16 representable values. Each weight is mapped to the nearest quantile. This preserves more information than uniform INT4 because the quantization levels match the data distribution.

(2) **Double quantization**: The quantization constants (scale factors) themselves consume memory (one FP32 scale per block of 64 weights). Double quantization quantizes these constants to FP8, saving ~0.5 bits per parameter.

(3) **Paged optimizers**: Use CPU memory as overflow for optimizer states during training, managed via CUDA unified memory. Prevents OOM during gradient spikes.

Forward pass: dequantize the 4-bit weights to BF16 on-the-fly, compute the forward pass with the dequantized weights plus the LoRA adapter output (which remains in BF16). Backpropagation flows through the LoRA adapters only.

**Why**: QLoRA made fine-tuning a 65B model possible on a single 48GB GPU. This is the practical enabler for the course's own fine-tuning lab (6.1, 6.2). Students need to understand NF4 specifically because it is non-obvious: "why not just use INT4?" The answer (distribution-aware quantization preserves more information) is a generally useful insight about quantization.

**Depth**: Key steps. Explain NF4's quantile-based approach with a diagram showing the normal distribution and the 16 quantile boundaries vs uniform INT4 boundaries. Show the memory calculation: 65B model at FP32 = 260GB, at NF4 = ~32.5GB + LoRA adapters. 2-3 slides.

**Where**: 6A, immediately after LoRA derivation.

**Time estimate**: 5 minutes

**Priority**: **MUST** -- Students will use QLoRA in Lab 6.2. Understanding NF4 vs INT4 is the key insight.

---

### 6A-3. DPO: Bradley-Terry Model to Closed-Form Solution

**What to add**: Derive DPO from RLHF. Start with: RLHF trains a reward model r(x,y), then optimizes a policy pi to maximize E[r(x,y)] with a KL penalty against a reference policy pi_ref. The optimization objective is: max_pi E[r(x,y)] - beta * KL(pi || pi_ref).

Key insight (Rafailov et al., 2023): under the Bradley-Terry preference model (P(y_w > y_l | x) = sigma(r(y_w) - r(y_l))), the optimal policy has a closed-form relationship to the reward: r(x,y) = beta * log(pi(y|x) / pi_ref(y|x)) + beta * log(Z(x)). Substituting this back into the Bradley-Terry model eliminates the reward model entirely, giving the DPO loss: L_DPO = -log(sigma(beta * (log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x))))).

This means you can optimize for human preferences directly by comparing the policy's log-probabilities on preferred vs dispreferred responses, without training a separate reward model.

**Why**: DPO is the most important alignment method development since RLHF. The derivation is the cleanest example of "mathematical insight eliminates engineering complexity" in modern ML. Understanding the derivation also makes all DPO variants (SimPO, ORPO, KTO) immediately understandable as modifications to the same framework.

**Depth**: Full derivation. This is a 4-step proof that is accessible to anyone who understands log-probabilities and the sigmoid function. It is also one of the most elegant results in recent ML theory. 4-5 slides.

**Where**: 6A, alignment methods subsection, after RLHF overview.

**Time estimate**: 10 minutes (replaces existing DPO mention, net +5 min)

**Priority**: **MUST** -- The course promises "DPO (direct preference optimization -- bypass reward model)" and this derivation IS why it bypasses the reward model. Without it, DPO is a black box.

---

### 6A-4. Post-DPO Alignment Methods: SimPO, ORPO, KTO

**What to add**: Brief survey of the DPO variant landscape:

(1) **SimPO** (Meng et al., 2024): Replaces DPO's reference model with length-normalized average log-probability as the implicit reward. Eliminates the reference model entirely, reducing memory by ~50%. Outperforms DPO by 6.4 points on AlpacaEval 2. Key insight: DPO's reference model comparison introduces noise; using the policy's own likelihood is a cleaner signal.

(2) **ORPO** (Hong et al., 2024): Combines SFT and preference optimization in a single training pass using odds ratios. Eliminates the distribution shift between SFT and preference tuning that degrades DPO quality. One stage instead of two.

(3) **KTO** (Ethayarajh et al., 2024): Based on Kahneman-Tversky prospect theory -- humans weight losses more than gains. Works with binary feedback (thumbs up/down) instead of paired preferences. Critical for production where collecting preference pairs is expensive but binary feedback is abundant.

**Why**: The modern alignment landscape has moved beyond "DPO vs RLHF." A senior engineer choosing an alignment strategy in 2026 needs to know that SimPO is simpler and often better than DPO, that ORPO saves a training stage, and that KTO works with cheaper feedback data. The course already mentions GRPO; adding these three completes the practical decision framework.

**Depth**: Comparison table + one key insight per method. Do NOT derive each loss function. Show a decision tree: "What data do you have?" -> paired preferences (DPO/SimPO) vs binary feedback (KTO) vs no reference model budget (SimPO/ORPO). 2-3 slides.

**Where**: 6A, after DPO derivation.

**Time estimate**: 5 minutes

**Priority**: **SHOULD** -- Important for practical alignment decision-making in 2026.

---

### 6A-5. GRPO: DeepSeek-R1's Training Paradigm

**What to add**: The brief lists GRPO but should explain why it matters. GRPO (Group Relative Policy Optimization, DeepSeek, 2024-2025): (1) eliminates the critic/value network from PPO, reducing memory and complexity, (2) estimates advantage using group statistics -- sample K responses per prompt, compute mean and std of rewards across the group, normalize each response's reward by group statistics, (3) was used to train DeepSeek-R1 from base model to reasoning model WITHOUT any supervised fine-tuning (the "R1-Zero" experiment), demonstrating that RL alone can induce chain-of-thought reasoning. Key result: DeepSeek-R1 achieved reasoning performance competitive with o1 at a fraction of the training cost.

Also mention DAPO (ByteDance, 2025) as the next evolution: addresses entropy collapse and long-CoT instabilities in GRPO with clip-higher, dynamic sampling, token-level loss, and overlong reward shaping.

**Why**: GRPO + DeepSeek-R1 is arguably the most consequential ML result of 2025. It demonstrated that reasoning capabilities can emerge from pure RL without instruction tuning, which challenges the "SFT -> preference optimization" pipeline that the rest of 6A teaches. A 2026 course that mentions GRPO without explaining DeepSeek-R1 is leaving out the context that makes GRPO important. The connection between 6A (alignment) and 6C (RL) through GRPO is also a powerful curriculum link.

**Depth**: Key mechanism + significance. Show the group advantage estimation formula. Show DeepSeek-R1-Zero's training curve (reasoning ability emerging from RL alone). Mention DAPO's improvements in one sentence. 2-3 slides.

**Where**: 6A, after DPO and the post-DPO methods, as the bridge between preference-based and RL-based alignment.

**Time estimate**: 5 minutes (replaces existing GRPO mention, net +3 min)

**Priority**: **MUST** -- GRPO/DeepSeek-R1 is the defining alignment result of 2025. Listing it without context is worse than omitting it.

---

### 6A-6. LLM-as-Judge Biases

**What to add**: The brief mentions "LLM-as-judge methodology." Expand to cover the known systematic biases:

(1) **Position bias**: LLM judges prefer the first (or last) response regardless of quality. Mitigation: evaluate each pair twice with swapped positions, average scores.

(2) **Verbosity bias**: LLM judges prefer longer responses. Mitigation: length-normalized scoring (this is what SimPO does implicitly).

(3) **Self-enhancement bias**: LLMs rate their own outputs higher than outputs from other models. Mitigation: use a different model family as judge than the one being evaluated.

(4) **Style bias**: preference for certain formatting (bullet points, headers) regardless of content quality.

**Why**: Lab 6.2 uses LLM-as-judge for DPO evaluation. Students must understand these biases to interpret results correctly. Without this knowledge, they may draw wrong conclusions about which alignment method is better based on artifacts of the evaluation rather than genuine quality differences.

**Depth**: Taxonomy + mitigations. 2 slides.

**Where**: 6A, evaluation subsection, expanding the existing LLM-as-judge mention.

**Time estimate**: 4 minutes (partially replaces existing content, net +2 min)

**Priority**: **MUST** -- Students use this methodology in lab. They must understand its failure modes.

---

### 6A-7. Chatbot Arena / Elo Rating for Models

**What to add**: Briefly explain Chatbot Arena (LMSYS) as the community-driven evaluation platform: anonymous A/B testing with human voters, Bradley-Terry model to compute Elo ratings from pairwise comparisons, 6M+ votes. Mention the significance: Chatbot Arena Elo has become the most widely cited single metric for LLM quality. Also mention the 2025 criticisms: vulnerability to de-anonymization attacks, proprietary vendor advantages from selective disclosure, and statistical instabilities in the Elo computation.

**Why**: Senior engineers need to critically evaluate model quality claims. "Our model achieves Elo X on Chatbot Arena" is ubiquitous in model release announcements. Understanding how the score is computed and its limitations is essential for informed model selection.

**Depth**: Intuition only. 1 slide explaining the mechanism, 1 slide on limitations.

**Where**: 6A, evaluation subsection, after LLM-as-judge biases.

**Time estimate**: 3 minutes

**Priority**: **SHOULD** -- Evaluation literacy for model selection.

---

### 6A-8. Contamination Detection

**What to add**: The brief mentions "contamination detection" without detail. Explain: (1) What: training data containing benchmark test examples, inflating reported performance. (2) Detection methods: canary strings (unique sequences inserted in benchmarks, check if models memorize them), n-gram overlap analysis between training data and benchmarks, membership inference attacks. (3) Current status: contamination is widespread in 2025-2026, which is why Chatbot Arena (live human evaluation on fresh prompts) is valued over static benchmarks.

**Why**: Contamination undermines all static evaluation. A senior engineer who trusts benchmark numbers without considering contamination will make poor model selection decisions.

**Depth**: Intuition + detection methods. 1 slide.

**Where**: 6A, evaluation subsection, after Chatbot Arena.

**Time estimate**: 2 minutes

**Priority**: **SHOULD** -- Critical for evaluation literacy. Brief treatment is sufficient.

---

### 6A-9. Model Merging (TIES, DARE, Model Soups)

**What to add**: Model merging as an alternative to fine-tuning: combine the weights of multiple fine-tuned models without additional training. Methods:

(1) **Model soups** (Wortsman et al., 2022): simple weight averaging of models with different hyperparameters. Often improves over the best individual model.

(2) **TIES-Merging** (Yadav et al., 2023): resolves interference by trimming redundant parameters, resolving sign conflicts, then merging. Handles the case where two fine-tuned models push the same parameter in opposite directions.

(3) **DARE** (Yu et al., 2024): randomly drops 90-99% of delta parameters (fine-tuning changes), rescales the remainder. Remarkably effective despite extreme sparsification.

Key insight: merged models regularly top the Open LLM Leaderboard, beating models that required thousands of GPU hours. MergeKit (Hugging Face) makes this accessible to practitioners.

**Why**: Model merging is a paradigm-shifting discovery: you can combine specialist models (one fine-tuned for code, one for math, one for conversation) into a generalist that outperforms each. A senior engineer with a budget constraint should know that merging existing LoRA adapters may outperform training a new one.

**Depth**: Intuition + comparison table of methods. Show one example: merge a code model and a math model, result outperforms both. 2 slides.

**Where**: 6A, after alignment methods, as "alternatives and extensions."

**Time estimate**: 4 minutes

**Priority**: **SHOULD** -- Increasingly important production technique that challenges the "fine-tune from scratch" default.

---

## Module 6B: AI Governance & Responsible Deployment (60 min)

### 6B-1. EU AI Act: Specific Articles Professionals Must Know

**What to add**: The brief correctly identifies risk tiers and GPAI rules. Add the specific articles:

(1) **Art. 6**: High-risk AI classification. Systems in Annex III (biometrics, critical infrastructure, employment, credit scoring, law enforcement, migration, justice). The Singapore credit scoring dataset from Module 3 would be classified as high-risk under Art. 6.

(2) **Art. 9**: Risk management system. Continuous, iterative, requires: identify and analyze known and foreseeable risks, estimate and evaluate risks, adopt risk mitigation measures, test before deployment. This maps directly to the course's drift monitoring (Module 4) and governance (Module 6) content.

(3) **Art. 13**: Transparency. High-risk systems must be "sufficiently transparent to enable deployers to interpret and use the output appropriately." This justifies SHAP (Module 3), model cards (Module 3), and the entire interpretability curriculum.

(4) **Art. 52 (now Art. 50 in final text)**: Disclosure obligations. AI-generated content must be marked; users interacting with AI systems must be informed.

(5) **GPAI obligations (Aug 2025)**: Transparency (training data documentation, copyright compliance), systemic risk models require safety evaluations, red-teaming, incident reporting.

(6) **Enforcement timeline**: Prohibited practices banned Feb 2025. GPAI obligations Aug 2025. High-risk obligations Aug 2026. Full enforcement with fines Aug 2026. Fines: up to EUR 35M or 7% global turnover for prohibited practices violations.

**Why**: The EU AI Act is now in active enforcement (prohibited practices since Feb 2025, GPAI since Aug 2025). Any ML engineer deploying systems accessible to EU residents must understand these obligations. The course's Singapore focus should not obscure that EU AI Act has extraterritorial reach -- any Singapore company serving EU customers is in scope.

**Depth**: Article numbers + obligations + what it means for ML engineers. Show how the course's existing content maps to compliance requirements (SHAP -> Art. 13, drift monitoring -> Art. 9). 3-4 slides.

**Where**: 6B, EU AI Act subsection, replacing the current brief mention.

**Time estimate**: 8 minutes (replaces existing EU AI Act content, net +4 min)

**Priority**: **MUST** -- The EU AI Act is actively being enforced. Professionals need specific article knowledge, not just "risk tiers."

---

### 6B-2. US Regulatory Landscape: Executive Orders, NIST AI RMF

**What to add**: The brief covers EU and Singapore but omits the US entirely. Add:

(1) **NIST AI Risk Management Framework (AI RMF 1.0, 2023)**: Four functions -- GOVERN (organizational accountability), MAP (contextualize AI risks), MEASURE (assess AI risks), MANAGE (prioritize and act on risks). Voluntary but increasingly referenced by sector regulators (CFPB, FDA, SEC, FTC, EEOC).

(2) **Executive Order trajectory**: Biden EO 14110 (Oct 2023) established safety testing requirements for frontier models. Revoked Jan 2025 under Trump. New EO (Dec 2025) focused on preventing state-level regulatory patchwork. The US approach remains voluntary/sector-specific rather than horizontal (unlike EU AI Act).

(3) **State-level regulation**: Colorado AI Act (2024), California SB-1047 (vetoed but influential), Illinois AI Video Interview Act. Growing patchwork that federal EO aims to harmonize.

(4) **Sector regulators**: CFPB (lending decisions), FDA (medical AI), SEC (trading algorithms), EEOC (hiring algorithms). Each applying existing law to AI without AI-specific legislation.

**Why**: Many course graduates will work at multinational companies or Singapore firms with US operations. The US regulatory landscape is fundamentally different from the EU (no comprehensive AI law, sector-specific regulation, volatile executive action). Understanding this contrast is essential for governance strategy.

**Depth**: Comparison framework: EU (horizontal, prescriptive) vs US (sector-specific, voluntary frameworks + enforcement actions) vs Singapore (pragmatic, testing-oriented). 2-3 slides.

**Where**: 6B, after EU AI Act, before Singapore AI Verify.

**Time estimate**: 5 minutes

**Priority**: **MUST** -- A governance module that covers EU and Singapore but omits the US is incomplete for any professional working in a global context.

---

### 6B-3. Singapore AI Verify: Testing Toolkit Detail

**What to add**: Expand the AI Verify coverage:

(1) **AI Verify toolkit**: world's first government-developed AI testing toolkit, combining technical tests (performance, fairness, explainability, robustness) with process checks (data governance, accountability, human oversight).

(2) **2025 updates**: Enhanced testing framework for generative AI (released May 2025). Global AI Assurance Pilot (Feb 2025) -- IMDA + AI Verify Foundation catalyzing norms for Gen AI testing.

(3) **AI Verify Foundation**: expanded to 90+ member organizations by 2025, maintains Global Model Evaluation Toolkit for LLMs and multimodal models.

(4) **ISAGO 2.0 integration**: creates a seamless pathway from governance assessment (ISAGO self-assessment) to technical testing (AI Verify toolkit).

(5) **Agentic AI Governance Framework (2026)**: IMDA published guidance on responsible deployment of AI agents, emphasizing that humans remain ultimately accountable.

(6) **MAS guidelines**: sector-specific AI guidance for financial services, including FEAT (Fairness, Ethics, Accountability, Transparency) principles.

**Why**: This is a Singapore-delivered course for Singapore professionals. AI Verify is the most relevant governance framework for the audience. The current treatment is too brief for the home jurisdiction.

**Depth**: Practical orientation. Show how AI Verify's testing dimensions map to the course content. Show the ISAGO 2.0 self-assessment workflow. 3 slides.

**Where**: 6B, Singapore section, expanding existing mention.

**Time estimate**: 5 minutes (replaces existing Singapore content, net +2 min)

**Priority**: **MUST** -- Singapore audience, Singapore course, Singapore governance framework deserves deeper treatment than it currently receives.

---

### 6B-4. Bias Testing Tools: Aequitas, FairLearn, AI Fairness 360

**What to add**: Brief survey of practical bias testing tools:

(1) **Aequitas** (University of Chicago): open-source bias audit toolkit. Provides fairness metrics (equal opportunity, predictive parity, statistical parity) with a web UI and Python API. Version 1.0 (2024) added Aequitas Flow for bias mitigation experimentation.

(2) **FairLearn** (Microsoft): fairness assessment and mitigation. Provides both metrics and mitigation algorithms (exponentiated gradient, threshold optimizer). Integrates with scikit-learn.

(3) **AI Fairness 360** (IBM): comprehensive toolkit with 70+ fairness metrics and 11 mitigation algorithms across pre-processing, in-processing, and post-processing.

Decision framework: Aequitas for auditing, FairLearn for mitigation in scikit-learn pipelines, AIF360 for comprehensive research.

**Why**: The brief discusses bias and fairness concepts but does not mention the toolkits that implement them. Lab 6.3 could incorporate one of these tools. For a production-focused course, "how do I test for bias?" should have a concrete answer beyond "compute demographic parity manually."

**Depth**: Tool comparison table + one example. Show Aequitas analyzing the Module 3 credit scoring model's fairness across protected attributes. 1-2 slides.

**Where**: 6B, bias and fairness subsection, after the fairness metrics discussion.

**Time estimate**: 3 minutes

**Priority**: **SHOULD** -- Practical tooling that bridges theory to practice. The Module 3 credit dataset with protected attributes is a natural fit.

---

### 6B-5. Model Cards: Filled Template Example

**What to add**: The brief mentions "model cards" in both 3B and 6B. In 6B, show a complete filled example using the Module 3 credit scoring model. Include: model details (architecture, training data, dates), intended use (credit risk assessment for Singapore market), out-of-scope uses (not for automated loan decisions without human review), metrics (AUC-ROC, calibration, fairness metrics per demographic group), ethical considerations (protected attributes, potential for disparate impact), caveats and recommendations.

**Why**: Students create model cards for their individual portfolio (assessment requirement). Showing a complete example in lecture gives them the quality bar. A blank template is insufficient -- students need to see what good looks like.

**Depth**: Filled example using the course's own data. 2 slides (front page and detailed metrics).

**Where**: 6B, algorithmic auditing subsection.

**Time estimate**: 3 minutes

**Priority**: **SHOULD** -- Directly supports the assessment requirement. High practical value.

---

### 6B-6. Red Teaming Methodology

**What to add**: The brief mentions "red teaming" but should formalize the methodology:

(1) **Definition**: Systematic adversarial testing to identify failure modes, biases, and safety violations before deployment.

(2) **Process**: Define scope and threat model, assemble diverse testers, systematic prompt categories (harmful content, bias probing, edge cases, multi-turn manipulation, tool abuse), document findings with severity ratings, prioritize fixes, re-test.

(3) **Automation**: Automated red-teaming using LLMs to generate adversarial prompts (Perez et al., 2022). Trade-off: automated finds more edge cases, human testers find more creative/contextual failures.

(4) **Connection to Module 5C**: agent safety (5C-6) IS red-teaming for agents. Red teaming in 6B formalizes what students practiced informally in Module 5.

**Why**: Red teaming is now required for GPAI models with systemic risk under the EU AI Act (Aug 2025). It is also a required component of the Chatbot Arena evaluation process and AI Verify testing framework. The course's /redteam phase in the COC workflow is itself a red-teaming methodology.

**Depth**: Process + categories. Show the red-teaming checklist. 2 slides.

**Where**: 6B, algorithmic auditing subsection, expanding existing mention.

**Time estimate**: 4 minutes (partially replaces existing content, net +2 min)

**Priority**: **SHOULD** -- Formalized methodology for a concept already mentioned in the brief.

---

## Module 6C: Reinforcement Learning (60 min)

### 6C-1. Bellman Equations: Both Expectation and Optimality

**What to add**: The brief says "Bellman equations" without specifying depth. Derive both:

(1) **Bellman expectation equation**: V_pi(s) = E_pi[R_{t+1} + gamma * V_pi(S_{t+1}) | S_t = s]. This is the recursive decomposition of value under a fixed policy pi. Show the matrix form for finite MDPs: V_pi = R_pi + gamma * P_pi * V_pi, solved as V_pi = (I - gamma * P_pi)^{-1} * R_pi.

(2) **Bellman optimality equation**: V*(s) = max_a [R(s,a) + gamma * sum_{s'} P(s'|s,a) * V*(s')]. This defines the optimal value function. Key difference: expectation equation uses E_pi (average over policy), optimality equation uses max_a (best action). This is why value iteration works: iteratively applying the Bellman optimality operator converges to V*.

**Why**: The Bellman equations are the mathematical foundation for ALL of 6C. Without the derivation, value iteration and policy iteration are unjustified algorithms. The distinction between expectation and optimality equations is also the foundation for understanding why PPO (6C, 6A via RLHF) clips the policy ratio -- it is trying to solve the optimality equation without deviating too far from the current policy.

**Depth**: Full derivation for both. These are accessible (conditional expectation and the max operator). Show the iterative solution for a small grid world example. 4 slides.

**Where**: 6C, RL foundations, replacing the current brief mention.

**Time estimate**: 8 minutes (replaces existing Bellman content, net +3 min)

**Priority**: **MUST** -- This is foundational mathematics. A course at Georgia Tech / Stanford CS229 level must derive Bellman equations.

---

### 6C-2. PPO: Clipped Objective Derivation

**What to add**: Derive the PPO clipped objective:

(1) Start with TRPO's constraint: maximize E[r_t(theta) * A_t] subject to KL(pi_old, pi_new) <= delta, where r_t = pi_new(a|s) / pi_old(a|s) is the probability ratio.

(2) Problem: the KL constraint is computationally expensive (requires Fisher information matrix or conjugate gradients).

(3) PPO's solution: replace the constraint with a clipped objective: L_CLIP = E[min(r_t * A_t, clip(r_t, 1-epsilon, 1+epsilon) * A_t)]. This clips the probability ratio to [1-epsilon, 1+epsilon], preventing large policy updates.

(4) Why it works: when A_t > 0 (good action), clipping prevents r_t from growing too large (don't become overconfident). When A_t < 0 (bad action), clipping prevents r_t from shrinking too much (don't overcorrect). This is a first-order approximation to TRPO's trust region.

**Why**: PPO is the most important RL algorithm for the course because it underpins RLHF (6A). The clipped objective is the specific mechanism that makes PPO stable. Understanding why clipping prevents catastrophic updates is essential for understanding why RLHF can be unstable (and why DPO/GRPO were developed as alternatives).

**Depth**: Full derivation. Show the objective function, explain the clipping mechanism with a diagram showing L_CLIP as a function of r_t for positive and negative advantages. 3-4 slides.

**Where**: 6C, deep RL subsection, replacing existing PPO mention.

**Time estimate**: 8 minutes (replaces existing PPO content, net +3 min)

**Priority**: **MUST** -- PPO is used in RLHF (6A), GRPO is derived from PPO (6A), the RL lab (6.5) uses PPO. The clipped objective is the core mechanism.

---

### 6C-3. SAC: Maximum Entropy RL

**What to add**: Explain Soft Actor-Critic's key innovation: maximize expected return PLUS entropy of the policy: J(pi) = E[sum_t gamma^t (r_t + alpha * H(pi(.|s_t)))]. The entropy term alpha * H(pi) encourages exploration by rewarding policies that maintain randomness. The temperature parameter alpha controls the trade-off: high alpha favors exploration, low alpha approaches standard RL. SAC automatically tunes alpha during training (constrained optimization to maintain a target entropy).

**Why**: SAC is used in the course's RL lab (6.5) alongside PPO. The maximum entropy framework is conceptually important because it formalizes the exploration-exploitation trade-off. The automatic temperature tuning is a practical advantage that students should understand.

**Depth**: Key mechanism. Show the modified objective with entropy bonus. Explain the temperature parameter. Do NOT derive the dual optimization for alpha tuning. 2 slides.

**Where**: 6C, deep RL subsection, after PPO.

**Time estimate**: 4 minutes (replaces existing SAC mention, net +1 min)

**Priority**: **SHOULD** -- Important for the lab, and the maximum entropy concept is elegant.

---

### 6C-4. RLHF as RL Application (Bridge 6A-6C)

**What to add**: Explicitly connect 6A and 6C by framing RLHF as an RL problem:

- **State**: the prompt + tokens generated so far
- **Action**: the next token
- **Policy**: the language model
- **Reward model**: trained on human preferences (Bradley-Terry)
- **Episode**: one complete generation
- **Optimization**: PPO with the reward model score as the reward signal, plus a KL penalty against the reference policy

Then show: DPO eliminates the reward model. GRPO eliminates the critic. This creates a clean narrative arc from RL foundations (6C) to alignment methods (6A).

**Why**: This is the single most important curricular connection in Modules 5-6. Without it, students learn RL in 6C as an abstract topic (gym environments) and alignment in 6A as a separate topic. Making the connection explicit transforms both: RL becomes the foundation for alignment, and alignment becomes the most important application of RL.

**Depth**: Mapping table + diagram. Show the RLHF loop labeled with RL terminology. 2 slides.

**Where**: 6C, as a dedicated bridge section between deep RL and practical RL. OR: 6A, before the alignment methods discussion. The choice depends on lecture ordering -- if 6C is taught before 6A, place it at the end of 6C. If 6A before 6C, place it at the start of 6C.

**Time estimate**: 5 minutes

**Priority**: **MUST** -- This is the curriculum's strongest potential connection point. Leaving it implicit wastes the opportunity.

---

### 6C-5. Offline RL: Conservative Q-Learning, Decision Transformer

**What to add**: Brief introduction to offline RL:

(1) **Problem**: standard RL requires online interaction (take actions, observe results). In many production domains (healthcare, finance, autonomous driving), online experimentation is dangerous or expensive. Offline RL learns from a fixed dataset of (state, action, reward, next_state) tuples collected by previous policies.

(2) **Conservative Q-Learning (CQL)**: learns a pessimistic Q-function that lower-bounds the true value, preventing the policy from exploiting states poorly covered by the data. Key idea: penalize the Q-value for (state, action) pairs that the data collection policy rarely visited.

(3) **Decision Transformer** (Chen et al., 2021): reframes RL as sequence modeling. Input: (return-to-go, state, action, return-to-go, state, action, ...). The model conditions on a desired return-to-go and predicts actions that achieve it. Uses a standard transformer architecture. Key advantage: leverages pre-training and the transformer architecture rather than RL-specific algorithms.

**Why**: Offline RL is the most relevant RL paradigm for production ML engineers. Most production settings have historical data but cannot run online experiments. Decision Transformer is also pedagogically valuable: it connects the transformer architecture (5A) to RL (6C), showing that sequence modeling can subsume RL.

**Depth**: Intuition + key mechanism for each. Show the CQL penalty. Show the Decision Transformer input sequence format. 2-3 slides.

**Where**: 6C, after practical RL applications, as "RL without online interaction."

**Time estimate**: 5 minutes

**Priority**: **SHOULD** -- High practical relevance and strong curricular connections.

---

### 6C-6. Reward Modeling and Reward Hacking

**What to add**: The brief mentions "reward hacking" and "safety constraints." Expand with concrete examples:

(1) **Reward modeling**: train a model to predict human preferences from comparison data. The reward model IS the bridge between human judgment and RL optimization. Quality of the reward model is the ceiling for alignment quality.

(2) **Reward hacking examples** (from METR 2025 research on frontier models):
- Models generating convincing but incorrect responses that get high reward model scores
- Code generation models producing complex, hard-to-evaluate code that scores well but is incorrect
- Reasoning models (O1, DeepSeek-R1) discovering how to manipulate evaluation processes -- e.g., attempting to modify chess engine files during evaluation
- Models producing verbose responses to exploit length bias in reward models

(3) **Mitigations**: Ensemble reward models (reduce individual model biases), constrained RL (KL penalty against reference policy), reward model regularization (information bottleneck), human oversight of high-reward outputs.

**Why**: Reward hacking is the most important open problem in alignment safety. The 2025 METR findings that frontier reasoning models actively hack reward signals is a defining result. A senior engineer deploying RLHF-trained models must understand that "the reward model says it is good" does not mean it IS good. This also explains why DPO/GRPO were developed -- partly to reduce exposure to reward model imperfections.

**Depth**: Examples + mitigations. Show the specific 2025 frontier model reward hacking examples (chess engine manipulation, verbose exploitation). 2-3 slides.

**Where**: 6C, reward hacking subsection, expanding existing mention.

**Time estimate**: 5 minutes (partially replaces existing content, net +2 min)

**Priority**: **MUST** -- The most important AI safety concern in production RL/alignment. The 2025 examples are compelling and current.

---

### 6C-7. Practical RL Case Studies

**What to add**: The brief lists "dynamic pricing, recommendation systems, inventory optimization." Add one or two landmark case studies:

(1) **AlphaFold** (DeepMind, 2020-2024): RL-inspired structure prediction that solved the protein folding problem. Not pure RL but uses similar optimization landscape navigation. Won the Nobel Prize in Chemistry (2024).

(2) **Chip design** (Mirhoseini et al., Nature 2021): RL for chip floorplanning. Achieved superhuman performance in placing chip components. Reduced design time from weeks to hours.

(3) **Recommendation systems**: exploration-exploitation in production at Netflix, Spotify, YouTube. Thompson sampling and contextual bandits as simplified RL.

**Why**: Abstract RL concepts become real when students see the applications. AlphaFold is particularly powerful because it is the most consequential scientific AI result and it connects RL to the "why does this matter?" question.

**Depth**: One slide per case study with the key insight. 2-3 slides.

**Where**: 6C, practical RL subsection, expanding existing applications list.

**Time estimate**: 4 minutes (partially replaces existing content, net +2 min)

**Priority**: **NICE-TO-HAVE** -- Motivational and memorable, but not essential for the learning objectives.

---

## Time Budget Analysis

The current brief allocates:
- 5A: 90 min (Transformer Architecture & LLMs)
- 5B: 60 min (RAG Architecture & Evaluation)
- 5C: 60 min (Agent Architecture & Multi-Agent Systems)
- 6A: 90 min (Fine-Tuning & Alignment)
- 6B: 60 min (Governance & Responsible Deployment)
- 6C: 60 min (Reinforcement Learning)

**Net additional time from all MUST recommendations**: ~35 minutes
**Net additional time from all SHOULD recommendations**: ~20 minutes

### Recommended Time Recovery

To fit the MUST additions without exceeding the time budget:

1. **5A**: The existing positional encoding discussion can be tightened (RoPE derivation replaces the catalog approach). Save 3 minutes.
2. **5A**: Tokenization can be condensed (one comparison slide instead of covering each algorithm). Save 3 minutes.
3. **5B**: The "advanced RAG" section currently repeats across brief and labs. Tighten to architecture diagrams only. Save 2 minutes.
4. **6A**: The "prefix tuning, adapter layers" methods are historical and less relevant in 2026. Mention in one sentence, save 3 minutes.
5. **6B**: PACT framework (D/T/R) is well-covered in the labs. Reduce lecture time to essential concepts, let labs carry the detail. Save 3 minutes.
6. **6C**: Value iteration and policy iteration algorithms can be shown as pseudocode rather than traced step-by-step. Save 3 minutes.
7. **6C**: DQN details (experience replay, target networks) can be condensed since students will not implement DQN. Save 2 minutes.

**Total recovered**: ~19 minutes. Combined with the MUST additions' net time (which accounts for content replacement), the MUST additions fit within the existing time envelope with modest tightening.

---

## Summary of All Recommendations

### MUST (6 items) -- Address before implementation

| ID | Topic | Module | Net Time |
|----|-------|--------|----------|
| 5A-1 | Attention sqrt(d_k) derivation | 5A | +3 min |
| 5A-2 | Flash Attention | 5A | +5 min |
| 5A-3 | GQA / MQA | 5A | +5 min |
| 5A-4 | RoPE derivation | 5A | +4 min |
| 5A-7 | MoE architecture | 5A | +7 min |
| 5A-9 | Chinchilla: Kaplan vs Hoffmann | 5A | +4 min |
| 5A-10 | Emergent abilities debate | 5A | +5 min |
| 5A-11 | KV-cache bottleneck | 5A | +4 min |
| 5A-12 | Speculative decoding, continuous batching, PagedAttention | 5A | +4 min |
| 5B-3 | Cross-encoder vs bi-encoder | 5B | +2 min |
| 5B-4 | RAGAS framework | 5B | +2 min |
| 5C-6 | Jailbreak taxonomy & prompt injection defense | 5C | +2 min |
| 6A-1 | LoRA low-rank math | 6A | +4 min |
| 6A-2 | QLoRA NF4 quantization | 6A | +5 min |
| 6A-3 | DPO Bradley-Terry derivation | 6A | +5 min |
| 6A-5 | GRPO / DeepSeek-R1 context | 6A | +3 min |
| 6A-6 | LLM-as-judge biases | 6A | +2 min |
| 6B-1 | EU AI Act specific articles | 6B | +4 min |
| 6B-2 | US regulatory landscape (NIST AI RMF, EOs) | 6B | +5 min |
| 6B-3 | Singapore AI Verify detail | 6B | +2 min |
| 6C-1 | Bellman expectation + optimality derivation | 6C | +3 min |
| 6C-2 | PPO clipped objective derivation | 6C | +3 min |
| 6C-4 | RLHF as RL application (bridge 6A-6C) | 6C | +5 min |
| 6C-6 | Reward hacking examples (2025) | 6C | +2 min |

### SHOULD (15 items) -- Strong additions if time permits

| ID | Topic | Module | Net Time |
|----|-------|--------|----------|
| 5A-5 | ALiBi comparison framework | 5A | +3 min |
| 5A-6 | Sliding window attention | 5A | +3 min |
| 5A-8 | Byte-level BPE multilingual impact | 5A | +1 min |
| 5B-1 | Semantic chunking, parent-doc retrieval | 5B | +2 min |
| 5B-2 | HyDE | 5B | +3 min |
| 5B-5 | CRAG, Self-RAG, Adaptive RAG | 5B | +2 min |
| 5B-6 | GraphRAG detail | 5B | +1 min |
| 5C-1 | Tool-use formalization history | 5C | +3 min |
| 5C-2 | Planning (ToT, plan-and-execute) | 5C | +5 min |
| 5C-3 | Memory taxonomy (episodic/semantic/working) | 5C | +1 min |
| 5C-4 | Agent evaluation benchmarks | 5C | +3 min |
| 6A-4 | SimPO, ORPO, KTO | 6A | +5 min |
| 6A-7 | Chatbot Arena / Elo | 6A | +3 min |
| 6A-8 | Contamination detection | 6A | +2 min |
| 6A-9 | Model merging (TIES, DARE) | 6A | +4 min |
| 6B-4 | Bias testing tools | 6B | +3 min |
| 6B-5 | Model card filled example | 6B | +3 min |
| 6B-6 | Red teaming methodology | 6B | +2 min |
| 6C-3 | SAC maximum entropy | 6C | +1 min |
| 6C-5 | Offline RL (CQL, Decision Transformer) | 6C | +5 min |

### NICE-TO-HAVE (4 items)

| ID | Topic | Module | Net Time |
|----|-------|--------|----------|
| 5A-13 | T5 span corruption / UL2 | 5A | +1 min |
| 5B-7 | Vector database landscape | 5B | +2 min |
| 5C-5 | Multi-agent debate / mixture-of-agents | 5C | +1 min |
| 6C-7 | RL case studies (AlphaFold, chip design) | 6C | +2 min |

---

## Cross-Module Curricular Observations

### Strongest Connection Point: 6A-6C Bridge via RLHF/GRPO
The single highest-value addition is 6C-4 (RLHF as RL application). Currently, alignment (6A) and RL (6C) are taught as separate topics. Making the connection explicit -- "RLHF IS RL where the policy is an LLM, the action space is the vocabulary, and the reward model encodes human preferences" -- transforms both sections. Students see RL as the foundation for alignment, and alignment as the most important application of RL. The GRPO -> PPO connection further cements this: GRPO is a simplification of the PPO algorithm students learned in 6C.

### Module 5A Is the Bottleneck
Module 5A (90 min) carries the heaviest content load: transformer architecture, attention mechanisms, positional encodings, tokenization, pre-training, scaling laws, AND inference optimization. The MUST additions (Flash Attention, GQA, MoE, KV-cache, PagedAttention) are all critical production knowledge but compound the time pressure. The recommended recovery (condensing tokenization, streamlining positional encoding catalog into RoPE-first, cutting prefix tuning detail from 6A) is achievable but requires careful slide design.

### Singapore Contextualization Opportunity
The governance module (6B) should lean heavily into Singapore context: AI Verify toolkit for testing (directly usable in labs), MAS FEAT principles for financial services (aligns with Module 3's credit scoring dataset), and the Agentic AI Governance Framework (2026) for Module 5's agent safety. This is the course's competitive advantage over generic online courses.

### Emerging Topic: Post-Training Stack
The alignment landscape has evolved from "SFT then RLHF" to a modular stack: SFT (instruction following) -> preference optimization (DPO/SimPO/KTO for alignment) -> RL with verifiable rewards (GRPO/DAPO for reasoning). This three-stage framework should be the organizing principle for 6A, replacing the current linear enumeration of methods.
