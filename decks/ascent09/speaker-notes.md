# Module 9: LLMs, AI Agents & RAG — Speaker Notes

Total time: ~180 minutes (3 hours)

---

## Slide 1: LLMs, AI Agents & RAG Systems

**Time**: ~1 min
**Talking points**:

- Read the provocation aloud: "Bloomberg's model was 10x smaller than GPT-3 — but beat it on finance tasks. Size is not intelligence. Architecture is."
- Let it land. Ask the room: "How can a smaller model beat a larger one?" Take 2-3 answers before proceeding.
- If beginners look confused: "Today we build systems where the AI can reason, search for information, and take actions — not just predict numbers."
- If experts look bored: "We'll derive why decoder-only transformers scale the way they do, and why the agent architecture beats monolithic models on domain tasks."

**Transition**: "Let me show you where we've come from before we look at where we're going..."

---

## Slide 2: Where We Are

**Time**: ~2 min
**Talking points**:

- Walk through the cumulative table — M1 through M8. Do not re-teach anything. This is a reference.
- Emphasise: "Everything so far has been about prediction. Today we move to reasoning and action."
- If beginners look overwhelmed: "Don't worry — you don't need to remember all of these today. Today is a fresh start."
- If experts look impatient: "We'll move fast through the recap and get to the new material within five minutes."

**Transition**: "Let me highlight what M8 specifically gave you..."

---

## Slide 3: M8 Recap — Key Capabilities

**Time**: ~2 min
**Talking points**:

- Five bullets: neural networks, attention, transformer architecture, positional encoding, InferenceServer.
- This is the bridge from M8 to M9. M8 built the transformer. M9 turns it into a reasoning system.
- If experts look impatient: "We'll accelerate through the recap. The new material starts in two minutes."
- Emphasise: "You understand how transformers work internally. Now we scale them to billions of parameters and give them tools, knowledge, and actions."

**Transition**: "So what actually changes today compared to every other module?"

---

## Slide 4: What Changes Today

**Time**: ~2 min
**Talking points**:

- Two columns: M1-M8 is prediction (static input-to-output), M9 is reasoning + action (dynamic, tool-using, iterative).
- Make the distinction vivid: "A prediction model is like a calculator. An agent is like an assistant who can use a calculator, a search engine, and a database — and decide which one to use."
- If beginners look confused: "Everything you built before returns a single answer. Today's systems can ask follow-up questions and search for information before answering."
- If experts look bored: "The theoretical shift is from supervised learning to in-context reasoning and planning under uncertainty."

**Transition**: "Here's the map for today..."

---

## Slide 5: Module 9 Roadmap

**Time**: ~1 min
**Talking points**:

- Sections A through H. Give the 30-second overview: "We start with what LLMs are, learn to talk to them effectively, build RAG to give them knowledge, build agents to give them tools, then deploy everything to production."
- If beginners look confused: "Think of it as building a smart assistant, step by step — architecture, then communication, then memory, then action, then deployment."
- Do not go into detail on any section. This is a signpost.

**Transition**: "Let me make this concrete with a real case study before we dive in..."

---

## Slide 6: Case Study: BloombergGPT

**Time**: ~2 min
**Talking points**:

- Tell the story: "Bloomberg — one of the most data-rich companies on Earth — decided to train their own model. They didn't try to build the biggest model. They built a focused one."
- Ask: "Why would a 50B model beat a 175B model?" Expected answer: domain-specific training data. Then add: "And tools. Bloomberg's model could access live financial data via agent calls."
- If beginners look confused: "More data about your specific domain beats more data about everything. Specialisation wins."
- If experts look bored: "This is the empirical foundation for the efficiency thesis — compute-optimal training on domain-specific data, then tool-augmented inference."

**Transition**: "This case demonstrates a broader principle about system design..."

---

## Slide 7: Why Agents Beat Monoliths

**Time**: ~2 min
**Talking points**:

- Analogy: "A monolithic model is like one person who read every book once. An agent is a person with access to a library, a calculator, and a phone."
- Kaizen approach: smaller model + tools + retrieval. Uses live data. Always up-to-date. Cites sources.
- If experts look bored: "The Plan-and-Execute pattern — a frontier model for planning and a smaller model for execution — reduces costs by up to 90%."
- Connect to Kailash: "Kaizen agents wrap any LLM with tools, memory, and typed contracts. You build the specialist, not the generalist."

**Transition**: "Let me show you the four building blocks..."

---

## Slide 8: The Agent Stack

**Time**: ~2 min
**Talking points**:

- Four components: LLM (brain), Tools (hands), Memory (filing cabinet), Contracts (rules of engagement).
- Walk through the Kailash mapping: `Delegate`, `ReActAgent`, `RAGResearchAgent`, `Signature`.
- Mention: "By the end of today, you will have built systems using all four of these."
- If beginners look confused: "Each component is a separate Kailash class. You compose them like building blocks."

**Transition**: "And the context for all of this is a field moving at extraordinary speed..."

---

## Slide 9: The 2025-2026 Landscape

**Time**: ~2 min
**Talking points**:

- Set expectations: "This field moves fast. What I'm teaching today reflects early 2026. The principles will last — the specific model names will change. Focus on the patterns, not the products."
- Four structural shifts: reasoning models, agent systems (1,445% year-over-year growth), MCP protocol (Linux Foundation), A2A protocol.
- If experts look bored: "The structural shift is that five independent model families reached frontier quality simultaneously in 2025. This is not a one-off — it has changed the economics of ML permanently."

**Transition**: "Now let's understand what's actually inside these models. Section A — LLM fundamentals."

---

## Slide 10: A1 — Decoder-Only Transformer

**Time**: ~2 min
**Talking points**:

- Recap from M8: "Remember the transformer had two halves — encoder and decoder. Modern LLMs threw away the encoder. Why? Because predicting the next token is all you need if you have enough data."
- Decoder-only: generates tokens left-to-right, each token attends only to previous tokens, scales to trillions of training tokens.
- If beginners look confused: "Think of it as autocomplete on steroids. It reads what came before and predicts what comes next."
- If experts look bored: "The simplicity of decoder-only is what enables scaling. One architecture, one objective, no cross-attention bottleneck."

**Transition**: "The mechanism that makes this work is causal masking..."

---

## Slide 11: Causal Masking

**Time**: ~1 min
**Talking points**:

- Explain the mask visually: "Imagine a grid. The model can only look down and to the left — never forward. This prevents it from cheating by reading the answer before generating it."
- The equation shows the causal mask M: upper-triangle is negative infinity (masked out), lower-triangle is zero (visible).
- If experts look bored: "This is why decoder-only models are autoregressive — each output depends only on past outputs. The masking enables parallel training while preserving the sequential generation property."
- If beginners look confused: "The practical upshot: the model can't see words that come after the current position. It predicts each word without peeking."

**Transition**: "Modern LLMs make specific choices about how to implement this architecture..."

---

## Slide 12: Key Architectural Choices

**Time**: ~2 min
**Talking points**:

- Walk through the table without going deep: position encoding, attention type, activation, normalisation, FFN layout.
- Frame as engineering choices, not magic: "Think of building a car — you choose the engine, transmission, and tires. These are the equivalent choices for LLMs."
- The Llama recipe (RoPE + GQA + SwiGLU + RMSNorm) has become the de facto open-source standard.
- If experts look bored: "GQA reduces KV cache memory by sharing key-value heads across query groups. DeepSeek's MLA goes further, compressing KV cache by 93% via learned low-rank projection."

**Transition**: "Let's look at why RoPE won the position encoding competition..."

---

## Slide 13: RoPE — Rotary Position Encoding

**Time**: ~1 min
**Talking points**:

- Intuition: "Imagine each token is a clock hand. Its position is encoded by how far the hand has rotated. Two tokens that are close together have similar angles — so their attention is naturally high."
- Key properties: relative position (distance matters, not absolute position), extrapolation to longer sequences, no learned parameters.
- If beginners look confused: "Don't worry about the matrix. The key idea: position is encoded as rotation, and rotation is efficient for the model to compute."
- If experts look bored: "RoPE's extrapolation property enables context window extension techniques like YaRN and LongRoPE. The frequency base can be scaled to extend the effective context window post-training."

**Transition**: "GQA is the memory efficiency trick that makes large models practical..."

---

## Slide 14: GQA — Grouped-Query Attention

**Time**: ~1 min
**Talking points**:

- Three attention variants: MHA (each head has own K/V), MQA (all heads share one K/V), GQA (groups share K/V).
- GQA is the sweet spot: 4-8 groups, near-MHA quality, near-MQA speed.
- For beginners: "The takeaway — newer models are more memory-efficient. A model that needed 4 GPUs in 2023 can run on 1 GPU in 2025."
- If experts look bored: "GQA was the key innovation enabling Llama 2's 70B to run efficiently on consumer hardware. DeepSeek's MLA goes further — it projects K and V into a much smaller latent space, achieving 93.3% KV cache compression."

**Transition**: "Memory efficiency matters because these models are enormous..."

---

## Slide 15: Model Sizes: Parameters and Memory

**Time**: ~2 min
**Talking points**:

- Make it concrete: "A 7B model at INT4 fits on a laptop GPU. A 70B model needs a workstation with 2 GPUs. A 405B model needs a small cluster."
- Rule of thumb: ~1 GB VRAM per billion parameters at INT4, plus 20% overhead for KV cache and activations.
- Ask: "How many of you have a GPU with 8GB VRAM?" — that's enough for a 7B model.
- If experts look bored: "The memory calculation is more nuanced in practice — add activation memory, the optimiser state during fine-tuning, and the KV cache for long contexts."

**Transition**: "Speaking of KV cache — let's understand why it grows with context..."

---

## Slide 16: KV Cache — Why Inference Memory Grows

**Time**: ~1 min
**Talking points**:

- Analogy: "Imagine you're writing an essay and you need to remember every word you've already written. The longer the essay, the more memory you need. That's the KV cache."
- Example: Llama 3 70B at FP16 with 128K context needs ~42 GB just for KV cache.
- If experts look bored: "Paged attention (vLLM) treats KV cache like virtual memory — pages in/out based on demand. Flash Attention reduces memory by computing attention in hardware-efficient tiles."
- If beginners look confused: "The practical implication: longer conversations cost more memory. That's why there's a context window limit."

**Transition**: "Now let's look at compute cost per token..."

---

## Slide 17: Inference Compute: FLOPs per Token

**Time**: ~1 min
**Talking points**:

- Rule of thumb: ~2N FLOPs per token for an N-parameter model.
- This is why inference at scale is expensive: a 70B model uses ~140B FLOPs per token generated.
- If experts look bored: "This approximation holds for the prefill phase. The decode phase is memory-bandwidth-bound, not compute-bound — which is why GPU memory bandwidth matters more than FLOP count for autoregressive generation."

**Transition**: "Now let's look at how these models are trained..."

---

## Slide 18: A2 — Pre-Training: Next Token Prediction

**Time**: ~2 min
**Talking points**:

- Emphasise the simplicity: "The entire training objective is: predict the next word. That's it. Everything else — understanding, reasoning, coding — emerges from doing this at massive scale."
- If beginners look confused: "It's like the world's most intense autocomplete training. The model reads trillions of sentences and learns to predict what comes next."
- If experts look bored: "The cross-entropy loss is equivalent to maximum likelihood estimation over the token distribution. The emergent capabilities arise because accurate next-token prediction requires building a world model."

**Transition**: "But the quality of training depends heavily on the data..."

---

## Slide 19: Data Quality and Processing

**Time**: ~1 min
**Talking points**:

- Key message: "Garbage in, garbage out applies even more at LLM scale. One of the biggest investments in training is not compute — it's data curation."
- Llama 3's data pipeline: quality classifier trained on human-rated web pages, aggressive deduplication removing 90%+ of Common Crawl.
- If experts look bored: "The training data recipe is arguably more important than the architecture. Two identical architectures trained on different data mixtures will have very different capabilities."
- If beginners look confused: "Think of it as: the model is only as good as the text it reads. Feeding it junk text produces a model that writes junk."

**Transition**: "Before training, text has to be converted to numbers through tokenization..."

---

## Slide 20: Tokenization: BPE

**Time**: ~1 min
**Talking points**:

- BPE: start with characters, iteratively merge the most common pairs until you reach target vocabulary size.
- Show a live demo if possible: use tiktoken or the Hugging Face tokeniser on a sentence and count tokens.
- Key point: "Larger vocabularies handle more languages and code patterns efficiently, but increase embedding table size."
- If experts look bored: "The tokeniser vocabulary is fixed at training time and cannot be easily expanded post-training without degrading quality — this is one reason multilingual models require careful vocabulary engineering."

**Transition**: "After pre-training, models undergo post-training to become useful assistants..."

---

## Slide 21: Post-Training: SFT and Alignment

**Time**: ~2 min
**Talking points**:

- Analogy: "Pre-training teaches the model English. SFT teaches it to be a helpful assistant. RLHF teaches it to be a safe and responsible assistant."
- Three stages: pre-training (vast web text), SFT (instruction-following examples), alignment (human feedback).
- Note: "We'll dive deep into alignment methods in M10 — today we just need to know why post-training exists."
- If experts look bored: "DPO has largely replaced PPO-based RLHF for alignment because it's more stable and doesn't require a separate reward model. We'll cover the derivation in M10."

**Transition**: "How much compute and data do you need? The Chinchilla paper answered this..."

---

## Slide 22: Scaling Laws

**Time**: ~2 min
**Talking points**:

- Chinchilla insight: the industry was building models too large and training on too little data. Optimal: smaller models, more data.
- Key insight: "If you care about deployment cost, over-train even further. A small model trained on way more data punches above its weight at inference time."
- If experts look bored: "Chinchilla's compute-optimal frontier: N ≈ 20D (parameters ≈ 20x tokens). Llama 3 deliberately violated this to produce a cheaper-to-serve model — tokens/params ≈ 300x. This is the inference-optimal vs training-optimal trade-off."

**Transition**: "A newer approach lets you scale at inference time instead of training time..."

---

## Slide 23: Test-Time Compute Scaling

**Time**: ~2 min
**Talking points**:

- One of the biggest shifts in 2024-2025: "Traditional scaling — make the model bigger before deployment. Test-time scaling — let the model generate more reasoning tokens at inference. Variable cost per query."
- For experts: "GRPO (Group Relative Policy Optimization) is how DeepSeek trained R1 to reason, cutting RL costs by ~50% compared to PPO. No separate reward model needed — compare outputs within a batch."
- If beginners look confused: "Think of it as giving the model time to think. Instead of answering immediately, it writes out its work first — and that dramatically improves accuracy on hard problems."

**Transition**: "Let's look at the history of how we got here..."

---

## Slide 24: A3 — The LLM Timeline

**Time**: ~1 min
**Talking points**:

- Walk through the timeline. "GPT-3 showed scale works. ChatGPT made it accessible. GPT-4 proved multimodal matters. Claude pioneered long context (200K tokens) and safety-first design."
- For beginners: "Don't memorise the versions. Understand the trend: models get smarter, cheaper, and more capable every 6 months."
- If experts look bored: "The interesting story is not the capability progression but the business model evolution — from API-only to multi-modal to agent-native interfaces."

**Transition**: "The story of 2025 is about open-source catching up..."

---

## Slide 25: The Open-Source Revolution

**Time**: ~1 min
**Talking points**:

- Key message: "Open-source models caught up with closed models in 2025. DeepSeek R1 matches o1 on reasoning. Qwen leads multiple benchmarks. You can now run frontier-quality models on your own infrastructure."
- The structural shift: five independent model families reached frontier simultaneously — this is not a fluke.
- If experts look bored: "The democratisation of frontier AI has direct implications for data privacy, on-premises deployment, and fine-tuning economics. The per-token pricing barrier to frontier capability has effectively been removed."

**Transition**: "Google's contribution was making million-token context practical..."

---

## Slide 26: Google Gemini Evolution

**Time**: ~1 min
**Talking points**:

- Highlight: "Google's key contribution was making million-token context practical. When every other model maxed at 128K tokens, Gemini handled 1M — that changed what's possible for document analysis."
- For experts: "Gemini's multi-modal token efficiency suggests a sophisticated routing mechanism for different modalities. Its long-context performance comes from architectural innovations not yet fully documented publicly."

**Transition**: "The most important development of 2025 came from a different direction..."

---

## Slide 27: DeepSeek: The Efficiency Breakthrough

**Time**: ~1 min
**Talking points**:

- This is one of the most important developments of 2025: "A Chinese lab trained a model matching GPT-4 for under $6 million. Previously, frontier models cost $100M+. They published everything."
- Key technical innovations: GRPO (no separate reward model), MLA (93% KV cache compression), MoE with 256 fine-grained experts.
- If experts look bored: "DeepSeek V3's training paper is required reading. They proved that hardware efficiency (FP8 training, custom communication kernels) matters more than raw GPU count."

**Transition**: "This raises a practical question: do you use open or closed models?"

---

## Slide 28: Open vs Closed: The License Landscape

**Time**: ~1 min
**Talking points**:

- Frame as a practical decision: "If you have patient data or financial records that can't leave your servers, you need an open model you can self-host. If you need cutting-edge capability and latency doesn't matter, a closed API might be faster to prototype."
- Note: "Llama Community License has restrictions for large-scale commercial use — read the fine print."
- If experts look bored: "BSL 1.1 vs Apache 2.0 matters for derivative work and commercial use — know the difference before committing to a stack."

**Transition**: "What can these models actually do, and where do they still fail?"

---

## Slide 29: Frontier Capabilities and Limitations

**Time**: ~1 min
**Talking points**:

- Be honest: "These models are incredibly capable, but hallucination is not solved — it's managed. That's why we spend half this module on RAG and tools: they provide the guardrails that keep LLMs honest."
- Four limitations: hallucination, training cutoff, context limits, reasoning errors on novel problems.
- If experts look bored: "The gap between benchmark performance and real-world reliability is the central unsolved problem in LLM deployment. GPQA Diamond scores of 90%+ coexist with confident errors on simple arithmetic."

**Transition**: "Let me now go deeper on some of the advanced architectural innovations..."

---

## Slide 30: SwiGLU Activation Function

**Time**: ~1 min
**Talking points**:

- Intuition: "SwiGLU is like having two filters instead of one. The first processes information. The second decides how much of that processed information to let through."
- For beginners: "The key takeaway is that modern models use smarter activation functions that learn better representations."
- If experts look bored: "SwiGLU = Swish gating over GLU. The gating mechanism creates a soft selection over intermediate features, which empirically improves performance over GELU at the cost of 50% more FFN parameters for the same hidden size."

**Transition**: "Alongside SwiGLU, modern LLMs standardised on RMSNorm..."

---

## Slide 31: RMSNorm vs LayerNorm

**Time**: ~1 min
**Talking points**:

- "RMSNorm is simpler and faster than LayerNorm. It skips the mean-centering step, which turns out not to matter much for training stability. Every modern LLM uses RMSNorm."
- For beginners: "Normalisation keeps the numbers in the model from getting too large or too small. RMSNorm does this efficiently."
- If experts look bored: "Pre-norm (normalise before attention/FFN) vs post-norm (normalise after) matters as much as which norm you use. Pre-norm is now standard — it provides better training stability at large scale."

**Transition**: "MoE is the innovation that lets models have huge capacity without huge inference cost..."

---

## Slide 32: Mixture-of-Experts (MoE) Architecture

**Time**: ~1 min
**Talking points**:

- Analogy: "MoE is like a hospital: you have 100 specialist doctors, but each patient only sees 2-3 of them. The router decides which specialists are needed."
- Examples: DeepSeek V3 (671B total / 37B active), Qwen 3.5, Mistral Large 3, Llama 4 Maverick.
- For beginners: "The key benefit — a 671B model activates only 37B parameters per token. You get a huge model's knowledge at a medium model's compute cost."
- If experts look bored: "The sparsity pattern in MoE creates interesting training dynamics — experts specialise organically. Auxiliary load-balancing losses are needed to prevent expert collapse."

**Transition**: "The hardest part of MoE is the router design..."

---

## Slide 33: MoE: Router Design and Load Balancing

**Time**: ~1 min
**Talking points**:

- "If the router collapses — sends everything to the same expert — you've wasted all those parameters. DeepSeek's fine-grained approach with 256 small experts solves this: each expert naturally specialises."
- If experts look bored: "Top-K routing with auxiliary loss is the standard approach. DeepSeek uses Top-2 routing with a balancing constraint. The auxiliary loss penalises imbalanced expert utilisation during training."

**Transition**: "For deployment, quantisation brings these large models to accessible hardware..."

---

## Slide 34: Quantization for Deployment

**Time**: ~1 min
**Talking points**:

- Make it practical: "If you have a laptop with 8GB VRAM, you can run a 7B model at INT4. If you have a desktop with 24GB, you can run a 13B at INT4 or 7B at FP16."
- GGUF is the standard format for local deployment via llama.cpp and Ollama.
- If experts look bored: "GPTQ and AWQ are weight-only quantisation methods that apply calibration data to minimise quantisation error. They preserve quality better than naive rounding, especially at 4-bit."

**Transition**: "Speculative decoding is another inference trick that reduces latency..."

---

## Slide 35: Speculative Decoding

**Time**: ~1 min
**Talking points**:

- "The magic: verification is parallelisable but generation is not. A cheap draft model generates tokens sequentially, then the target model verifies them all in one forward pass. Accepted tokens are free speedup."
- If experts look bored: "Acceptance rates of 60-80% are typical, giving 2-3x speedup on tasks where the draft model is reasonably calibrated. The speedup is highest for repetitive or formulaic text."

**Transition**: "Flash Attention solves the memory bottleneck for long contexts..."

---

## Slide 36: Flash Attention

**Time**: ~1 min
**Talking points**:

- Analogy: "Standard attention is like reading an entire library to find one fact. Flash Attention brings relevant books to your desk in batches, reads them quickly, and puts them back. Same result, dramatically less memory."
- If experts look bored: "Flash Attention uses tiling to keep the attention computation in GPU SRAM, avoiding HBM reads/writes for the attention matrix. Flash Attention 3 adds parallelism over sequence length, enabling efficient multi-GPU long-context inference."

**Transition**: "These optimisations enabled the context window revolution..."

---

## Slide 37: Context Window Evolution

**Time**: ~1 min
**Talking points**:

- "In 2020, you could fit 3 pages into an LLM. In 2025, you can fit an entire codebase or a full book. This changes what's possible — you can analyse entire legal contracts or research paper collections in a single prompt."
- If experts look bored: "The remaining challenge is not fitting text in the context — it's the 'lost in the middle' phenomenon. Models attend more strongly to the beginning and end of context. RAG + reranking often outperforms stuffing documents into a long context."

**Transition**: "Reasoning models represent a qualitative shift in how LLMs are used..."

---

## Slide 38: Reasoning Models: The New Paradigm

**Time**: ~2 min
**Talking points**:

- "o1, o3, and R1 were trained with reinforcement learning to reason step-by-step. They generate thousands of tokens of internal reasoning before giving you an answer. For hard problems, this dramatically improves accuracy."
- If experts look bored: "The distinction between fast thinking (direct generation) and slow thinking (extended reasoning chains) maps directly onto the test-time compute scaling literature. The RL training signal teaches the model when extended reasoning is worth the cost."

**Transition**: "DeepSeek R1's training method — GRPO — is worth understanding in detail..."

---

## Slide 39: GRPO: How DeepSeek Trained R1

**Time**: ~2 min
**Talking points**:

- "GRPO is elegant: instead of learning a complex reward function, you compare outputs within a batch. If one solution is better than the group average, reinforce it. If worse, penalise it."
- If experts look bored: "GRPO eliminates the critic network that PPO requires. The reference policy provides a KL constraint to prevent mode collapse. The format reward (correct structure) is separate from the accuracy reward — this cleanly separates reasoning quality from output formatting."

**Transition**: "Mistral has been the most consistent open-source contributor after Meta..."

---

## Slide 40: Mistral: Open Multilingual Multimodal

**Time**: ~1 min
**Talking points**:

- "Mistral's strategy: make specialised models that excel at specific tasks, rather than one model that does everything adequately."
- Mistral is a French company — particularly relevant for European regulatory contexts and multilingual APAC use cases.
- If experts look bored: "Mistral's codestral family is state-of-the-art for code generation. Their multimodal Pixtral uses a vision encoder connected directly to a Mistral decoder — a simpler architecture than contrastive CLIP-based approaches."

**Transition**: "The multimodal dimension is increasingly important for production agents..."

---

## Slide 41: Multimodal LLMs

**Time**: ~1 min
**Talking points**:

- "Multimodal means the model can see, hear, and read — not just read. For data science, this means analysing charts and dashboards directly without building a separate OCR pipeline."
- If experts look bored: "The architecture choice matters: contrastive pre-training (CLIP-style) vs native multi-token embeddings (Gemini style). Native multi-token tends to give better document understanding; contrastive gives better zero-shot image classification."

**Transition**: "Now let's see how all this theory maps to Kailash code..."

---

## Slide 42: Kailash Bridge: LLM Fundamentals

**Time**: ~2 min
**Talking points**:

- Bridge slide: "You now understand what's inside an LLM. Kailash abstracts all of that — you don't need to manage attention heads or KV caches. You choose the model from environment variables, set the budget, and Kaizen handles the rest."
- Show the `Delegate` and `Signature` pattern. Model names always come from environment variables — never hardcoded.
- If experts look bored: "Notice that `Delegate` decouples model selection from task definition. You can swap models without changing any business logic — critical for the cost optimisation patterns we cover later."

**Transition**: "A few deeper technical slides on training before we move to prompting..."

---

## Slide 43: Pre-training Data Composition

**Time**: ~1 min
**Talking points**:

- "The training data recipe is arguably more important than the architecture. Two identical architectures trained on different data mixtures will have very different capabilities."
- Data curation teams at frontier labs are as important as architecture teams.
- If experts look bored: "Dolma, RedPajama, and DCLM are the open pre-training datasets that document their composition. Studying these reveals why open models have specific capability gaps — especially in low-resource languages."

**Transition**: "Training these models requires enormous distributed infrastructure..."

---

## Slide 44: Distributed Training at Scale

**Time**: ~1 min
**Talking points**:

- "DeepSeek V3's training paper showed you don't need 16K GPUs — you need better algorithms."
- For beginners: "You won't train these models yourself, but understanding the scale helps you appreciate why model selection matters — and why DeepSeek's $6M training cost was revolutionary."
- If experts look bored: "Pipeline, tensor, and data parallelism serve different bottlenecks. ZeRO stages 1-3 reduce the memory for optimiser state, gradients, and parameters respectively. DeepSeek's FP8 training required custom numerical stability techniques."

**Transition**: "One of the most debated phenomena in LLM research is emergent abilities..."

---

## Slide 45: Emergent Abilities and Phase Transitions

**Time**: ~1 min
**Talking points**:

- "Some abilities seem to turn on suddenly at certain scales. Whether this is a true phenomenon or a measurement artefact is still being debated."
- Practical advice: "Test on your task. Don't assume a smaller model will work just because a similar task worked."
- If experts look bored: "Schaeffer et al. (2023) argued that emergence is an artefact of non-linear metrics. With continuous metrics like cross-entropy, the transitions appear smoother. The debate is unresolved."

**Transition**: "If you deploy open-source models, inference optimisation is critical..."

---

## Slide 46: Inference Optimization: vLLM and Beyond

**Time**: ~1 min
**Talking points**:

- "vLLM can serve 10x more concurrent users than a naive implementation. For production, always use an optimised inference engine — never raw PyTorch."
- Technologies: vLLM (paged attention), llama.cpp (CPU + quantisation), TensorRT-LLM (NVIDIA), ExLlamaV2 (community optimised).
- If experts look bored: "PagedAttention treats KV cache like OS virtual memory. Continuous batching fills the GPU between requests rather than waiting for batch completion. These two innovations together are why vLLM achieves such high throughput."

**Transition**: "The choice between API and self-hosted is a practical decision every team faces..."

---

## Slide 47: Model Licensing and Deployment Decisions

**Time**: ~1 min
**Talking points**:

- "Start with APIs for prototyping and low volume. Move to self-hosted when: you have sensitive data, you need fine-tuning, or your volume makes per-token pricing expensive."
- Breakeven: roughly 100K+ queries/day, self-hosting often becomes cheaper.
- If experts look bored: "The hidden cost of self-hosting is engineering time and model maintenance. GPU instances cost money whether you use them or not. APIs scale to zero. Model your total cost of ownership carefully."

**Transition**: "Here's the complete landscape as of early 2026..."

---

## Slide 48: The Complete LLM Landscape (Q1 2026)

**Time**: ~1 min
**Talking points**:

- "Five independent open model families at frontier quality. Reasoning models as a distinct category. Embedding models going multimodal."
- Key message: "Focus on the patterns, not the specific model names. The landscape will look different in 6 months."
- If experts look bored: "The interesting signal is convergence on the Llama recipe: RoPE + GQA + SwiGLU + RMSNorm + MoE at scale. The architecture competition has largely settled."

**Transition**: "Now let's look at how to call these models in practice..."

---

## Slide 49: LLM API Patterns

**Time**: ~1 min
**Talking points**:

- "This is the pattern you will use thousands of times. Create a message, specify the model from environment variables, set parameters, optionally provide tools. Every LLM API works this way."
- Show a live API call if possible. The differences between providers are in the details.
- If experts look bored: "The OpenAI-compatible API format has become the de facto standard. vLLM, Ollama, and most self-hosted inference engines implement it. You can swap providers by changing a base URL and an API key."

**Transition**: "Understanding token economics is critical for production cost control..."

---

## Slide 50: Token Economics

**Time**: ~1 min
**Talking points**:

- Make it concrete: "Work out the economics for your use case. How many queries per day? How many tokens per query? This tells you whether APIs or self-hosting makes financial sense."
- Breakeven: roughly 100K+ queries/day, self-hosting becomes cheaper than API pricing.
- If experts look bored: "Cached input tokens are priced differently from new input tokens on most APIs. Cache-aware prompt design — keeping the system prompt stable — can reduce costs by 50%+ for conversation-heavy workloads."

**Transition**: "Qwen deserves special mention for this audience..."

---

## Slide 51: Qwen: The Multilingual Leader

**Time**: ~1 min
**Talking points**:

- "Qwen is particularly relevant for Singapore and APAC applications. Its multilingual capabilities, especially for Chinese and other Asian languages, are best-in-class."
- If you're building for a multilingual audience, Qwen should be in your evaluation set.
- If experts look bored: "Qwen's multilingual capability comes from its data mixture: a higher proportion of Chinese, Japanese, Korean, and other Asian language text than any Western lab uses. The vocabulary is also larger to accommodate CJK characters efficiently."

**Transition**: "Two more architectural details worth knowing..."

---

## Slide 52: Parallel Attention + FFN

**Time**: ~1 min
**Talking points**:

- "Parallel attention + FFN: instead of waiting for attention to finish before starting FFN, run both simultaneously and add the results. Slightly worse quality in theory, but the speed benefit is worth it at scale."
- If experts look bored: "This residual stream perspective — attention and FFN as parallel writers to a shared residual — is the interpretability-friendly way to think about transformer computation."

**Transition**: "ALiBi is the alternative to RoPE for position encoding..."

---

## Slide 53: Positional Encoding: ALiBi

**Time**: ~1 min
**Talking points**:

- "RoPE won the popularity contest — used by Llama, Qwen, Mistral, DeepSeek. ALiBi is simpler and extrapolates better, but less common. Both solve the same problem: encoding position without absolute position embeddings."
- If experts look bored: "ALiBi adds a linear bias to attention scores based on distance. It extrapolates more smoothly than RoPE at lengths beyond training. The tradeoff: slightly lower quality on standard benchmarks but better out-of-distribution length generalisation."

**Transition**: "Before we move to prompting, let's settle the most important practical question..."

---

## Slide 54: Fine-Tuning vs RAG vs Prompting

**Time**: ~2 min
**Talking points**:

- "This is the most important practical decision in the module. Most people jump to fine-tuning when prompting + RAG would solve their problem."
- Rule of thumb: Start with prompting. Add RAG for knowledge. Only fine-tune for style or behaviour that doesn't depend on external knowledge.
- If experts look bored: "Fine-tuning and RAG are not mutually exclusive. Fine-tuning teaches the model HOW to behave with your data. RAG gives it access to WHAT data. Together they outperform either alone."

**Transition**: "Now let's master how to talk to these models. Section B — prompt engineering."

---

## Slide 55: B1 — Anatomy of an LLM Call

**Time**: ~2 min
**Talking points**:

- Start simple: "Think of the system prompt as a job description, the user prompt as a task, and the assistant response as the employee's work."
- Three message types: system (persistent instructions), user (current request), assistant (model response so far).
- Show a live API call if possible.
- If beginners look confused: "The system prompt is where you tell the model what role to play and what rules to follow. It applies to the entire conversation."

**Transition**: "Two parameters control how the model generates text..."

---

## Slide 56: Temperature and Sampling

**Time**: ~1 min
**Talking points**:

- Analogy: "Temperature is like a dial between 'focused accountant' (T=0) and 'creative brainstormer' (T=1). For code generation, use low temperature. For creative writing, use high temperature."
- For experts: "Top-p and top-k can be combined. Top-p is more robust than top-k because it adapts to the distribution shape. The sampling order matters: temperature, then top-k, then top-p, then repetition penalty."

**Transition**: "For agent systems, structured output is essential..."

---

## Slide 57: Structured Output

**Time**: ~1 min
**Talking points**:

- Key point: "Function calling is how agents use tools. The model generates a structured instruction to call a specific function with specific arguments — not free text."
- If experts look bored: "JSON mode and constrained sampling (using grammar-based decoding like GBNF) guarantee valid structured output without post-processing. Kaizen's Signature enforces this at the framework level."

**Transition**: "Kailash makes prompt templates type-safe..."

---

## Slide 58: Prompt Templates

**Time**: ~1 min
**Talking points**:

- Emphasise: "Instead of manually writing prompts as strings, Kailash lets you define them as Python classes with typed fields. This prevents the entire class of bugs where you forget a variable or pass the wrong type."
- `Signature` is the core Kailash primitive: typed inputs, typed outputs, automatic prompt construction.
- If experts look bored: "Signatures are typed contracts between the agent and the LLM. They enable static analysis, automatic documentation, and schema-validated responses — none of which is possible with raw string prompts."

**Transition**: "Now let's cover the three core prompting strategies..."

---

## Slide 59: B2 — Zero-Shot Prompting

**Time**: ~2 min
**Talking points**:

- "Zero-shot means no examples. You just describe the task and trust the model. For frontier models, this works surprisingly well on standard tasks."
- Practical advice: be specific and explicit. "Analyse this data" is bad. "Analyse this CSV, find the 3 strongest predictors of price, output as a markdown table" is good.
- If beginners look confused: "Zero-shot is asking someone who has read everything to do a task they have never specifically been asked to do before. They will try — but they might interpret the request differently than you meant."

**Transition**: "You can dramatically improve reliability by showing examples..."

---

## Slide 60: Few-Shot Prompting (In-Context Learning)

**Time**: ~2 min
**Talking points**:

- "Choose examples that cover the edge cases. Include one tricky example where the answer isn't obvious. This teaches the model your specific standards."
- For experts: "Few-shot learning is a form of in-context learning — the model's weights don't change, but it conditions on the examples in the prompt. The model is essentially doing Bayesian inference over the task distribution."
- Practical tip: for sentiment analysis, include one clearly positive, one clearly negative, and one ambiguous example.

**Transition**: "Chain-of-thought is the most powerful single technique in this section..."

---

## Slide 61: Chain-of-Thought (CoT) Prompting

**Time**: ~2 min
**Talking points**:

- Live demo opportunity: show a maths problem where the model fails without CoT but succeeds with "Let's think step by step."
- Key point: "CoT doesn't teach the model anything new — it gives it more time to think by generating intermediate tokens."
- If beginners look confused: "Show your work, like in school. The model is more accurate when it writes out each step."
- If experts look bored: "Zero-shot CoT consistently outperforms few-shot on novel reasoning tasks because it elicits the model's own reasoning rather than anchoring to potentially misleading examples."

**Transition**: "Why does writing out reasoning help so much? The theory is illuminating..."

---

## Slide 62: CoT as Implicit Computation

**Time**: ~2 min
**Talking points**:

- "This is the theoretical basis for test-time compute scaling. Each reasoning token adds ~2N FLOPs of compute. A model generating 1,000 reasoning tokens effectively does 1,000x more computation than one that answers directly."
- Connection to DeepSeek R1: "R1 was trained to generate these reasoning chains through RL. The emergent reasoning chains look remarkably like deliberate problem-solving strategies."
- If beginners look confused: "More thinking tokens = more compute = more accurate. The model is literally thinking longer."

**Transition**: "Tree-of-Thought takes this further by exploring multiple paths..."

---

## Slide 63: Tree-of-Thought and Self-Consistency

**Time**: ~1 min
**Talking points**:

- Analogy for ToT: "Instead of one person thinking linearly, imagine a team brainstorming multiple approaches simultaneously, then picking the best."
- Self-consistency: "Ask the model the same question 5 times. If 4 out of 5 agree, that's probably right."
- If experts look bored: "ToT requires a value function to evaluate intermediate states — typically another LLM call. The compute cost grows as branching factor times depth. Use it only for high-value decisions."

**Transition**: "ReAct is the bridge between prompting and agents..."

---

## Slide 64: ReAct: Reasoning + Acting

**Time**: ~1 min
**Talking points**:

- "ReAct is the most important prompting pattern because it turns an LLM from a text generator into an agent that can take actions. The key: let the model think out loud, then act, then observe the result, then think again."
- Walk through one cycle: Thought → Action → Observation → Thought → Action...
- Ask: "What would happen if the first search returned no results?" Expected: "The model tries a different search query."
- If experts look bored: "ReAct is the simplest correct implementation of the sense-plan-act loop from classical robotics, applied to language models. The thinking step prevents compounding action errors."

**Transition**: "Reflection teaches the model from its own failures..."

---

## Slide 65: Reflection and Meta-Prompting

**Time**: ~1 min
**Talking points**:

- "Reflexion is like a student who gets feedback on a test, writes down what they got wrong and why, and uses those notes on the next attempt."
- For experts: "Reflexion uses verbal reinforcement — the model generates a reflection on its failure, stores it in episodic memory, and uses it on the next attempt. This achieves measurable improvement on HumanEval coding benchmarks without any gradient updates."

**Transition**: "Section B3 — the practical guidelines for production prompts..."

---

## Slide 66: B3 — Prompt Engineering Best Practices

**Time**: ~1 min
**Talking points**:

- The biggest mistake: writing vague prompts. Walk through the before/after: "Help me analyse this data" vs "Analyse this CSV of Singapore HDB resale prices. Find the 3 strongest predictors of price. Output as a markdown table."
- Four principles: specificity, role definition, output format, examples for edge cases.
- If beginners look confused: "Think of your prompt as a job spec for a contractor. Vague job specs produce unpredictable work."

**Transition**: "The most important security risk in the entire module..."

---

## Slide 67: Prompt Injection: The #1 Security Risk

**Time**: ~1 min
**Talking points**:

- Critical: "Every agent you build must defend against prompt injection. There is no perfect defence — but there are layers."
- Four defence categories: prompting-based (instruction hierarchy), alignment-based (trained refusal), filtering-based (classifiers), system-level (tool permissions).
- If experts look bored: "The indirect injection vector is the most dangerous for RAG systems — malicious instructions hidden in documents that the agent retrieves. This is a live attack vector documented in 2025 incident reports."

**Transition**: "Defence in depth means using all these layers together..."

---

## Slide 68: Defending Against Injection

**Time**: ~1 min
**Talking points**:

- "No single technique is sufficient. You need all of these layers working together."
- Emphasise: "Kailash's Signature contracts are your first line of defence — the model can't return arbitrary data if the output type is enforced."
- If experts look bored: "A strict output schema combined with input sanitisation at the tool level is currently the most practical defence in production. Constitutional AI-style trained refusal helps but cannot be relied upon as the only layer."

**Transition**: "Agent prompts are system engineering, not creative writing..."

---

## Slide 69: Prompt Design for Agents

**Time**: ~1 min
**Talking points**:

- "Agent prompts define tools, rules, and output formats. A well-structured agent prompt prevents most agent failures before they happen."
- Three components: tool definitions (precise, with examples), constraints (what the agent must not do), output contract (format and required fields).
- If beginners look confused: "Think of it as writing the manual for a new employee. The more specific the manual, the fewer mistakes."

**Transition**: "Example selection is a skill in itself..."

---

## Slide 70: Few-Shot Example Selection

**Time**: ~2 min
**Talking points**:

- "Don't grab random examples. For a sentiment classifier: one clearly positive, one clearly negative, and one ambiguous. The ambiguous one teaches the model your specific threshold."
- For experts: "Retrieval-augmented few-shot selection — finding examples similar to the current query — consistently outperforms random selection. This can be implemented using the same vector search infrastructure as your RAG pipeline."

**Transition**: "Structured output deserves its own treatment..."

---

## Slide 71: Output Formatting Techniques

**Time**: ~1 min
**Talking points**:

- "Structured output is not just convenience — it enables automation. If the model returns JSON, your code can parse it reliably. If it returns free text, you're back to regex and prayer."
- Three approaches: JSON mode, function calling with schema, Signature-enforced output.
- If experts look bored: "Grammar-constrained decoding guarantees valid structured output at the token level — the model physically cannot generate invalid JSON."

**Transition**: "For complex tasks, chaining multiple prompts outperforms one mega-prompt..."

---

## Slide 72: Prompt Chaining: Complex Workflows

**Time**: ~1 min
**Talking points**:

- "A single mega-prompt fails unpredictably. Chaining lets you validate each step before moving to the next. If the extraction step misses a fact, you catch it before the analysis step builds on wrong data."
- If experts look bored: "Chaining is the same principle as pipeline design in ML — each step has a clear interface and can be independently tested and improved. SequentialPattern in Kaizen formalises this."

**Transition**: "Now let's see how Kailash systematises everything we've covered..."

---

## Slide 73: Kailash Bridge: Prompt Engineering

**Time**: ~1 min
**Talking points**:

- "Prompt engineering as a discipline is being absorbed into frameworks. Instead of manually crafting prompts, you define Signatures. Instead of manually chaining prompts, you use SequentialPattern."
- Show the Signature pattern: class definition replaces string prompt, typed fields replace placeholders.
- If experts look bored: "Kaizen's Signature is executable — it's not just type hints. It enforces schema-validated responses and handles retry-with-correction automatically."

**Transition**: "Now we tackle one of the most important topics in production LLM systems. Section C — RAG."

---

## Slide 74: C1 — Why RAG?

**Time**: ~2 min
**Talking points**:

- Start with the problem: "Imagine asking your LLM about a regulation that changed last week. Without RAG, it gives you the old answer — confidently. With RAG, it looks up the latest document and gives you the right answer with a citation."
- Three problems RAG solves: training cutoff (stale knowledge), hallucination (unverified claims), domain specificity (general vs domain knowledge).
- If beginners look confused: "RAG gives the model a library card. Instead of relying on what it memorised during training, it can look up the answer."

**Transition**: "Let me walk through the full RAG pipeline..."

---

## Slide 75: The RAG Pipeline

**Time**: ~2 min
**Talking points**:

- Walk through with a concrete example: "User asks: 'What is the maximum LTV ratio for HDB loans?' The system searches the MAS regulations database, finds the relevant paragraph, pastes it into the prompt, and the LLM answers with a citation to MAS Notice 645."
- Two phases: offline indexing (load, chunk, embed, store) and online retrieval (embed query, retrieve, augment, generate).
- If beginners look confused: "Index once, query many times. The expensive embedding step runs offline — query time is just a search."

**Transition**: "Naive RAG works for simple cases but breaks on complex queries..."

---

## Slide 76: Naive RAG vs Advanced RAG

**Time**: ~1 min
**Talking points**:

- "Naive RAG works for simple cases but fails on complex queries, ambiguous questions, or when relevant information is spread across multiple documents."
- Advanced RAG improvements: query rewriting, HyDE, re-ranking, multi-query retrieval, Self-RAG.
- If experts look bored: "The failure modes of naive RAG are well-documented: precision drops with ambiguous queries, recall drops when relevant text is far from the query's surface form, faithfulness drops when retrieved context is only partially relevant."

**Transition**: "There are now seven distinct RAG architectures in use..."

---

## Slide 77: The 7 RAG Architectures (2026)

**Time**: ~1 min
**Talking points**:

- Don't explain all seven. Focus on three: "Vanilla is the starting point. Hybrid is what you should use in production. Agentic RAG is the future — the system itself decides how to search."
- For experts: "GraphRAG builds entity-relationship graphs that enable theme-level queries like 'What are the compliance risks across all vendor contracts?' with full traceability."

**Transition**: "Document loading is where most RAG projects fail silently..."

---

## Slide 78: C2 — Document Loading and Processing

**Time**: ~1 min
**Talking points**:

- "Document loading sounds boring but it's where most RAG projects fail. A PDF with tables will have garbled text if you use a basic parser."
- Invest in good document processing: PyMuPDF for PDFs, Unstructured.io for mixed formats, Docling for complex layouts.
- If experts look bored: "The document loading choice determines your RAG ceiling. You can have perfect chunking, perfect embeddings, and perfect retrieval — but if your text extraction garbled the table data, the answers will be wrong."

**Transition**: "Chunking strategy determines how well retrieval works..."

---

## Slide 79: Chunking Strategies

**Time**: ~1 min
**Talking points**:

- Practical advice: "Start with recursive chunking at 512 tokens with 50-token overlap. This works for 80% of use cases. Only switch to semantic or agentic chunking if your retrieval quality is poor."
- Analogy: "Chunking is like cutting a book into index cards. Too small and each card is meaningless. Too large and you can't find the specific fact you need."
- If experts look bored: "Semantic chunking uses an embedding model to detect topic shifts. Late chunking (JinaAI) chunks after embedding full documents — better for preserving contextual relationships."

**Transition**: "Context matters as much as content for each chunk..."

---

## Slide 80: Metadata and Contextual Chunking

**Time**: ~2 min
**Talking points**:

- "Contextual retrieval is one of the simplest yet most impactful RAG improvements. Use a cheap, fast model to generate a 1-2 sentence context prefix for each chunk."
- Always store: source URL, publication date, document type, section title alongside each chunk.
- If experts look bored: "Anthropic's contextual retrieval paper showed 35-67% reduction in retrieval failures by prepending context. The prefix is generated once at index time — no query-time overhead."

**Transition**: "Now let's understand how the search actually works..."

---

## Slide 81: C3 — Text Embeddings

**Time**: ~2 min
**Talking points**:

- Start visual: "Imagine a map where every sentence is a dot. Similar sentences cluster together. 'Singapore economy' and 'SG GDP' are near each other, but 'chicken rice recipe' is far away. Embeddings create this map in high-dimensional space."
- If beginners look confused: "An embedding converts text into a list of numbers. Similar texts have similar lists of numbers. This lets you find related text using maths."

**Transition**: "Which embedding model should you use?"

---

## Slide 82: Embedding Models (Q1 2026)

**Time**: ~1 min
**Talking points**:

- "The embedding model is one of the most important choices in your RAG system. Use Cohere or OpenAI text-3 for best quality, BGE-M3 for self-hosted, Jina for multilingual."
- Note: "MTEB (Massive Text Embedding Benchmark) is the standard for comparing embedding models."
- If experts look bored: "BGE-M3 supports dense, sparse (SPLADE-style), and multi-vector (ColBERT-style) retrieval from a single model — making it particularly versatile for hybrid search systems."

**Transition**: "Matryoshka embeddings let you trade quality for speed at query time..."

---

## Slide 83: Matryoshka Embeddings

**Time**: ~1 min
**Talking points**:

- "Matryoshka Representation Learning trains the model so that the first K dimensions form a valid embedding for any K. You can use 256 dimensions for coarse search and 1024 for precise re-ranking."
- For experts: "MRL is achieved by adding training losses at multiple dimension checkpoints. The nested structure is guaranteed during training. OpenAI text-3 and Jina v3 both support this with the `dimensions` parameter."

**Transition**: "Embeddings need to be stored and searched efficiently — that's what vector databases do..."

---

## Slide 84: C4 — Vector Databases

**Time**: ~1 min
**Talking points**:

- "For learning and prototyping, use ChromaDB — it runs locally with pip install. For production, use Qdrant or Pinecone."
- The decision: "Self-hosted gives you data control. Managed gives you less ops work. If your data can't leave your servers, self-host."
- If beginners look confused: "A vector database is a regular database, but instead of searching by ID or text, you search by similarity."

**Transition**: "How does the similarity search actually work at scale?"

---

## Slide 85: Vector Index Types

**Time**: ~1 min
**Talking points**:

- Analogy for HNSW: "Imagine a city where you can jump between neighbourhoods via express highways. HNSW builds a multi-level highway system for vectors — at the top level, you jump between distant clusters. At lower levels, you walk to the exact nearest neighbours."
- If experts look bored: "HNSW's construction is greedy — at each layer, nodes connect to their nearest neighbours within the layer. Query time is O(log n) for approximate search. The ef parameter controls the quality/speed tradeoff."

**Transition**: "Dense search alone isn't always enough — hybrid search combines two approaches..."

---

## Slide 86: Hybrid Search

**Time**: ~1 min
**Talking points**:

- "Dense search finds semantically similar documents. Sparse search finds exact keyword matches. Hybrid search combines both. In practice, hybrid search outperforms either alone on most benchmarks."
- Example: "If a user searches for 'MAS 645 LTV ratio', dense search finds documents about loan limits, sparse search finds documents mentioning 'MAS 645' specifically. You need both."
- If experts look bored: "RRF (Reciprocal Rank Fusion) is the standard for combining dense and sparse rankings. It's parameter-free and robust to score scale differences."

**Transition**: "Section C5 — even better retrieval with advanced strategies..."

---

## Slide 87: C5 — Advanced Retrieval Strategies

**Time**: ~2 min
**Talking points**:

- HyDE is counterintuitive: "You ask: 'What causes diabetes?' Instead of searching for that question, the model first generates a hypothetical answer and THAT text is embedded and searched. It works because the hypothetical answer looks more like a document than the question does."
- Other strategies: query decomposition, multi-query retrieval, step-back prompting.
- If experts look bored: "HyDE closes the representation gap between short queries and long document passages. The gap exists because embedding models were often trained on document-to-document similarity, not query-to-document."

**Transition**: "Re-ranking is the quality gate after initial retrieval..."

---

## Slide 88: Re-Ranking with Cross-Encoders

**Time**: ~1 min
**Talking points**:

- "Think of bi-encoders as a librarian who reads the title and summary. Cross-encoders read the full document alongside your question. Cross-encoders are more accurate but too slow for millions of documents."
- Strategy: bi-encoders to shortlist (top 50), cross-encoders to finalise (top 5).
- If experts look bored: "Cross-encoders achieve significantly higher NDCG than bi-encoders because they process query-document pairs jointly. The latency tradeoff is acceptable when applied to a small candidate set."

**Transition**: "ColBERT gets cross-encoder quality at near bi-encoder speed..."

---

## Slide 89: ColBERT: Late Interaction

**Time**: ~1 min
**Talking points**:

- For experts: "ColBERT pre-computes per-token embeddings for all documents. At query time, you only compute per-token embeddings for the query, then take the maximum similarity for each query token — the MaxSim operation."
- If beginners look confused: "This is an advanced optimisation technique. The key takeaway: retrieval quality can be significantly improved without proportional latency increase."

**Transition**: "How do you measure whether your RAG system is actually working?"

---

## Slide 90: C6 — RAG Evaluation: RAGAS Framework

**Time**: ~2 min
**Talking points**:

- Walk through each metric: "If the answer includes a fact not in the retrieved context, faithfulness drops. If the answer is faithful but doesn't address the question, relevancy drops. If you retrieved irrelevant documents, precision drops. If you missed key documents, recall drops."
- Four metrics: faithfulness, answer relevancy, context precision, context recall.
- If experts look bored: "RAGAS uses LLM-as-judge to evaluate most metrics, which introduces evaluator bias. Calibrate against human judgements on a held-out set before trusting RAGAS scores as ground truth."

**Transition**: "Self-RAG and corrective RAG add intelligence to the retrieval decision itself..."

---

## Slide 91: Self-RAG and Corrective RAG

**Time**: ~1 min
**Talking points**:

- "Self-RAG is efficient — retrieves only when needed. CRAG is accurate — verifies what it retrieves. Agentic RAG combines both: an agent that decides when to retrieve, what to retrieve, and whether the retrieved content is good enough."
- For experts: "These systems aren't fixed pipelines anymore — they're autonomous decision-making agents with retrieval as one tool among many."

**Transition**: "GraphRAG handles questions that require synthesising across many documents..."

---

## Slide 92: GraphRAG

**Time**: ~1 min
**Talking points**:

- "GraphRAG is the answer to: 'What if my question requires synthesising information from many documents?' Standard RAG retrieves individual chunks. GraphRAG builds a map of how concepts relate across your entire corpus."
- If experts look bored: "Microsoft's GraphRAG builds entity-relationship graphs using LLM extraction, then uses community detection algorithms to identify themes. Query time uses the graph structure to find relevant entity clusters, not just nearest-neighbour chunks."

**Transition**: "Let me walk through building a complete RAG pipeline step by step..."

---

## Slide 93: Building a RAG Pipeline: Step by Step

**Time**: ~1 min
**Talking points**:

- Walk through each step: "First we load the PDFs. Then we chunk into 512-token pieces with 50-token overlap. Then we embed each chunk into a vector. Then we store in ChromaDB."
- Emphasise: "This is the indexing pipeline — run once, query many times."
- If beginners look confused: "Think of it as building a really smart index for your documents. You do the expensive work once so searches are fast later."

**Transition**: "And here's what happens at query time..."

---

## Slide 94: RAG Pipeline: Query Time

**Time**: ~2 min
**Talking points**:

- "The simplest possible RAG query function: embed the question, find similar chunks, paste them into the prompt, and ask the LLM to answer."
- Key instruction: "Answer based ONLY on the context" — this reduces hallucination significantly.
- If experts look bored: "The grounding instruction is critical but not foolproof. Strong models can still confabulate when the context is partially relevant. RAGAS faithfulness measurement catches this."

**Transition**: "Why does chunk overlap matter so much?"

---

## Slide 95: Chunk Overlap: Why and How Much

**Time**: ~1 min
**Talking points**:

- "Overlap is insurance against bad splits. It ensures information at chunk boundaries appears in both adjacent chunks, so at least one retrieval will contain the complete answer."
- 50-token overlap is the standard starting point for 512-token chunks (~10%).
- If experts look bored: "Overlap increases storage and embedding costs proportionally. At 50-token overlap for 512-token chunks, you add ~10% to your index size. The retrieval quality improvement is typically worth this overhead."

**Transition**: "Multi-query retrieval dramatically improves recall..."

---

## Slide 96: Multi-Query Retrieval in Practice

**Time**: ~1 min
**Talking points**:

- "Multi-query retrieval is cheap insurance. It costs 3-4 extra LLM calls to rephrase, but dramatically improves recall."
- Production tip: "Cache the rephrased queries to avoid repeated LLM calls for the same question."
- If experts look bored: "Multi-query with query decomposition is particularly powerful for multi-hop questions — complex questions that decompose into independent sub-queries about different aspects."

**Transition**: "When RAG fails, here's how to diagnose it..."

---

## Slide 97: RAG Failure Modes

**Time**: ~1 min
**Talking points**:

- Debugging strategy: "When your RAG system gives a wrong answer, the first question is: was the right document retrieved? Check retrieval first. If yes, the problem is generation. If no, the problem is retrieval."
- Always log retrieved chunks alongside the generated answer.
- If experts look bored: "Systematically instrument your RAG pipeline: log query, retrieved chunks with scores, final prompt, and generated answer. RAGAS metrics tell you which component is underperforming."

**Transition**: "Let me clarify the architecture of bi-encoders..."

---

## Slide 98: Sentence Transformers: Bi-Encoder Architecture

**Time**: ~1 min
**Talking points**:

- "A bi-encoder creates an embedding for the query and an embedding for each document independently. Similarity is computed by cosine distance."
- This is fast — document embeddings are computed once during indexing, query embedding at query time.
- If experts look bored: "The bi-encoder's independence assumption enables pre-computation and fast retrieval. It also prevents the model from seeing the query when encoding documents — which is why cross-encoders outperform bi-encoders on precise relevance."

**Transition**: "Which vector database should you choose?"

---

## Slide 99: Choosing a Vector Database

**Time**: ~1 min
**Talking points**:

- "The vector database choice is less important than you think for getting started. Start with ChromaDB. When you need hybrid search, metadata filtering, or scale beyond a million vectors, migrate to Qdrant."
- Decision matrix: ChromaDB (prototype), Qdrant (production self-hosted), Pinecone (managed), pgvector (existing Postgres).

**Transition**: "Let me show you RAGAS evaluation in practice..."

---

## Slide 100: RAGAS Evaluation in Practice

**Time**: ~1 min
**Talking points**:

- "RAGAS gives you four numbers that tell you exactly where your RAG pipeline is failing. Low faithfulness? The model is hallucinating. Low context precision? You're retrieving irrelevant documents."
- Connect to earlier modules: use ExperimentTracker from M3 to compare RAG configurations.
- If experts look bored: "RAGAS requires a test dataset with question, ground truth answer, and pipeline answer. Generating synthetic test datasets using an LLM is now standard practice — use Ragas's TestsetGenerator."

**Transition**: "Here's how Kailash's RAGResearchAgent wraps all of this..."

---

## Slide 101: Kailash Bridge: RAG Systems

**Time**: ~1 min
**Talking points**:

- "RAGResearchAgent wraps the entire RAG pipeline: chunking, embedding, retrieval, re-ranking, and generation. Configure it once and it handles the complexity."
- The Signature ensures the output always includes sources — no citations, no answer.
- If experts look bored: "RAGResearchAgent composes Kailash primitives: document loading, embedding, vector store, re-ranker, and Delegate. You can replace any component independently."

**Transition**: "Production RAG requires separating the offline and online pipelines..."

---

## Slide 102: RAG for Production: End-to-End Architecture

**Time**: ~2 min
**Talking points**:

- "Production RAG has two pipelines. The offline pipeline runs periodically to index new documents. The online pipeline runs on every user query. Keep these separate — don't re-embed documents on every query."
- If experts look bored: "The offline pipeline belongs in a scheduled job or event-driven pipeline. The online pipeline should have p99 latency under 2 seconds to be production-viable."

**Transition**: "Incremental indexing handles document updates efficiently..."

---

## Slide 103: Incremental Indexing

**Time**: ~1 min
**Talking points**:

- "For a corpus of 100K documents, full re-indexing takes hours. Incremental indexing handles only the changes — typically seconds. Track document versions and only re-embed what changed."
- If experts look bored: "Content hashing at the chunk level enables fine-grained incremental updates. Only chunks whose content changed need re-embedding. This reduces embedding API costs dramatically for document collections with frequent minor edits."

**Transition**: "Metadata filtering is one of the most underused RAG techniques..."

---

## Slide 104: RAG with Metadata Filtering

**Time**: ~1 min
**Talking points**:

- "Metadata filtering is underused but critical. If the user asks about 'current MAS regulations', filter to documents from the last 2 years before searching."
- Always store: source URL, publication date, document type, author, section.
- If experts look bored: "Combining metadata pre-filtering with vector search can improve precision dramatically for enterprise corpora where temporal or categorical relevance matters."

**Transition**: "Hybrid search implementation brings together everything we've covered..."

---

## Slide 105: Hybrid RAG: Dense + Sparse Implementation

**Time**: ~1 min
**Talking points**:

- "Hybrid search consistently outperforms either method alone. BM25 catches exact keyword matches that embedding search misses. Embedding search catches semantic matches that BM25 misses. RRF combines them elegantly."
- If experts look bored: "BM25 with SPLADE outperforms plain BM25 by expanding queries with related terms. Qdrant natively supports hybrid search with sparse vector support."

**Transition**: "Tables and images are the biggest challenge in enterprise RAG..."

---

## Slide 106: RAG with Images and Tables

**Time**: ~1 min
**Talking points**:

- "Most RAG failures in enterprise settings come from tables and images, not text. A financial report where 60% of the data is in tables will fail with text-only RAG."
- Solutions: Docling for table extraction, multimodal embeddings for images, structured markdown conversion for tables.
- If experts look bored: "Table-aware chunking splits along row boundaries, not token counts. Storing tables as structured JSON alongside embeddings of the table's textual summary enables both semantic and structured search."

**Transition**: "Agentic RAG is where agents and RAG converge..."

---

## Slide 107: Agentic RAG: The Agent Decides

**Time**: ~1 min
**Talking points**:

- "Agentic RAG: instead of a fixed pipeline, the system reasons about what information it needs, searches strategically, evaluates the results, and adapts."
- For experts: "Kaizen's RAGResearchAgent can be extended with adaptive retrieval strategies using the BaseAgent pattern — making retrieval a decision, not a fixed step."

**Transition**: "Now we move to the heart of today's module. Section D — AI Agents."

---

## Slide 108: D1 — What Is an Agent?

**Time**: ~2 min
**Talking points**:

- "The simplest way to understand agents: a chatbot talks. An agent does."
- Example: "If you ask a chatbot to book a flight, it tells you how. If you ask an agent to book a flight, it searches for flights, compares prices, and books one."
- If beginners look confused: "An agent is a program that uses an LLM to decide what to do next, takes action using tools, and keeps going until it finishes the task."

**Transition**: "The agent loop is the fundamental pattern..."

---

## Slide 109: The Agent Loop

**Time**: ~2 min
**Talking points**:

- "This loop is the fundamental pattern for all agent architectures. ReAct, Plan-and-Execute, Reflexion — they all implement this loop with different strategies for the Think step."
- Draw the loop on a whiteboard if possible: Perceive → Think → Act → Observe → repeat.
- If experts look bored: "The loop termination condition is critical. Agents that don't know when to stop will loop until they hit the context limit or the cost limit. Always define a clear stopping condition."

**Transition**: "Tools are how agents interact with the world..."

---

## Slide 110: Tool Use: Function Calling

**Time**: ~1 min
**Talking points**:

- "The model doesn't execute code. It generates a structured request — 'I want to call this function with these arguments.' Your code executes the function and returns the result to the model."
- This is the foundation of all agent capabilities: web search, database queries, code execution, API calls.
- If experts look bored: "Parallel tool calling allows the model to call multiple tools simultaneously. Kaizen's ReActAgent supports parallel tool execution with configurable concurrency."

**Transition**: "Memory is what separates sophisticated agents from simple chatbots..."

---

## Slide 111: Agent Memory

**Time**: ~1 min
**Talking points**:

- "Memory is what separates a sophisticated agent from a simple chatbot. Without memory, every conversation starts from scratch."
- Four types: in-context (conversation history), working (current task state), episodic (past experiences), semantic (factual knowledge / RAG).
- Analogy: "The difference between calling a random support agent vs calling your dedicated account manager."
- If experts look bored: "MemoryAgent in Kaizen automatically summarises older messages when the context window fills — preserving continuity without manual context management."

**Transition**: "The most important agent architecture pattern is ReAct..."

---

## Slide 112: D2 — ReAct Architecture

**Time**: ~1 min
**Talking points**:

- Walk through each step of the ReAct cycle: Thought → Action → Observation → Thought...
- "The key insight: the model explicitly writes out its reasoning before each action. This makes it debuggable — you can see WHY it made each decision."
- Ask: "What would happen if the first search returned no results?" Expected: "The model would try a different search query."
- If experts look bored: "ReAct's explicit chain-of-thought before each action prevents the compounding errors that plagued earlier tool-using systems. The observation step is critical — without it, the model can't recover from tool failures."

**Transition**: "Plan-and-Execute separates planning from execution for cost efficiency..."

---

## Slide 113: Plan-and-Execute Architecture

**Time**: ~2 min
**Talking points**:

- "Plan-and-Execute is how you make agents affordable. The planner thinks hard once using an expensive frontier model, then the executor follows the plan using a cheap model. Frontier-quality planning at budget-friendly execution cost."
- Cost reduction: up to 70-90% compared to using the expensive model for everything.
- If experts look bored: "The critical design choice: how much replanning to allow when the executor encounters unexpected results. Too little replanning produces brittle plans. Too much replanning negates the cost savings."

**Transition**: "Reflexion teaches agents from failure..."

---

## Slide 114: Reflexion Architecture

**Time**: ~2 min
**Talking points**:

- "Reflexion is like a student who gets feedback on a test, writes down what they got wrong and why, and uses those notes on the next attempt."
- For experts: "Reflexion achieves measurable improvement on HumanEval coding benchmarks by storing failed attempts and their reflections in episodic memory. The verbal reinforcement signal requires no gradient updates."

**Transition**: "LATS uses tree search to explore multiple solution paths..."

---

## Slide 115: LATS: Language Agent Tree Search

**Time**: ~1 min
**Talking points**:

- For experts: "LATS brings the exploration-exploitation trade-off from RL to agent systems. Particularly powerful for tasks where there are many possible action sequences and the cost of failure is high."
- For beginners: "Think of it as the agent considering multiple strategies simultaneously, like a chess player thinking several moves ahead before making a decision."

**Transition**: "Now let's see how Kailash Kaizen implements all of this..."

---

## Slide 116: Kailash Kaizen: Agent Framework

**Time**: ~1 min
**Talking points**:

- "Notice three things: (1) The contract is defined as a Python class, not a string prompt. (2) Tools are passed at construction, not called manually. (3) The output is typed — result.insights is a list of strings, guaranteed."
- Model names always come from environment variables — never hardcode them.
- If experts look bored: "Kaizen's Signature contract is enforced at runtime — if the model returns a response that doesn't match the schema, Kaizen retries with a correction prompt. This eliminates an entire class of agent output parsing errors."

**Transition**: "When should you use Delegate vs BaseAgent?"

---

## Slide 117: Delegate vs BaseAgent

**Time**: ~1 min
**Talking points**:

- "Start with Delegate. It handles the agent loop, tool routing, cost tracking, and error handling. Only drop down to BaseAgent when you need a custom reasoning strategy — like implementing Reflexion or LATS."
- Decision rule: if you're customising the reasoning strategy, use BaseAgent. If you're just adding tools or changing the task, use Delegate.
- If experts look bored: "BaseAgent gives you full control over the observe-think-act loop. You implement the `step()` method. Delegate is a specialised BaseAgent with an opinionated ReAct implementation."

**Transition**: "Kailash ML agents bridge the gap between ML pipelines and LLMs..."

---

## Slide 118: D3 — Kailash ML Agents

**Time**: ~1 min
**Talking points**:

- "These agents don't replace you — they augment you. DataScientist suggests features you might not think of. ModelSelector explains WHY it recommends XGBoost over Random Forest for your specific data."
- The ML-specific agents from M1-M8 are now tools that the agent can call.
- If experts look bored: "The natural language interface to ML pipelines bridges the gap between technical ML output and business stakeholder understanding — this is the key value proposition."

**Transition**: "Human oversight at the right level is critical for production agents..."

---

## Slide 119: Agent Double Opt-In

**Time**: ~2 min
**Talking points**:

- "This is the human-on-the-loop model. The agent does the heavy lifting — analysis, computation, comparison — but the human makes decisions at two key gates."
- Human-on-the-loop vs human-in-the-loop: approve strategy and results, not every individual tool call.
- If experts look bored: "The double opt-in pattern operationalises HITL without the throughput bottleneck. You get human oversight at the decisions that matter — not at every intermediate step."

**Transition**: "Cost control is non-negotiable for production agents..."

---

## Slide 120: LLMCostTracker: Budget Enforcement

**Time**: ~1 min
**Talking points**:

- "One of the biggest risks with agents is cost overrun. An agent in a loop can burn through hundreds of dollars of API calls in minutes. LLMCostTracker makes this impossible — it's a hard limit, not a guideline."
- Real-world cost estimates: a simple Q&A chatbot (fast, cheap model) ~$1/day for 1,000 queries; a research agent (frontier model) ~$375/day.
- If experts look bored: "Budget enforcement should be at the Signature level, not the application level. Each tool call deducts from the budget, and the agent halts gracefully when the budget is exhausted."

**Transition**: "Some tasks require multiple agents working together. Section D4..."

---

## Slide 121: D4 — Why Multi-Agent?

**Time**: ~1 min
**Talking points**:

- "Think of a single agent as a solo employee. A multi-agent system is a team. Some tasks need a team: one person researches, another analyses, another writes the report."
- Three reasons for multi-agent: parallelism (do things simultaneously), specialisation (right model for right task), verification (two agents checking each other's work).
- If beginners look confused: "Multi-agent is just multiple agents where each has a specific role, and one agent coordinates the others."

**Transition**: "There are four distinct orchestration patterns..."

---

## Slide 122: Orchestration Patterns

**Time**: ~1 min
**Talking points**:

- Walk through each pattern with a concrete example.
- Supervisor-Worker: "A project manager agent assigns tasks to a researcher, coder, and tester."
- Sequential: "An ETL pipeline where each agent handles one stage."
- Parallel: "Research three topics simultaneously and combine results."
- Debate: "Two agents argue about whether to approve a loan, a third synthesises the decision."
- If experts look bored: "The debate pattern is computationally expensive but produces significantly higher quality on decisions with high stakes and high ambiguity."

**Transition**: "Let me show you Kailash's implementations of these patterns..."

---

## Slide 123: Kailash Multi-Agent Patterns

**Time**: ~1 min
**Talking points**:

- "The supervisor uses the most capable model for planning and coordination. Workers use cheaper models because they execute well-defined subtasks. This can reduce costs by 70-90%."
- `SupervisorWorkerPattern`, `SequentialPattern`, `ParallelPattern`, `DebatePattern` — all in Kaizen.
- If experts look bored: "Each pattern has a different failure mode. Supervisor-Worker fails when the supervisor plan is wrong and workers execute it faithfully. Build validation into every pattern."

**Transition**: "A2A enables agents from different frameworks to communicate..."

---

## Slide 124: A2A Protocol: Agent Interoperability

**Time**: ~1 min
**Talking points**:

- "A2A is like a phone system for agents. Any agent, regardless of which framework built it, can discover other agents via their Agent Card and send them tasks."
- "A Kailash agent can delegate work to a LangGraph agent and vice versa."
- Key detail: "Google created A2A in April 2025. Over 50 companies adopted it. It's now under the Linux Foundation."
- If experts look bored: "A2A uses JSON-LD for capability advertisement and SSE for streaming task responses. The Agent Card includes supported task types, required authentication, and rate limits."

**Transition**: "How do you know if your agent is actually working well?"

---

## Slide 125: Agent Evaluation

**Time**: ~1 min
**Talking points**:

- "SWE-bench: can an agent autonomously fix bugs in real open-source projects? Frontier agents resolve about 70% of issues. Remarkable but means 30% of real bugs are still beyond them."
- For custom agents: trajectory evaluation (right steps?), outcome evaluation (right result?), cost evaluation (how much did it cost?).

**Transition**: "Safety is the other side of capability..."

---

## Slide 126: Agent Safety

**Time**: ~1 min
**Talking points**:

- "Every agent you build must account for these three risks. Prompt injection is the attacker. Tool misuse is the accident. Cost overruns are the oversight failure."
- Emphasise: "We'll go deeper into governance in M10."
- If experts look bored: "The minimal viable safety stack: Signature output validation, LLMCostTracker budget enforcement, tool permission scoping, input sanitisation before RAG retrieval."

**Transition**: "Your tool definitions are prompts to the model..."

---

## Slide 127: Tool Definition Best Practices

**Time**: ~1 min
**Talking points**:

- "Your tool descriptions are prompts to the model. If the description is vague, the model will use the tool incorrectly."
- Include: specific parameter types, allowed values (enums), what the tool returns, when NOT to use this tool.
- If experts look bored: "Tool description quality has a larger impact on agent performance than model choice for retrieval-augmented tasks. A well-described tool set with a mid-tier model outperforms a poorly-described tool set with a frontier model."

**Transition**: "Production agents need robust error handling..."

---

## Slide 128: Agent Error Handling

**Time**: ~1 min
**Talking points**:

- "The difference between a demo agent and a production agent is error handling. Demo agents crash on the first unexpected response. Production agents recover, retry, and gracefully degrade."
- Three error categories: tool failures (retry with backoff), malformed output (retry with correction prompt), budget exhaustion (graceful shutdown with partial results).
- If experts look bored: "For LLM output parsing failures, include the failed output in the retry prompt: 'Your last response was: X. It did not match the required format Y. Please try again.'"

**Transition**: "Observability is what makes production agents debuggable..."

---

## Slide 129: Agent Tracing and Observability

**Time**: ~1 min
**Talking points**:

- "Always enable tracing in development and production. The trace log is your debugger. When a customer reports a wrong answer, pull the trace and you'll see exactly where the agent went wrong."
- What to log: every tool call (input, output, latency, cost), every reasoning step, every model call.
- If experts look bored: "OpenTelemetry is the standard for agent tracing. Kaizen's built-in tracing exports to any OTel-compatible backend."

**Transition**: "Handoff passes context between specialised agents..."

---

## Slide 130: Handoff Pattern: Agent Context Transfer

**Time**: ~1 min
**Talking points**:

- "Handoff is like transferring a phone call to a specialist. The specialist gets a summary of the conversation so the customer doesn't have to repeat everything."
- `HandoffPattern` in Kaizen manages context transfer automatically: summarise what's been done, pass task state, set up the next agent's context.
- If experts look bored: "Handoff quality is the bottleneck in sequential multi-agent systems. Too much context creates noise. Too little creates errors. Kaizen's handoff summarisation uses a dedicated prompt to extract only task-relevant state."

**Transition**: "The debate pattern uses adversarial verification for high-stakes decisions..."

---

## Slide 131: Debate Pattern: Agent Verification

**Time**: ~2 min
**Talking points**:

- "Debate is expensive but effective. Two agents arguing about a loan approval are less likely to miss a red flag than one agent working alone. The synthesiser agent resolves disagreements by weighing the arguments."
- If experts look bored: "The key design choice: open debate (synthesiser sees both arguments) vs blind debate (synthesiser sees only final positions). Open debate allows weighting of arguments; blind debate prevents anchoring."

**Transition**: "Let's see how this all maps to Kailash code..."

---

## Slide 132: Kailash Bridge: AI Agents

**Time**: ~1 min
**Talking points**:

- "The pattern: understand the theory, then let Kailash handle the implementation complexity. ReAct becomes ReActAgent. Multi-agent becomes SupervisorWorkerPattern. Safety becomes Signature + CostTracker + PACT."
- If experts look bored: "Kaizen's agent abstractions compose: RAGResearchAgent is a BaseAgent that uses Delegate internally. SupervisorWorkerPattern is a coordination layer over multiple Delegates."

**Transition**: "Let me compare the major frameworks to help you make informed choices..."

---

## Slide 133: Agent Frameworks Comparison (Q1 2026)

**Time**: ~1 min
**Talking points**:

- "The differences that matter: how state is managed, how agents coordinate, and how tools are integrated."
- Kailash's advantage: typed contracts and deep ML engine integration.
- If experts look bored: "LangGraph uses a graph-based state machine. AutoGen uses multi-agent conversation with message-passing. CrewAI uses role-based agents. Kaizen uses typed Signatures. The choice depends on whether you need state machines, message passing, role semantics, or type safety."

**Transition**: "Let me show you how to build a ReAct agent from scratch..."

---

## Slide 134: Building a ReAct Agent from Scratch

**Time**: ~1 min
**Talking points**:

- "ReActAgent handles the think-act-observe loop automatically. You define the task Signature and the available tools. The agent decides which tools to use, in what order, and when it has enough information to answer."
- Walk through the code: Signature definition → tool list → ReActAgent construction → execute.
- If experts look bored: "The ReActAgent's internal loop uses a stopping condition based on the Signature's output fields. When all required fields are populated, the agent halts. This prevents unnecessary tool calls."

**Transition**: "Human-on-the-loop is the production model for responsible AI deployment..."

---

## Slide 135: Human-on-the-Loop vs Human-in-the-Loop

**Time**: ~1 min
**Talking points**:

- "Human-in-the-loop was the default in 2023-2024. But if you need human approval for every tool call, your agent is just fancy autocomplete."
- Human-on-the-loop: define boundaries (budget, tools, data access), let the agent operate, review results.
- If experts look bored: "The operating envelope — what the agent is allowed to do — is the key design decision. PACT (Module 10) formalises this with the D/T/R (Decide/Take action/Review) accountability grammar."

**Transition**: "Let's look at how agents manage memory over long sessions..."

---

## Slide 136: Agent Memory Architecture

**Time**: ~1 min
**Talking points**:

- "MemoryAgent is the solution to the 'context window fills up' problem. After 50 messages, it automatically summarises older messages, freeing space for new ones."
- Four memory types: in-context, working memory, episodic memory (Reflexion), semantic memory (RAG).
- If experts look bored: "Memory architecture is the unsolved problem in production agents. Current approaches — sliding window, summarisation, retrieval — all involve information loss. Perfect memory with perfect recall remains a research problem."

**Transition**: "Let me be concrete about the real economics of running agents..."

---

## Slide 137: Agent Cost Analysis: Real Numbers

**Time**: ~1 min
**Talking points**:

- "These are real-world cost estimates. Know your economics before deploying."
- Ask: "Which of these makes business sense for your use case?"
- If experts look bored: "Always frame AI costs against the alternative. If the agent handles 1,000 queries per day that would each take a human 15 minutes, that's 250 human-hours per day. The agent costs $500 vs $25,000+ in human labour."

**Transition**: "Let me catalogue the anti-patterns to avoid..."

---

## Slide 138: Production Agent Anti-Patterns

**Time**: ~1 min
**Talking points**:

- "These are the mistakes we see in every production agent deployment."
- Most common: too many tools (agent gets confused) and no budget limit (agent burns money in loops).
- If experts look bored: "The tool overload anti-pattern is subtle — agents with more than 20 tools show measurable performance degradation. Segment tools by agent role and give each agent only the tools it needs."

**Transition**: "Now let's look at how agents connect to the wider tool ecosystem. Section E — MCP."

---

## Slide 139: E — Model Context Protocol (MCP)

**Time**: ~2 min
**Talking points**:

- "MCP was created by Anthropic in November 2024 and donated to the Linux Foundation in December 2025. By 2026, it is the de facto standard — adopted by OpenAI, Google, Microsoft, and thousands of developers."
- If beginners look confused: "MCP is a universal plug socket for AI tools. Instead of every AI application building its own custom integrations, MCP provides one standard connector that works everywhere."

**Transition**: "Let me show you the architecture..."

---

## Slide 140: MCP Architecture

**Time**: ~1 min
**Talking points**:

- Walk through: "The LLM is the client. It discovers available tools from the MCP server, then calls them via JSON-RPC. The server executes the tool and returns the result."
- Key: "MCP is bidirectional — the server can also push notifications to the client."
- Three primitives: Tools (execute actions), Resources (read data), Prompts (reusable templates).
- If experts look bored: "The protocol specification is minimal: tool discovery via `tools/list`, execution via `tools/call`, resource access via `resources/read`. This simplicity is why adoption was so rapid — implementation takes hours."

**Transition**: "Let me show you how to build an MCP server..."

---

## Slide 141: Building an MCP Server

**Time**: ~1 min
**Talking points**:

- "This MCP server exposes two ML tools: dataset exploration and model training. Any MCP client can discover and use these tools. You write the tool once, and it works everywhere."
- Show the Kailash pattern: `@mcp_server.tool()` decorator, `MCPServer` class, tool description as the prompt to the model.
- If experts look bored: "FastMCP handles the JSON-RPC protocol, session management, and capability negotiation. You write the tool logic; FastMCP handles the protocol."

**Transition**: "Security is as important for MCP servers as for any API..."

---

## Slide 142: MCP Security

**Time**: ~1 min
**Talking points**:

- "Think of MCP security like API security. Authentication: who is calling? Authorisation: are they allowed to use this tool? Rate limiting: how often? Input validation: are the parameters safe?"
- Every tool you expose must answer all four questions.
- If experts look bored: "MCP's security model uses OAuth 2.0 for authentication and capability-based authorisation. Input validation at the tool level is not optional — MCP clients are often LLMs that can be prompt-injected."

**Transition**: "MCP and A2A work together to form the complete interoperability picture..."

---

## Slide 143: MCP + A2A: The Full Picture

**Time**: ~1 min
**Talking points**:

- "These two protocols are complementary. MCP is the arm that lets an agent reach tools. A2A is the voice that lets agents talk to each other. Both are now under Linux Foundation governance."
- If experts look bored: "MCP handles synchronous tool invocation with structured I/O. A2A handles asynchronous task delegation between autonomous agents with streaming responses."

**Transition**: "Resources give agents read access to data without tool execution risk..."

---

## Slide 144: MCP Resources: Data Access

**Time**: ~1 min
**Talking points**:

- "Resources are like database views — the agent can read them but can't modify them. This is safer than giving the agent a tool that runs arbitrary SQL queries."
- If experts look bored: "Resources are URI-addressed and support subscription for live data. An agent can subscribe to a resource and receive updates when the underlying data changes — enabling event-driven agent architectures."

**Transition**: "MCP Prompts standardise AI workflows across all clients..."

---

## Slide 145: MCP Prompts: Reusable Templates

**Time**: ~1 min
**Talking points**:

- "MCP Prompts are like stored procedures for AI — reusable, standardised instructions that any client can invoke. Whether you use any supported MCP client, the analysis follows the same methodology."
- If experts look bored: "Prompts in MCP are parameterised templates that return structured message sequences. They're server-defined, versioned, and shareable across clients."

**Transition**: "The MCP ecosystem has grown to thousands of available servers..."

---

## Slide 146: MCP Ecosystem: Available Servers

**Time**: ~1 min
**Talking points**:

- "Whatever tool you need — database access, web search, email, cloud infrastructure — there's probably an MCP server for it. And if there isn't, building one takes a few hours."
- Highlight categories relevant to the class: file systems, databases, web search, code execution, data analysis.
- If experts look bored: "Evaluate MCP servers the same way you'd evaluate any third-party library — check the source code, look for security issues, understand the data access model."

**Transition**: "Here's how Kailash integrates MCP into the agent stack..."

---

## Slide 147: Kailash Bridge: MCP

**Time**: ~1 min
**Talking points**:

- "Kailash makes MCP integration seamless. If you have a workflow, you can expose it as an MCP server with a decorator. If you use Nexus, the MCP server is generated automatically alongside the API and CLI."
- Model names always from environment variables — never hardcoded in MCP tool definitions.
- If experts look bored: "The `@workflow_as_mcp_tool` pattern converts any Kailash workflow into an MCP tool automatically, handling parameter schema generation and result serialisation."

**Transition**: "Now let's look at production deployment. Section F — Kailash Nexus."

---

## Slide 148: F — Kailash Nexus: Multi-Channel Deployment

**Time**: ~2 min
**Talking points**:

- "Nexus solves the deployment problem. Usually, you'd write a REST API, then a CLI wrapper, then an MCP server — three separate codebases for the same logic. Nexus generates all three from your workflow definition."
- If beginners look confused: "Think of Nexus as a deployment machine. You write the logic once, and it exposes it as a web API, a command-line tool, and an AI tool simultaneously."

**Transition**: "Let me show you how quick the setup is..."

---

## Slide 149: Nexus: Quick Start

**Time**: ~1 min
**Talking points**:

- Show the three ways to use the deployed service: `curl localhost:8000/analyze?town=Tampines` for API, `hdb-analyzer analyze --town Tampines` for CLI, and any MCP client for the MCP channel.
- If experts look bored: "Nexus generates OpenAPI specs, CLI argument parsers, and MCP manifests from the same workflow definition. Any change to the workflow is automatically reflected in all three interfaces."

**Transition**: "Production services need authentication and access control..."

---

## Slide 150: Authentication and RBAC

**Time**: ~1 min
**Talking points**:

- "Authentication answers: who are you? Authorisation answers: what can you do? RBAC lets you define roles with specific permissions."
- Example: an analyst can explore and predict but can't train new models — that's reserved for engineers.
- If experts look bored: "Nexus implements JWT-based authentication with RBAC at the route level. Roles and permissions are defined in the workflow decorator — no separate access control configuration needed."

**Transition**: "Middleware provides the cross-cutting concerns..."

---

## Slide 151: Middleware Stack

**Time**: ~1 min
**Talking points**:

- "Middleware is like airport security checkpoints. Every request passes through each checkpoint in order. If any checkpoint rejects the request, the request never reaches your handler."
- Highlight DriftMiddleware: "This automatically monitors every prediction for data drift — connected to your M4 DriftMonitor."
- If experts look bored: "Middleware ordering matters. Authentication should run before rate limiting (for per-user rate limits), which should run before drift monitoring (to avoid paying for monitoring unauthorised requests)."

**Transition**: "Drift monitoring connects your deployment to your training pipeline..."

---

## Slide 152: Monitoring and DriftMonitor Integration

**Time**: ~1 min
**Talking points**:

- "This connects M4's DriftMonitor to M9's Nexus deployment. Every prediction request is automatically checked for data drift. If the incoming data looks different from training data, you get an alert."
- If experts look bored: "The DriftMiddleware uses the reference distribution from your TrainingPipeline (M4) to compute PSI and KS statistics on incoming request features. Alerting thresholds are configurable per-feature."

**Transition**: "Session management is critical for conversational agents..."

---

## Slide 153: Session Management

**Time**: ~1 min
**Talking points**:

- "Session management is critical for production agents. Users expect continuity. Nexus handles session storage, expiry, and summarisation automatically."
- If experts look bored: "Nexus stores sessions in Redis with configurable TTL. The session summarisation hook is called when a session exceeds the configured token limit, ensuring context continuity without unbounded memory growth."

**Transition**: "Scaling to production requires addressing several concerns..."

---

## Slide 154: Scaling and Production Architecture

**Time**: ~1 min
**Talking points**:

- "Production deployment is not just 'make it work' — it's 'make it work reliably at scale under adversarial conditions.' This checklist covers the non-negotiables."
- If experts look bored: "Horizontal scaling of stateless Nexus workers behind a load balancer is straightforward. The stateful component is the session store (Redis). For GPU-backed inference, use a separate inference cluster with its own auto-scaling policy."

**Transition**: "Webhooks complete the event-driven monitoring picture..."

---

## Slide 155: Webhooks: Event-Driven Notifications

**Time**: ~1 min
**Talking points**:

- "Webhooks let your system push events to external services. When DriftMonitor detects a shift, Slack gets notified. When a prediction fails, PagerDuty creates an alert."
- If experts look bored: "Nexus's webhook system uses at-least-once delivery with configurable retry backoff. For compliance use cases, the webhook payload includes a signature for verification."

**Transition**: "Good API design is table stakes for production ML..."

---

## Slide 156: API Design for ML Services

**Time**: ~1 min
**Talking points**:

- "Include health checks (for load balancers), batch endpoints (for efficiency), and streaming (for UX). Nexus generates most of these automatically."
- Health checks are mandatory for Kubernetes deployments.
- If experts look bored: "Streaming endpoints for LLM responses require chunked transfer encoding with proper SSE formatting. Nexus handles this automatically — you just yield from your workflow handler."

**Transition**: "Error tracking closes the observability loop..."

---

## Slide 157: Error Tracking and Alerting

**Time**: ~1 min
**Talking points**:

- "Production monitoring answers: is it working? is it fast? is it accurate? is it affordable? Track all four dimensions."
- If experts look bored: "The p99 latency threshold is the most operationally important metric for interactive ML services. Set your SLO budget based on p99, not p50 — rare slow requests are what users complain about."

**Transition**: "Here's the complete Nexus summary..."

---

## Slide 158: Kailash Bridge: Production Deployment

**Time**: ~1 min
**Talking points**:

- "Nexus eliminates the deployment busywork. Instead of writing API routes, CLI parsers, and MCP servers separately, you define your workflow once and Nexus generates all three."
- Add middleware for auth, monitoring, and rate limiting — done.
- If experts look bored: "The three-channel deployment pattern aligns with three stakeholder types in enterprise ML: engineering (API), operations (CLI), and AI systems (MCP). One codebase, three interfaces."

**Transition**: "Now let's talk about how to evaluate the models we've been using. Section G — LLM Benchmarks."

---

## Slide 159: G — LLM Benchmarks (Q1 2026)

**Time**: ~2 min
**Talking points**:

- "Benchmarks are the standardised tests of the AI world. They tell you how models compare, but they don't tell the whole story. A model that scores 94% on GPQA might still fail on your specific use case."
- Key message: "Use benchmarks for shortlisting, then evaluate on YOUR data."
- If experts look bored: "The benchmark saturation problem is real — MMLU, HellaSwag, and TruthfulQA are all close to saturated. The field is moving to harder benchmarks: GPQA Diamond, AIME 2025, LiveCodeBench."

**Transition**: "Chatbot Arena is the most trustworthy evaluation system we have..."

---

## Slide 160: Chatbot Arena and Elo Ratings

**Time**: ~1 min
**Talking points**:

- "Chatbot Arena is like a boxing tournament for AI. Models fight head-to-head, anonymously, and users vote on the winner. Over time, the Elo ratings converge to a reliable ranking."
- This is arguably the most trustworthy evaluation system we have — real users, real queries, blind comparison.
- If experts look bored: "The Bradley-Terry model underlies the Elo calculation. Elo computed from pairwise comparisons is more robust than absolute scoring because it's immune to score scale differences across evaluators."

**Transition**: "For production systems, LLM-as-judge scales evaluation economically..."

---

## Slide 161: LLM-as-Judge

**Time**: ~1 min
**Talking points**:

- "LLM-as-judge gives you 80% of the quality at 1% of the cost of human evaluation."
- Warn: "Always account for the biases. Swap positions, use strict rubrics, and validate a sample against human judgements."
- If experts look bored: "Known biases: positional bias (favours first response), verbosity bias (favours longer responses), self-enhancement bias. Mitigation: swap A/B positions, use shorter but more precise rubrics, calibrate against human labels."

**Transition**: "Red teaming finds the failure modes before deployment..."

---

## Slide 162: Red Teaming LLMs

**Time**: ~1 min
**Talking points**:

- "Red teaming is adversarial testing — trying to break your own system before attackers do. For production LLM deployments, red teaming is not optional."
- Tools: Garak, Promptfoo, manual red teaming.
- If experts look bored: "Red teaming LLM agents requires adversarial examples at three levels: prompt injection (input), tool misuse (action), and output safety (generation)."

**Transition**: "Match the evaluation metric to the specific task..."

---

## Slide 163: Task-Specific Evaluation

**Time**: ~1 min
**Talking points**:

- "Match the metric to the task. If you're building a code agent, pass@k is the right metric. If you're building a RAG system, RAGAS faithfulness matters most."
- Evaluation matrix: code (pass@k, SWE-bench), QA (RAGAS, F1), summarisation (ROUGE, G-Eval), classification (accuracy, F1).

**Transition**: "Build your evaluation pipeline early..."

---

## Slide 164: Building Your Own Evaluation Pipeline

**Time**: ~1 min
**Talking points**:

- "Build your evaluation pipeline early, not as an afterthought. 50+ test cases covering: normal questions, edge cases, adversarial inputs, and out-of-scope questions."
- If experts look bored: "The evaluation dataset is as important as the production dataset. Invest in quality: diverse query types, ground truth answers from subject matter experts, and adversarial cases from a red team."

**Transition**: "Let me catalogue the evaluation anti-patterns..."

---

## Slide 165: Evaluation Anti-Patterns

**Time**: ~1 min
**Talking points**:

- "The most common evaluation mistake: testing on 5 examples and declaring success. Systematic evaluation with 50-100+ test cases is the minimum."
- If experts look bored: "Goodhart's Law applies heavily to LLM evaluation. Maintain a held-out blind test set that is never used for development decisions."

**Transition**: "Here's how to use Kailash tools for evaluation..."

---

## Slide 166: Kailash Bridge: LLM Evaluation

**Time**: ~1 min
**Talking points**:

- "Use ExperimentTracker to log every evaluation run. Compare RAG v1 vs v2 vs v3 with tracked metrics. Use Delegate with evaluation Signatures to build consistent LLM-as-judge pipelines."
- This connects back to M3's ExperimentTracker — evaluation is just another experiment.
- If experts look bored: "The evaluation loop in Kaizen: generate answers with your RAG/agent system → score with LLM-as-judge Delegate → log to ExperimentTracker → compare runs. Fully automated, fully reproducible."

**Transition**: "Now let's synthesise everything into a complete architecture. Section H."

---

## Slide 167: H — Kailash Agent Architecture

**Time**: ~2 min
**Talking points**:

- "This is the complete Kaizen toolkit. For 90% of use cases, you'll use Delegate with a Signature. For RAG, use RAGResearchAgent. For custom reasoning, use BaseAgent."
- Walk through the architecture diagram top-to-bottom.
- If experts look bored: "The architecture follows the same composition principle as the rest of Kailash: simple primitives compose into complex systems. No magic, just composition."

**Transition**: "The pattern library gives you reusable blueprints..."

---

## Slide 168: Agent Pattern Library

**Time**: ~1 min
**Talking points**:

- "SequentialPattern is the simplest coordination — chain agents in order. Each agent's output becomes the next agent's input. Research → Analysis → Report."
- Give concrete examples for each pattern.
- If experts look bored: "Use patterns for standard coordination, primitives for custom reasoning strategies. The pattern library is the high-level API; BaseAgent, Delegate, and Signature are the low-level API."

**Transition**: "RAGResearchAgent is a complete RAG pipeline in one class..."

---

## Slide 169: RAGResearchAgent

**Time**: ~1 min
**Talking points**:

- "RAGResearchAgent handles embedding, retrieval, re-ranking, prompt augmentation, and generation. The Signature ensures the output always includes sources — no citations, no answer."
- Show the constructor parameters: document list, chunk size, re-ranking enabled, output Signature.
- If experts look bored: "RAGResearchAgent is a BaseAgent that uses Delegate internally for LLM calls, a vector store for retrieval, and a cross-encoder for re-ranking. Each layer is independently replaceable."

**Transition**: "ML Agents bridge the gap between ML pipelines and conversational AI..."

---

## Slide 170: ML Agents Integration

**Time**: ~1 min
**Talking points**:

- "This is the end-to-end ML workflow augmented by agents. DataExplorer profiles the data (M1). DataScientist suggests features. ModelSelector recommends algorithms. TrainingPipeline trains the model (M4). Each step connects to the next."
- If experts look bored: "The ML agents pattern is an implementation of the Plan-and-Execute architecture applied to ML workflows. The orchestrator (LLM) plans the analysis; the workers (Kailash ML engines) execute deterministically."

**Transition**: "The full deployment pipeline from agent to production..."

---

## Slide 171: Full Agent-to-Deployment Pipeline

**Time**: ~1 min
**Talking points**:

- "This is the complete picture. M1 taught you DataExplorer. M3-M5 taught you TrainingPipeline and ModelRegistry. M9 adds the agent layer on top. M10 will add governance."
- If experts look bored: "M9 doesn't replace any previous engine — it adds a coordination layer on top. Your DataExplorer, TrainingPipeline, and DriftMonitor are now tools that an agent can orchestrate automatically."

**Transition**: "ChainOfThoughtAgent is ideal when you need auditable reasoning..."

---

## Slide 172: ChainOfThoughtAgent: Structured Reasoning

**Time**: ~1 min
**Talking points**:

- "ChainOfThoughtAgent is ideal for diagnostic tasks where you need to see the reasoning. It generates a step-by-step analysis before giving the final answer. The reasoning is part of the output — auditable and verifiable."
- If experts look bored: "The structured CoT output enables automated validation of intermediate steps. You can check whether each reasoning step is factually grounded using your RAG pipeline."

**Transition**: "Model selection strategy determines your cost-quality profile..."

---

## Slide 173: Cost Optimization: Model Selection Strategy

**Time**: ~1 min
**Talking points**:

- "Don't use expensive frontier models for formatting a report — use a cheap fast model. Don't use a cheap model for complex planning — use a frontier model. Match model capability to task difficulty."
- Model names always from environment variables — swap models without changing business logic.
- If experts look bored: "The routing decision can itself be automated using a small classifier trained on task descriptions — automatically selecting the optimal model tier based on task complexity signals."

**Transition**: "Streaming is critical for production agent UX..."

---

## Slide 174: Agent Streaming and Events

**Time**: ~1 min
**Talking points**:

- "Streaming gives users real-time visibility into what the agent is doing. They see reasoning, tool calls, and results as they happen — not a blank screen for 30 seconds followed by a wall of text."
- This is critical for user trust and production UX.
- If experts look bored: "Kaizen's streaming uses SSE internally. Each event type — reasoning_step, tool_call, tool_result, final_answer — can be individually subscribed to for fine-grained UX control."

**Transition**: "Here's the complete Kailash architecture one final time..."

---

## Slide 175: Kailash Complete Architecture

**Time**: ~1 min
**Talking points**:

- "This is the full picture. You don't have to use everything — pick the layers you need. But when you compose them, each layer integrates seamlessly with the others."
- Walk through the layers: ML engines (M1-M8) → agent coordination (Kaizen) → deployment (Nexus) → governance (PACT, M10).
- If experts look bored: "The composition principle is the key architectural insight. Kailash doesn't replace your ML pipeline — it adds agency, deployment, and governance as composable layers."

**Transition**: "Let me set you up for the lab exercises..."

---

## Slide 176: Lab Setup

**Time**: ~1 min
**Talking points**:

- Walk through the exercise progression: "Each exercise builds on the previous one. By exercise 6, you'll have a full multi-agent research system deployed via Nexus with monitoring."
- Environment variables for model names — never hardcode in your solutions.
- If beginners look confused: "Follow the setup steps in order. Each exercise takes 20-30 minutes. Ask for help early — don't spend more than 10 minutes stuck."

**Transition**: "Let me facilitate a few discussion questions before we wrap up..."

---

## Slide 177: Discussion: RAG Debugging

**Time**: ~1 min
**Talking points**:

- Let students reason through this. The key insight: "When RAG gives a wrong answer, the problem is in one of four places: retrieval, context, generation, or the prompt. Check each systematically."
- Walk the debugging flowchart: did you retrieve the right chunks? → did the context contain the answer? → did the model faithfully use the context? → was the prompt correctly formulated?
- If nobody speaks: "Start by asking: if I put the correct answer directly in the prompt, does the model give it back correctly? That rules out generation as the problem."

**Transition**: "Let's debug a different kind of agent failure..."

---

## Slide 178: Discussion: Agent Loop Optimization

**Time**: ~1 min
**Talking points**:

- "47 tool calls for a simple question is almost always a tool description problem. The agent is confused about which tool to use, so it tries all of them."
- Fix strategy: improve descriptions, add examples, reduce the number of available tools per agent.
- If nobody speaks: "Imagine you're the agent reading the tool descriptions for the first time. Which tool would you try for this question? If more than one seems right, that's your problem."

**Transition**: "Let's talk about the business case for agents..."

---

## Slide 179: Discussion: Agent ROI

**Time**: ~1 min
**Talking points**:

- "Always frame AI costs against the alternative — human labour. If the agent handles 1,000 queries per day that would each take a human 15 minutes, that's 250 human-hours per day. The agent costs $500 vs $25,000+ in human labour."
- If nobody speaks: "What's the most repetitive analytical task in your organisation? What would it cost per day to have humans do it? What would an agent cost to handle it?"

**Transition**: "Finally, let's talk about A2A security..."

---

## Slide 180: Discussion: A2A Security

**Time**: ~1 min
**Talking points**:

- "A2A opens your agent to requests from any other agent. This is powerful but dangerous. Authenticate every caller. Rate limit requests. Validate inputs. Don't expose internal data in responses."
- If nobody speaks: "If you published your agent's A2A endpoint publicly, what's the worst thing a malicious agent could do with it? That's what you need to defend against."

**Transition**: "Let's close with the key takeaways..."

---

## Slide 181: Synthesis — Key Takeaways

**Time**: ~1 min
**Talking points**:

- Read through each column. "Everyone should leave today knowing what an agent is and how to build one with Kailash."
- Three tiers of takeaways: all students (agent fundamentals, Kaizen Delegate, RAG pipeline), theory followers (why these architectures work), experts (cutting-edge techniques still being developed).
- If beginners look confused: "The three things you must remember: LLMs predict tokens; RAG gives them knowledge; agents give them tools."

**Transition**: "Look at how M9 fits into the cumulative engine map..."

---

## Slide 182: Kailash Engines — Cumulative Map

**Time**: ~1 min
**Talking points**:

- "Notice how M9 doesn't replace any previous engine — it adds a coordination layer on top. Your DataExplorer, TrainingPipeline, and DriftMonitor are now tools that an agent can orchestrate automatically."
- This is the cumulative power of the course: each module adds a layer, each layer is usable independently or composable.
- If experts look bored: "The agents-as-orchestrators pattern is the key architectural insight. Kaizen agents don't contain ML logic — they coordinate Kailash ML engines via tool calls. The ML logic stays testable."

**Transition**: "Let me tell you what's coming in Module 10..."

---

## Slide 183: Preview: Module 10

**Time**: ~1 min
**Talking points**:

- "M9 gave you the ability to build agents. M10 teaches you how to make them safe, aligned, and governed. How to fine-tune models for your domain. How to enforce rules that agents can't break."
- New engines: AlignmentPipeline, AdapterRegistry, GovernanceEngine, PactGovernedAgent, RLTrainer.
- If experts look bored: "M10 introduces the D/T/R accountability grammar from PACT — Decide, Take action, Review. This is the formal framework for defining human-agent operating envelopes."

**Transition**: "The appendix slides are available for deeper dives on specific topics..."

---

## Slide 184: Attention Mechanism: Step-by-Step Review

**Time**: ~1 min
**Talking points**:

- Quick M8 review: "This is the engine inside every LLM. When the model decides that 'capital' is relevant to 'Singapore', it's because the Q vector of 'capital' has high dot product with the K vector of 'Singapore'."
- If experts have questions: "Walk through the four steps: linear projection to Q/K/V, scaled dot-product, softmax, weighted sum of V."

**Transition**: "Multi-head attention multiplies this by the number of heads..."

---

## Slide 185: Multi-Head Attention: Why Multiple Heads?

**Time**: ~1 min
**Talking points**:

- "Think of each attention head as a different lens. One looks for grammatical relationships. Another looks for semantic similarity. Another looks for long-range dependencies."
- If experts look bored: "Each head operates in a lower-dimensional subspace (d_model / n_heads). The concatenated outputs are projected back to d_model. Head specialisation is emergent — not enforced by the architecture."

**Transition**: "The sqrt(d_k) scaling factor is easily overlooked but important..."

---

## Slide 186: Why Scale by sqrt(d_k)?

**Time**: ~1 min
**Talking points**:

- "Without scaling, attention becomes binary — either full attention or zero attention. With scaling, attention can be graduated — some tokens get 70%, others 20%. This graded attention is what makes transformers powerful."
- If experts look bored: "The variance of the dot product between two d_k-dimensional vectors scales with d_k. Dividing by sqrt(d_k) normalises the variance back to ~1, keeping softmax gradients from vanishing."

**Transition**: "BPE tokenisation in detail..."

---

## Slide 187: BPE Tokenization: Detailed Example

**Time**: ~1 min
**Talking points**:

- "BPE is how every modern LLM tokeniser works. Start with characters, iteratively merge the most common pairs, until you reach target vocabulary size."
- Live demo: use tiktoken or the Hugging Face tokeniser on a sentence.
- If experts look bored: "Llama uses ~32K tokens; GPT-4 uses ~100K; Qwen uses ~150K for multilingual efficiency. Larger vocabulary = fewer tokens per text = shorter sequences = cheaper inference."

**Transition**: "The Chinchilla paper fundamentally changed LLM training strategy..."

---

## Slide 188: The Chinchilla Paper: Impact on LLM Design

**Time**: ~1 min
**Talking points**:

- "Chinchilla proved the industry was building models too large and trained on too little data. The optimal: smaller models, more data."
- This directly led to Llama's training strategy.
- If experts look bored: "Chinchilla's compute-optimal frontier (Hoffmann et al., 2022): N ∝ C^0.5 and D ∝ C^0.5. Llama's insight: if you care about inference cost, over-train smaller models — inference-optimal does not equal compute-optimal."

**Transition**: "Embedding similarity is the foundation of all RAG retrieval..."

---

## Slide 189: Embedding Similarity: Visual Intuition

**Time**: ~1 min
**Talking points**:

- "Think of embeddings as GPS coordinates for meaning. Sentences about economics are clustered together. When you search 'Singapore GDP', the system finds the nearest points in meaning-space."
- Cosine similarity vs dot product: cosine similarity normalises for vector length, making it robust to documents of different lengths.

**Transition**: "And the k-NN search is how we find those nearest points efficiently..."

---

## Slide 190: Vector Search: k-Nearest Neighbors

**Time**: ~1 min
**Talking points**:

- "This is the core retrieval operation. Your query becomes a vector, and we find the most similar document vectors. With HNSW indexing, this takes milliseconds even across millions of documents."
- Exact vs approximate: "For most RAG applications, approximate nearest neighbours is sufficient and orders of magnitude faster than exact search."

**Transition**: "Prompt injection is the most dangerous real-world attack vector..."

---

## Slide 191: Prompt Injection: Real Attack Examples

**Time**: ~1 min
**Talking points**:

- "These are real attack patterns from Q4 2025 data. The most successful vector is indirect injection — malicious instructions hidden in documents that the agent retrieves."
- Walk through each example. Ask: "How would you defend against this in your RAG system?"
- If experts look bored: "The indirect injection attack chain: attacker embeds instructions in a public document → victim's RAG retrieves the document → the injected instructions override the agent's system prompt → attacker achieves data exfiltration."

**Transition**: "The decision tree helps you choose the right agent pattern..."

---

## Slide 192: Agent Decision Trees: Which Pattern to Use?

**Time**: ~1 min
**Talking points**:

- "This decision tree covers 95% of agent use cases. Start with the simplest option (Delegate) and only escalate when you genuinely need specialisation or parallelism."
- Walk the tree: single task → Delegate; needs RAG → RAGResearchAgent; needs specialisation → SupervisorWorkerPattern; needs verification → DebatePattern.

**Transition**: "The development lifecycle structures how you build agents systematically..."

---

## Slide 193: The Agent Development Lifecycle

**Time**: ~1 min
**Talking points**:

- "Define the contract first (what goes in, what comes out), build the simplest working version, measure its quality, then add error handling and monitoring before deploying."
- If experts look bored: "The contract-first approach (Signature before implementation) enables parallel development: the frontend team can mock responses using the Signature schema while the backend team implements the agent."

**Transition**: "MCP transport options affect deployment architecture..."

---

## Slide 194: MCP Transport Comparison

**Time**: ~1 min
**Talking points**:

- "stdio is for local development — your terminal is the client. Streamable HTTP is for production — any client anywhere can connect."
- When you deploy an MCP server with Nexus, it uses Streamable HTTP automatically.
- If experts look bored: "SSE transport is being deprecated in favour of Streamable HTTP, which supports bidirectional streaming over standard HTTP/2. This is the main transport change in MCP spec v0.7."

**Transition**: "Nexus gives you three channels from one codebase..."

---

## Slide 195: Nexus: Three Channels from One Codebase

**Time**: ~1 min
**Talking points**:

- "Same workflow, three interfaces. Your web app calls the API. Your DevOps scripts use the CLI. Your AI assistants connect via MCP. One codebase handles all three."
- If experts look bored: "The channel abstraction in Nexus decouples the transport from the business logic. Adding a new channel requires only a new transport adapter."

**Transition**: "GPQA Diamond is the benchmark that actually differentiates frontier models..."

---

## Slide 196: GPQA Diamond: The New Gold Standard

**Time**: ~1 min
**Talking points**:

- "GPQA Diamond is the benchmark that actually differentiates frontier models. MMLU is saturated. GPQA has hard enough questions that even human experts struggle."
- If experts look bored: "GPQA Diamond (Graduate-Level Google-Proof Q&A) requires expert-level knowledge that can't be found by searching the web. The 'diamond' subset is the hardest 198 questions — those where even domain experts get it wrong most of the time."

**Transition**: "SWE-bench is the most relevant benchmark for practical software engineering..."

---

## Slide 197: SWE-bench: Can Agents Fix Real Bugs?

**Time**: ~1 min
**Talking points**:

- "SWE-bench: can an agent autonomously fix bugs in real open-source projects? Frontier agents resolve about 70% of issues. Remarkable but means 30% of real bugs are still beyond them."
- If experts look bored: "The February 2026 SWE-bench update upgraded scaffolding and environments, making scores more meaningful — realistic repository sizes and dependencies, not trimmed toy versions."

**Transition**: "ARC-AGI tests something fundamentally different from all other benchmarks..."

---

## Slide 198: ARC-AGI: Measuring Abstract Reasoning

**Time**: ~1 min
**Talking points**:

- "ARC-AGI is the test that keeps AI researchers humble. LLMs excel at tasks they've seen during training. ARC-AGI presents completely novel patterns. The 40% vs 98% human gap shows how far we are from general reasoning."
- If experts look bored: "ARC-AGI was created by Chollet to test pattern completion on tasks that require genuine abstraction, not memorisation. The 2024 contest winner reached 55% using test-time compute tricks. Human baseline is 98%."

**Transition**: "One final synthesis slide before the assessment..."

---

## Slide 199: The Full M9 Stack: Everything Connected

**Time**: ~2 min
**Talking points**:

- Final synthesis: "Look at how each section builds on the previous one. You can't build a good RAG system without understanding embeddings. You can't build a good agent without understanding prompt engineering. It all connects."
- If beginners look confused: "The stack from bottom to top: LLM (Section A) → prompting (B) → retrieval (C) → agents (D) → tools (E) → deployment (F) → evaluation (G) → integration (H)."

**Transition**: "Let me tell you what the assessment will look like..."

---

## Slide 200: Assessment Preview

**Time**: ~2 min
**Talking points**:

- "The assessment tests your specific exercise outputs — your agent's tool calls, your RAG retrieval results, your Nexus deployment configuration. You can't copy-paste answers from any AI assistant because the questions are about what YOUR code produced."
- This is AI-resilient assessment: context-specific application, process documentation, debugging, architecture decisions.
- If beginners look anxious: "The assessment is about demonstrating that you ran the exercises and understood what happened. Every question connects to a specific exercise step."
- If experts look bored: "The hardest questions ask you to debug a broken RAG pipeline or explain why your agent chose a specific tool sequence. These require running the code, not reading the slides."

**Transition**: "See you in Module 10 — where we teach your agents right from wrong."

---
