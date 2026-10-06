# Module 6 — Machine Learning with Language Models and Agentic Workflows

> _"The model is not the product. The governed system is the product."_

This is the final chapter of the MLFP programme. You arrived here from a long road: Python basics and data wrangling in Module 1, statistics and probability in Module 2, the full supervised ML pipeline in Module 3, unsupervised learning and the neural network bridge in Module 4, and every major deep learning architecture in Module 5. Everything you have learned converges here.

Module 6 is about building production AI systems with large language models. Not using them — building them. You will engineer prompts, fine-tune models with LoRA, align them with human preferences using DPO, ground them in facts with RAG, give them tools through agents, coordinate multiple agents, govern them with PACT, and deploy them with Nexus. Every lesson produces running code. Every system you build is governed — with access controls, cost budgets, operating envelopes, and audit trails.

The organising principle of Module 6 is the transition from model to system. A model takes an input and produces an output. A system takes a goal and achieves it — reasoning about sub-tasks, selecting tools, retrieving knowledge, coordinating specialists, respecting boundaries, and explaining its decisions. The distance between a model and a system is the distance between a trained neural network and a production application. This chapter covers that distance.

By the end of this chapter you will have deployed a complete, governed, multi-agent AI system accessible via API, CLI, and MCP simultaneously. That is the capstone of the MLFP programme, and it is the starting point of your career as an ML engineer.

---

## Learning Outcomes

By the end of this chapter you will be able to:

- Use LLMs effectively with prompt engineering techniques (zero-shot, few-shot, chain-of-thought, self-consistency) and extract structured output using Kaizen's Delegate and Signature APIs.
- Implement LoRA from scratch, understanding the low-rank factorisation mathematics, and implement adapter layers from scratch. Survey all 10+ fine-tuning techniques and select the right one for a given scenario.
- Derive the DPO loss function from the Bradley-Terry preference model and the RLHF objective. Implement DPO training with preference pairs. Explain GRPO and when to prefer it over DPO. Evaluate aligned models with LLM-as-judge and standard benchmarks.
- Build complete RAG pipelines with chunking, dense retrieval, sparse retrieval (BM25), hybrid retrieval, re-ranking, and HyDE. Evaluate RAG quality with RAGAS metrics.
- Build ReAct agents with custom tools, implement function calling with structured schemas, and bound agents with turn ceilings and cost budgets to prevent runaway loops and spending.
- Implement multi-agent patterns (supervisor-worker, sequential, parallel, handoff), build MCP servers, and configure agent memory.
- Implement PACT governance with D/T/R addressing, operating envelopes, budget cascading, and governance testing.
- Deploy a complete AI system with Nexus (API + CLI + MCP), implement authentication, integrate drift monitoring, and verify governance at the deployment level.

---

## Prerequisites

**Module 5 complete.** This chapter assumes you can:

- Fine-tune pre-trained models (BERT, ResNet) using transfer learning.
- Implement and train any major deep learning architecture.
- Export models for production with ONNX.
- Explain how PPO is used in RLHF for LLM alignment.

**From Module 4 specifically:** TF-IDF and BM25 (Lesson 4.6) and word embeddings (Lesson 4.6) — these reappear in the RAG lesson.

**From Module 3 specifically:** Evaluation metrics and drift monitoring (Lessons 3.5, 3.8) — these reappear in the capstone.

**Notation:**

- $\pi_\theta$ is a policy (the LLM) parameterised by $\theta$.
- $\pi_{\text{ref}}$ is a reference policy (the original, unaligned model).
- $y_w$ and $y_l$ are preferred (winning) and dispreferred (losing) responses.
- $\beta$ is the KL-penalty coefficient in DPO.
- $\sigma$ is the sigmoid function.

---

## How to Read This Chapter

Same structure as all previous modules. The scaffolding level is minimal (~20% code provided) — you are now a fluent ML engineer.

**Estimated reading time per lesson:**

| Lesson | Title                                                         | Reading | Exercise | Total   |
| ------ | ------------------------------------------------------------- | ------- | -------- | ------- |
| 6.1    | LLM Fundamentals and Prompt Engineering                       | 100 min | 60 min   | ~2h 40m |
| 6.2    | LLM Fine-Tuning — LoRA, Adapters, and the Technique Landscape | 130 min | 80 min   | ~3h 30m |
| 6.3    | Preference Alignment — DPO and GRPO                           | 120 min | 75 min   | ~3h 15m |
| 6.4    | RAG Systems                                                   | 110 min | 70 min   | ~3h     |
| 6.5    | AI Agents — ReAct, Tool Use, and Function Calling             | 110 min | 65 min   | ~2h 55m |
| 6.6    | Multi-Agent Orchestration and MCP                             | 110 min | 70 min   | ~3h     |
| 6.7    | AI Governance Engineering                                     | 100 min | 65 min   | ~2h 45m |
| 6.8    | Capstone — Full Production Platform                           | 120 min | 80 min   | ~3h 20m |

Total: roughly 24 hours.

---

# Lesson 6.1: LLM Fundamentals, Prompt Engineering, and Structured Output

## Why This Matters

The transformer architecture you built in Lesson 5.4 is the foundation of every large language model. GPT, Claude, Gemini, Llama — they are all transformers, scaled to billions of parameters and trained on trillions of tokens. But a pre-trained language model is not a product. It is a foundation. To turn it into a product, you need three capabilities: the ability to control its output (prompt engineering), the ability to customise it for your domain (fine-tuning, Lesson 6.2), and the ability to align it with human values (preference alignment, Lesson 6.3).

This lesson focuses on the first capability: prompt engineering. The difference between a well-prompted and poorly-prompted LLM can be the difference between a useful system and an unreliable one. You will learn five prompting techniques, from zero-shot to self-consistency, and you will use Kaizen's structured output APIs to extract type-safe results instead of free-form text.

## Core Concepts

### FOUNDATIONS: How LLMs are trained

An LLM like GPT is trained in two stages:

**Pre-training.** A GPT-style (decoder-only) model learns to predict the next token in a sequence. Given "The capital of Singapore is", the model learns to assign high probability to "Singapore" (or rather, to the token that represents "Singapore"). The training corpus is a large fraction of the internet — books, Wikipedia, code, web pages. Pre-training on trillions of tokens gives the model a broad understanding of language, facts, reasoning patterns, and code. BERT-style (encoder-only) models are pre-trained differently, with **masked language modelling**: about 15% of the input tokens are hidden and the model predicts them from context on both sides. That makes BERT a strong text encoder (you fine-tune one in Lesson 6.2) but not a text generator.

**Alignment.** The pre-trained model is a next-token predictor, not a helpful assistant. It will happily complete a harmful prompt or generate nonsense that looks authoritative. Alignment tunes the model to be helpful, harmless, and honest. This is done through RLHF (Reinforcement Learning from Human Feedback) or DPO (Direct Preference Optimization, Lesson 6.3): human annotators rank model outputs, and the model is trained to prefer the higher-ranked outputs.

**Scaling laws.** Pre-training loss falls predictably as you scale three factors: the number of parameters $N$, the number of training tokens $D$, and the compute spent (roughly $6ND$ floating-point operations). Hoffmann et al. (2022, the "Chinchilla" paper) fitted

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

where $E$ is the irreducible loss — the entropy of natural text, which no model can beat — and the two power-law terms shrink as the model and the data grow. Two consequences matter in practice: loss improves by a roughly constant amount each time you multiply $N$ or $D$ by a constant factor (diminishing returns, never zero loss), and for a fixed compute budget the loss is lowest when parameters and tokens grow together (Chinchilla's rule of thumb is about 20 training tokens per parameter). This is why LLMs grew from about 117 million parameters (GPT-1, 2018) to 175 billion (GPT-3, 2020) and beyond. The largest commercial models do not publish their parameter counts.

### FOUNDATIONS: Prompt engineering techniques

**Zero-shot prompting.** Give the model only a task description, no examples:

```
Classify the following review as positive or negative:
Review: "The laksa at this hawker stall is the best I've had in Katong."
Sentiment:
```

**Few-shot prompting.** Provide examples before the query:

```
Review: "Excellent char kway teow, generous portions." -> Positive
Review: "Too salty, overpriced for hawker standards." -> Negative
Review: "The laksa at this hawker stall is the best I've had in Katong." ->
```

**Chain-of-thought (CoT).** Prompt the model to reason step by step:

```
Q: Suppose a taxi charges S$3.90 flag-down + S$0.25 per 400m.
   What is the fare for a 12km trip?

Let's think step by step:
1. Total distance: 12,000m
2. Number of 400m units: 12,000 / 400 = 30
3. Distance charge: 30 × S$0.25 = S$7.50
4. Total fare: S$3.90 + S$7.50 = S$11.40
```

**Zero-shot CoT.** Append "Let's think step by step" without providing examples. Surprisingly effective — it elicits step-by-step reasoning without hand-crafted chain-of-thought examples. Kojima et al. (2022) reported that these five words lifted a 175B-parameter model (text-davinci-002) from 17.7% to 78.7% accuracy on the MultiArith arithmetic benchmark, and from 10.4% to 40.7% on GSM8K. Gains are largest on multi-step arithmetic and logic; on simple classification the extra tokens often buy little.

**Self-consistency.** Sample multiple chain-of-thought paths at a non-zero temperature and take the majority vote over their final answers (Wang et al., 2023). This reduces the variance of CoT prompting by aggregating diverse reasoning paths. It costs $N\times$ the tokens of a single call.

**Structured prompting.** Specify the output format explicitly — "reply with JSON containing `sentiment` and `confidence`", or a table with named columns. A stated format makes the output machine-checkable, which is the bridge to the typed Signatures below.

### FOUNDATIONS: Kaizen structured output

Free-form text is unreliable for production systems — the output format varies between calls. Kaizen provides structured output through Signatures:

```python
from kaizen import Signature, InputField, OutputField
from kaizen.core.base_agent import BaseAgent
from shared.mlfp06._ollama_bootstrap import DEFAULT_CHAT_MODEL, OLLAMA_BASE_URL

class SentimentSignature(Signature):
    """Classify the sentiment of a product review."""
    review: str = InputField(description="The review text")
    sentiment: str = OutputField(description="One of: positive, negative, neutral")
    confidence: float = OutputField(description="Confidence score 0-1")

# Local Ollama model; the model name comes from OLLAMA_CHAT_MODEL
# (default llama3.2:3b). No API key, no paid provider.
config = {
    "llm_provider": "ollama",
    "model": DEFAULT_CHAT_MODEL,
    "base_url": OLLAMA_BASE_URL,
    "use_async_llm": True,              # required for run_async
    "temperature": 0.2,
    "response_format": {"type": "json_object"},
    "structured_output_mode": "explicit",
}
agent = BaseAgent(config=config, signature=SentimentSignature())

result = await agent.run_async(review="Best laksa in Katong!")
print(result["sentiment"])    # e.g. "positive"
print(result["confidence"])   # e.g. 0.9 (the model's self-reported score)
```

The Signature defines the input and output schema. The BaseAgent renders that schema into the prompt, makes the LLM call, and parses the reply into a dict keyed by the OutputField names, not an unparsed string. Parsing can still fail — a small local model sometimes omits a field — so production code checks that every expected key is present rather than filling a placeholder. The `confidence` value is whatever number the model writes; it is not a calibrated probability.

### ADVANCED: Inference considerations

**KV-cache.** During autoregressive generation, each new token attends to every previous token. Without a cache, the model would recompute the keys and values of the whole prefix at every step. The KV-cache stores them once and appends one new key/value pair per layer per step, trading memory for compute. The cache grows linearly with sequence length and batch size, and every decoding step must read the model weights plus the cache from GPU memory to produce a single token — which is why token-by-token decoding is limited by memory bandwidth rather than arithmetic.

**Speculative decoding.** A small, fast "draft" model generates candidate tokens, which a larger model verifies in parallel. This can speed up generation by 2–3× without changing the output distribution.

**Continuous batching.** Instead of waiting for all requests in a batch to finish, new requests are added to the batch as old ones complete. This maximises GPU utilisation for serving.

## Mathematical Foundations

### THEORY: The softmax temperature

The LLM's output is a probability distribution over the vocabulary, computed via softmax with a temperature parameter $T$:

$$P(w_i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

At $T = 1$ (default), the distribution is as trained. At $T < 1$, the distribution becomes peakier (more deterministic — the model is more confident). At $T > 1$, the distribution becomes flatter (more random — the model explores more). Temperature 0 is equivalent to argmax (always pick the most likely token). For factual tasks, use low temperature; for creative tasks, use higher temperature.

## The Kailash Engine: Kaizen Delegate via the M6 Bootstrap

M6 routes every LLM call through `shared.mlfp06._ollama_bootstrap.make_delegate`, which constructs a Kaizen `Delegate` backed by a locally-running **Ollama** daemon. There are no API keys to manage, and the `OllamaUnreachableError` raised when the daemon is down points the student straight at the fix command (`ollama serve` / `ollama pull <model>`). See `specs/redlines.md` Redline 14 for the full mandate.

```python
from shared.mlfp06._ollama_bootstrap import make_delegate, run_delegate_text

# No model= argument: the model comes from OLLAMA_CHAT_MODEL (default
# llama3.2:3b). Passing model= explicitly would bypass that setting.
delegate = make_delegate(temperature=0.4)

reviews = [
    "The laksa at this hawker stall is the best I've had in Katong.",
    "Too salty, overpriced for hawker standards.",
]

# run_sync returns the complete response as a plain string
for review in reviews:
    text = delegate.run_sync(
        f"Classify this review as positive or negative. Reply with one word.\n\n{review}"
    )
    print(f"{review[:40]!r} -> {text.strip()}")

# run_delegate_text streams the same call and also reports token usage
text, usage, seconds = await run_delegate_text(delegate, f"Classify: {reviews[0]}")
print(usage["total_tokens"], f"{seconds:.1f}s")
```

Cost note: the bootstrap forces `budget_usd=None` because Kaizen's cost estimator mis-prices Ollama at hosted-API rates even though local inference is free. The honest comparison signal across techniques is **token count and latency**, surfaced via `run_delegate_text(...) -> (text, usage_dict, elapsed_s)`. Exercise 6.1 converts tokens into an illustrative hosted price only for comparison.

A `Delegate` is the streaming, free-text interface; a `BaseAgent` with a `Signature` (above) is the structured interface. Exercise 6.1 uses the Delegate for files 01–05 (zero-shot, few-shot, CoT, zero-shot CoT, self-consistency on SST-2 movie-review sentences) and the BaseAgent + Signature for file 06 (structured output).

## Worked Example: Prompt Engineering Comparison

```python
from kaizen import Signature, InputField, OutputField
from kaizen.core.base_agent import BaseAgent
from shared.mlfp06._ollama_bootstrap import DEFAULT_CHAT_MODEL, OLLAMA_BASE_URL

class MathSolver(Signature):
    """Solve a math word problem step by step."""
    problem: str = InputField(description="The word problem")
    reasoning: str = OutputField(description="Step-by-step reasoning")
    answer: float = OutputField(description="Final numerical answer")

config = {
    "llm_provider": "ollama",
    "model": DEFAULT_CHAT_MODEL,
    "base_url": OLLAMA_BASE_URL,
    "use_async_llm": True,
    "temperature": 0.2,
    "response_format": {"type": "json_object"},
    "structured_output_mode": "explicit",
}
agent = BaseAgent(config=config, signature=MathSolver())

# (problem, correct answer) — the answers are computed exactly below
problems = [
    ("An HDB flat costs S$485,000. The buyer pays 25% down and finances the "
     "rest at 2.6% annual interest over 25 years, repaid monthly. What is the "
     "monthly payment on the loan?", 1650.22),
    ("A hawker sells 150 plates of chicken rice per day at S$4.50 each. "
     "Operating costs are S$280 per day. What is the weekly profit?", 2765.00),
]

# Zero-shot vs CoT comparison
for problem, expected in problems:
    result_zero = await agent.run_async(problem=problem)
    result_cot = await agent.run_async(problem=f"Think step by step. {problem}")
    print(f"Expected {expected:,.2f} | zero-shot {result_zero.get('answer')} "
          f"| CoT {result_cot.get('answer')}")
    print(f"CoT reasoning: {result_cot.get('reasoning', '')[:200]}")
```

The reference answers are exact. The hawker problem is (150 × 4.50 − 280) × 7 = S$2,765. The loan is the annuity formula $M = P\,r(1+r)^n / ((1+r)^n - 1)$ with $P = 0.75 \times 485{,}000 = 363{,}750$, $r = 0.026/12$ and $n = 300$, giving S$1,650.22 a month. Check which of the two your model gets right. The hawker problem is three multiplications; the annuity needs $(1+r)^{300}$, which a language model cannot compute reliably token by token. If your model misses it even with CoT, that is a lesson in itself: for exact arithmetic, give the agent a calculator tool (Lesson 6.5) rather than more prompting. Your printed answers depend on your model and on sampling; `.get()` is used because a small model can omit a field.

## Try It Yourself

**Drill 1.** Compare zero-shot, few-shot (3 examples), and CoT prompting on 10 math word problems with known answers. Which technique achieves the highest accuracy? At what token cost per problem?

**Solution:** extend the worked example's `problems` list to ten `(problem, answer)` pairs. A Delegate gives you the token count of every call, so ask for the answer on a fixed last line and parse it.

```python
import re
from shared.mlfp06._ollama_bootstrap import make_delegate, run_delegate_text

delegate = make_delegate(temperature=0.0)
FORMAT = "End your reply with a final line of the form 'ANSWER: <number>'."
FEW_SHOT = (
    "Q: A stall sells 40 kopi at S$1.50. What is the revenue?\nANSWER: 60\n\n"
    "Q: A 3-room flat of 68 sqm sells for S$408,000. What is the price per sqm?\nANSWER: 6000\n\n"
    "Q: A bus travels 18 km in 45 minutes. What is its speed in km/h?\nANSWER: 24\n\n"
)
techniques = {
    "zero_shot": lambda p: f"{p}\n{FORMAT}",
    "few_shot": lambda p: f"{FEW_SHOT}Q: {p}\n{FORMAT}",
    "cot": lambda p: f"{p}\nLet's think step by step. {FORMAT}",
}

def parse_answer(text: str) -> float | None:
    match = re.search(r"ANSWER:\s*S?\$?\s*(-?[\d,]*\.?\d+)", text)
    return float(match.group(1).replace(",", "")) if match else None

async def compare(problems):
    for name, build in techniques.items():
        correct, tokens, unparsed = 0, 0, 0
        for problem, expected in problems:
            text, usage, _secs = await run_delegate_text(delegate, build(problem))
            tokens += usage["total_tokens"]
            answer = parse_answer(text)
            if answer is None:
                unparsed += 1           # count format failures separately
            elif abs(answer - expected) <= 0.01 * abs(expected):
                correct += 1            # within 1% of the exact answer
        print(f"{name:10s} {correct}/{len(problems)} correct, "
              f"{unparsed} unparsed, {tokens / len(problems):.0f} tokens/problem")

await compare(problems)
```

Expect CoT to cost several times the tokens of zero-shot. Whether it buys accuracy depends on how many reasoning steps the problems need. Report unparsed replies separately: a format failure is not the same error as a wrong answer.

**Drill 2.** Implement self-consistency: sample 5 CoT responses (temperature=0.7) and take the majority-vote answer. Does self-consistency improve accuracy over single-sample CoT?

**Solution:** reuse `techniques["cot"]` and `parse_answer` from Drill 1. Sampling needs a non-zero temperature, otherwise all five paths are (nearly) identical.

```python
from collections import Counter
from shared.mlfp06._ollama_bootstrap import make_delegate, run_delegate_text

sampler = make_delegate(temperature=0.7)

async def self_consistency(problem: str, n_samples: int = 5):
    answers = []
    for _ in range(n_samples):
        text, _usage, _secs = await run_delegate_text(sampler, techniques["cot"](problem))
        answer = parse_answer(text)
        if answer is not None:
            answers.append(round(answer, 2))
    if not answers:
        return None, 0.0                 # every path failed to give an answer
    winner, votes = Counter(answers).most_common(1)[0]
    return winner, votes / n_samples     # answer and its agreement rate

answer, agreement = await self_consistency(problems[1][0])
print(answer, f"agreement {agreement:.0%}")
```

The agreement rate is a useful by-product: when the five paths disagree, the problem is hard for this model and the answer deserves less trust.

**Drill 3.** Build a classification system with a typed Signature for the SST-2 movie-review sentences that Exercise 6.1 uses. Test it on 50 sentences and report accuracy against the gold labels.

**Solution:**

```python
from kaizen import Signature, InputField, OutputField
from kaizen.core.base_agent import BaseAgent
from shared.mlfp06._ollama_bootstrap import DEFAULT_CHAT_MODEL, OLLAMA_BASE_URL
from shared.mlfp06.ex_1 import load_sst2

class ReviewSentiment(Signature):
    """Classify the sentiment of a short movie-review phrase."""
    text: str = InputField(description="A sentence or phrase from a movie review")
    sentiment: str = OutputField(description="Exactly one of: positive, negative")
    confidence: float = OutputField(description="Confidence score 0-1")

classifier = BaseAgent(
    config={
        "llm_provider": "ollama", "model": DEFAULT_CHAT_MODEL,
        "base_url": OLLAMA_BASE_URL, "use_async_llm": True, "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "structured_output_mode": "explicit",
    },
    signature=ReviewSentiment(),
)

sample = load_sst2().head(50)                 # columns: text, label, label_id
correct, invalid = 0, 0
for text, gold in zip(sample["text"], sample["label"]):
    out = await classifier.run_async(text=text)
    predicted = str(out.get("sentiment", "")).strip().lower()
    if predicted not in {"positive", "negative"}:
        invalid += 1                           # missing or off-schema label
    elif predicted == gold:
        correct += 1
print(f"accuracy {correct}/50, invalid outputs {invalid}")
```

SST-2 sentences are fragments ("soulful and", "be fruitful"), so some are genuinely ambiguous; read the misclassified rows before blaming the prompt.

**Drill 4.** Implement token tracking: process 100 classification requests and report total tokens, average tokens per request, and a breakdown by prompting technique. (Ollama is free, so token count — not dollars — is the honest workload signal.)

**Solution:**

```python
import polars as pl
from shared.mlfp06._ollama_bootstrap import make_delegate, run_delegate_text
from shared.mlfp06.ex_1 import load_sst2

delegate = make_delegate(temperature=0.0)
prompts = {
    "zero_shot": "Classify as positive or negative. Reply with one word.\n\n{t}",
    "cot": "Classify as positive or negative. Let's think step by step, "
           "then give the label on the last line.\n\n{t}",
}

rows = []
for text in load_sst2().head(50)["text"]:          # 50 texts x 2 prompts = 100 calls
    for name, template in prompts.items():
        _reply, usage, secs = await run_delegate_text(delegate, template.format(t=text))
        rows.append({"technique": name, "tokens": usage["total_tokens"], "seconds": secs})

log = pl.DataFrame(rows)
print(f"total tokens {log['tokens'].sum()}, mean per request {log['tokens'].mean():.0f}")
print(log.group_by("technique").agg(
    pl.col("tokens").sum().alias("total_tokens"),
    pl.col("tokens").mean().alias("mean_tokens"),
    pl.col("seconds").mean().alias("mean_seconds"),
))
```

**Drill 5.** Explain the relationship between temperature, top-p (nucleus sampling), and output quality. Run the same prompt at temperatures 0, 0.3, 0.7, and 1.0 ten times each. Measure the variance of the outputs.

**Solution:** temperature rescales the logits before the softmax (Mathematical Foundations below); top-p then keeps only the smallest set of tokens whose cumulative probability reaches $p$ and samples from that set. Both trade diversity against reliability. Measure variance as the number of distinct answers out of ten (or the spread of a parsed number). What you should expect: at temperature 0 the ten outputs are identical or nearly so (greedy decoding; tiny differences can come from floating-point non-determinism on the GPU). As temperature rises, phrasing diversifies first; at 1.0 the conclusions themselves start to vary and occasional errors appear. For factual and classification tasks use temperature 0–0.3; for brainstorming and creative text, 0.7–1.0. Your own counts are the answer to this drill — report them, not this paragraph.

## Cross-References

- **Lesson 5.4** derived the transformer architecture. LLMs are transformers scaled to billions of parameters.
- **Lesson 6.2** will fine-tune LLMs for domain-specific tasks.
- **Lesson 6.3** will align LLMs with human preferences using DPO.
- **Lesson 6.4** will ground LLMs in facts using RAG.

## Reflection

You should now be able to:

- Explain how LLMs are pre-trained and aligned.
- Apply six prompt engineering techniques and know when each is appropriate.
- Use a Kaizen Delegate for streaming text and a BaseAgent with a Signature for structured, type-safe output.
- Track token usage and latency per technique, and convert tokens into an estimated hosted cost when you need one.

---

# Lesson 6.2: LLM Fine-Tuning — LoRA, Adapters, and the Technique Landscape

## Why This Matters

Prompt engineering is limited. No matter how clever your prompt, the model's knowledge is fixed at pre-training. If you need a model that understands Singapore legal terminology, medical Mandarin-English code-switching, or your company's internal product taxonomy, you need to fine-tune. But full fine-tuning of a 7-billion-parameter model with the Adam optimiser needs roughly 16 bytes per parameter (FP16 weights and gradients, plus FP32 master weights and two Adam moments) — about 112 GB of GPU memory before activations — which is impractical for most teams. Parameter-efficient fine-tuning (PEFT) methods like LoRA and adapters achieve comparable results by modifying only a tiny fraction of the parameters.

## Core Concepts

### THEORY: LoRA — Low-Rank Adaptation

LoRA (Low-Rank Adaptation of Large Language Models) is based on the observation that the weight updates during fine-tuning have low intrinsic rank. Instead of updating the full weight matrix $\mathbf{W} \in \mathbb{R}^{d \times k}$, LoRA decomposes the update into two low-rank matrices:

$$\mathbf{W}' = \mathbf{W}_0 + \frac{\alpha}{r}\mathbf{B}\mathbf{A}$$

where $\mathbf{W}_0$ is the frozen pre-trained weight, $\mathbf{B} \in \mathbb{R}^{d \times r}$ and $\mathbf{A} \in \mathbb{R}^{r \times k}$ are the trainable low-rank matrices, $r \ll \min(d, k)$ is the rank, and $\alpha / r$ is a fixed scaling factor (so changing $r$ does not change the update's magnitude). One of the two matrices starts at zero, so at initialisation $\mathbf{W}' = \mathbf{W}_0$ exactly: training starts from the pre-trained model, not from a perturbed one.

The connection to Module 4: LoRA IS low-rank matrix factorisation (Lesson 4.3, SVD). The pre-trained weights capture the bulk of the model's knowledge; the low-rank update captures the task-specific adaptation. Typical ranks are $r = 4, 8, 16$ — meaning you train a fraction of a percent of the total parameters.

The from-scratch implementation below is the one Exercise 6.2 (`ex_2/01_lora_from_scratch.py`) builds. Because the code multiplies a row vector `x` on the left, `lora_A` has shape (d_in, r) and `lora_B` has shape (r, d_out); the update to the weight is their product, which is rank $r$ at most.

```python
import math
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """The trainable low-rank path: (x @ A @ B) * (alpha / r)."""

    def __init__(self, in_features, out_features, rank=8, alpha=16.0):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # random A
        nn.init.zeros_(self.lora_B)                            # B = 0, so A @ B = 0

    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class LoRALinear(nn.Module):
    """A pre-trained nn.Linear, frozen, plus a trainable LoRA path."""

    def __init__(self, pretrained_linear: nn.Linear, rank=8, alpha=16.0):
        super().__init__()
        self.linear = pretrained_linear
        for p in self.linear.parameters():     # freeze weight AND bias
            p.requires_grad = False
        self.lora = LoRALayer(
            pretrained_linear.in_features, pretrained_linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)
```

The wrapper takes the _existing_ pre-trained layer and freezes it. A common mistake is to build a fresh `nn.Linear` inside the LoRA module, freeze that, and then swap the pre-trained layer back in — the freeze then applies to a layer nobody uses, and every pre-trained weight is silently trained.

### THEORY: Adapter layers

Adapter layers (Houlsby et al., 2019) insert small bottleneck modules inside each transformer layer:

$$\mathbf{h}' = \mathbf{h} + f(\text{LN}(\mathbf{h}) \mathbf{W}_{\text{down}}) \mathbf{W}_{\text{up}}$$

where $\mathbf{W}_{\text{down}} \in \mathbb{R}^{d \times m}$ projects to a lower dimension $m$, $f$ is an activation function, and $\mathbf{W}_{\text{up}} \in \mathbb{R}^{m \times d}$ projects back. Only the adapter's weights are trained; the rest of the model is frozen. Initialising $\mathbf{W}_{\text{up}}$ to zero makes the adapter start as the identity — the same trick as LoRA's $\mathbf{B} = 0$.

```python
class AdapterLayer(nn.Module):
    """x -> LayerNorm -> down -> GELU -> up -> + x (as in ex_2/02)."""

    def __init__(self, d_model, bottleneck_dim=64):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        nn.init.zeros_(self.up_proj.weight)   # start as the identity
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        h = self.up_proj(self.activation(self.down_proj(self.layer_norm(x))))
        return x + h
```

### FOUNDATIONS: LoRA vs Adapter comparison

| Dimension           | LoRA                                     | Adapter                                    |
| ------------------- | ---------------------------------------- | ------------------------------------------ |
| Parameter update    | Low-rank matrices A, B beside a weight   | Bottleneck FC layers inside each block     |
| Where applied       | Weight matrices (Q, K, V, O, FFN)        | After the attention and/or FFN sub-layers  |
| Merge at inference  | Yes (add $\frac{\alpha}{r}BA$ to $W_0$)  | No (module remains)                        |
| Inference overhead  | Zero after merging                       | Small (extra layers on every forward pass) |
| Typical parameter % | 0.1–1%                                   | 0.5–5%                                     |
| Implementation      | Wrap individual Linear layers            | Wrap or insert whole sub-layers            |
| Flexibility         | Swap adapters per task; merge or unmerge | Stack or swap per task; never merged       |

On BERT-base (110M parameters) the configurations used in this lesson's worked example and drills measure as follows (counted with the code below; each includes the 1,538-parameter classification head):

| Configuration                          | Trainable parameters | Share of total |
| -------------------------------------- | -------------------- | -------------- |
| LoRA r = 8 on Q and V (12 layers)      | 296,450              | 0.27%          |
| LoRA r = 8 on Q, K and V               | 443,906              | 0.40%          |
| Adapter, bottleneck 64, after each FFN | 1,209,602            | 1.09%          |
| Full fine-tuning                       | 109,483,778          | 100%           |

### FOUNDATIONS: The fine-tuning landscape (survey)

Exercise 6.2 (`ex_2/03_finetuning_landscape.py`) surveys ten techniques and builds a decision tree over them.

| Technique                      | Key idea                                                                         | Parameters trained   |
| ------------------------------ | -------------------------------------------------------------------------------- | -------------------- |
| Task-specific full fine-tuning | Update all weights, with an LR schedule, gradient clipping, mixed precision      | 100%                 |
| LoRA                           | Low-rank update beside frozen weight matrices                                    | 0.1–1%               |
| Adapters                       | Bottleneck modules inside each block                                             | 0.5–5%               |
| Prefix tuning                  | Learnable key/value vectors prepended at every attention layer                   | < 1%                 |
| Prompt tuning                  | Learnable soft-prompt embeddings prepended to the input only                     | < 0.1%               |
| LLRD                           | Layer-wise learning-rate decay: lower LR for earlier layers                      | 100% (different LRs) |
| Progressive freezing           | Unfreeze layers top-down over the course of training                             | Varies               |
| Knowledge distillation         | Train a small student on a large teacher's soft labels                           | 100% of the student  |
| Differential privacy (DP-SGD)  | Clip each example's gradient and add Gaussian noise; privacy budget ε            | Any of the above     |
| Elastic weight consolidation   | Penalise moving weights the Fisher information marks as important to an old task | 100% (regularised)   |

Two rows need a sentence more. **DP-SGD** bounds how much any single training example can influence the weights, so the model cannot memorise (and later leak) one patient record or one customer email; the price is lower accuracy for a stronger privacy guarantee. **EWC** fights catastrophic forgetting: the loss gains a term $\sum_i \frac{\lambda}{2} F_i (\theta_i - \theta^_\_i)^2$ that anchors each weight to its old value $\theta^__i$ in proportion to its Fisher information $F_i$.

### ADVANCED: Model merging

After fine-tuning several adapters for different tasks you can merge them into one model without further training. Write each task's change as a **task vector** $\tau_t = \theta_t - \theta_0$ (for LoRA, $\tau_t = \frac{\alpha}{r} B_t A_t$ per wrapped layer).

- **Task arithmetic:** $\theta = \theta_0 + \lambda \sum_t \tau_t$. Adding a vector adds a skill; subtracting one removes it.
- **TIES (Trim, Elect Sign, Merge):** for each task, keep only the top-$k$% of $\tau_t$ entries by magnitude (trim); for each parameter, elect the sign of the _sum_ of the trimmed values (sign election); then average only the trimmed values that agree with the elected sign (disjoint merge). This stops two tasks' opposite-signed updates from cancelling to noise.
- **DARE (Drop And REscale):** drop each entry of $\tau_t$ with probability $p$ and multiply the survivors by $1/(1-p)$, which keeps the expected update unchanged; it is usually applied before TIES or task arithmetic to reduce interference.
- **SLERP:** spherical linear interpolation between two weight vectors, which preserves their norm better than a straight average. It merges exactly two models.

### FOUNDATIONS: Quantisation

Reduce model precision to fit larger models on smaller hardware. Memory scales with bytes per weight: FP32 uses 4 bytes, FP16/BF16 2, INT8 1, and 4-bit formats 0.5. So INT8 halves the memory of FP16 (and quarters FP32); 4-bit quarters FP16. A 7B model needs about 14 GB of weights in FP16, about 7 GB in INT8 and about 3.5 GB in 4-bit.

- **GPTQ:** post-training quantisation that corrects each layer's rounding error using approximate second-order (Hessian) information.
- **AWQ:** activation-aware quantisation that protects the small fraction of weight channels that matter most for the activations.
- **GGUF:** the llama.cpp file format with mixed-precision quantisation levels (Q2_K … Q8_0), built for CPU and laptop inference — and what Ollama runs.
- **bitsandbytes:** the PyTorch library that loads a model in 8-bit or 4-bit (NF4) on the fly; it is what QLoRA uses during training.
- **QLoRA:** quantise the frozen base model to 4-bit NF4, then train FP16/BF16 LoRA adapters on top. Dettmers et al. (2023) fine-tuned a 65B model on a single 48 GB GPU this way.

Quantise when the deployment hardware is the constraint (a CPU server, a laptop, a single small GPU), and measure the quality drop on your own evaluation set — it is usually small at 8-bit and grows at 4-bit and below.

## Mathematical Foundations

### THEORY: Why low rank works

During fine-tuning, the weight update $\Delta \mathbf{W} = \mathbf{W}' - \mathbf{W}_0$ has been empirically observed to have a low intrinsic rank. Aghajanyan et al. (2021) showed that pre-trained models have a low "intrinsic dimensionality" — only a small number of dimensions in parameter space need to change to adapt to a new task. LoRA exploits this by constraining the update to rank $r$, which acts as a regulariser and reduces the number of trainable parameters from $d \times k$ to $(d + k) \times r$.

For a weight matrix $\mathbf{W} \in \mathbb{R}^{768 \times 768}$ with $r = 8$: full fine-tuning trains $589,824$ parameters; LoRA trains $(768 + 768) \times 8 = 12,288$ parameters — a $48\times$ reduction.

## The Kailash Engine: kailash-align

The from-scratch code teaches the mechanism; for real LLM fine-tuning the course uses `kailash-align`, which wraps the TRL trainers behind one typed config. This is the pattern of `ex_2/06_sft_alignment_pipeline.py`. Training and registration are both `async`.

```python
import os
import polars as pl
from datasets import Dataset
from kailash_align import (AdapterRegistry, AdapterSignature, AlignmentConfig,
                           AlignmentPipeline, LoRAConfig, SFTConfig)
from shared.mlfp06.ex_2 import load_imdb_sft

# Base model is a HuggingFace repo id from SFT_BASE_MODEL
# (course default Qwen/Qwen2.5-0.5B-Instruct) — not an Ollama tag.
config = AlignmentConfig(
    method="sft",
    base_model_id=os.environ.get("SFT_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
    lora=LoRAConfig(rank=8, alpha=16, target_modules=("q_proj", "v_proj")),
    sft=SFTConfig(num_train_epochs=3, learning_rate=2e-4),
)

# SFT trains on the "text" column. Build it as instruction + response;
# the raw review alone would teach the model IMDB prose, not the task.
_full, train_df, _eval = load_imdb_sft()
sft_frame = train_df.select(
    (pl.col("instruction") + "\n\n" + pl.col("response")).alias("text")
)
train_ds = Dataset.from_dict(sft_frame.to_dict(as_series=False))

pipeline = AlignmentPipeline(config)
result = await pipeline.train(train_ds, adapter_name="imdb-sentiment-lora")
print(result.training_metrics.get("train_loss"), result.adapter_path)

# Store a versioned adapter for later use (Lesson 6.8 loads it again)
version = await AdapterRegistry().register_adapter(
    name="imdb-sentiment-lora",
    adapter_path=result.adapter_path,
    signature=AdapterSignature(base_model_id=config.base_model_id,
                               adapter_type="lora", training_method="sft"),
)
```

`AlignmentResult` carries `adapter_name`, `adapter_path`, `adapter_version`, `training_metrics` (the raw trainer metrics dict), `experiment_dir` and `method`. It does not evaluate the model — accuracy, win rates and benchmark scores are your own evaluation step.

## Worked Example: LoRA Fine-Tuning for Sentiment Classification

Wrap BERT's query and value projections with `LoRALinear`, on the IMDB reviews Exercise 6.2 loads. The order matters: freeze the whole model first, then wrap, then unfreeze only the new classification head.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from shared.mlfp06.ex_2 import load_imdb_sft

def build_lora_bert(rank=8, targets=("query", "value")):
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    for p in model.parameters():          # 1. freeze every pre-trained weight
        p.requires_grad = False
    for layer in model.bert.encoder.layer:  # 2. wrap the chosen projections
        attn = layer.attention.self
        for name in targets:
            setattr(attn, name, LoRALinear(getattr(attn, name), rank=rank, alpha=2 * rank))
    for p in model.classifier.parameters():  # 3. the new head must learn too
        p.requires_grad = True
    return model

model = build_lora_bert()
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total:,}, Trainable: {trainable:,} ({100 * trainable / total:.2f}%)")
# Total: 109,778,690, Trainable: 296,450 (0.27%)

# Data: IMDB reviews with positive/negative labels
_full, train_df, eval_df = load_imdb_sft()      # 1,800 train / 200 eval rows
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def batches(df, batch_size=16):
    for start in range(0, df.height, batch_size):
        chunk = df.slice(start, batch_size)
        enc = tokenizer(chunk["text"].to_list(), truncation=True, max_length=256,
                        padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([int(lbl == "positive") for lbl in chunk["label"]])
        yield enc

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=2e-4
)
model.train()
for batch in batches(train_df.sample(fraction=1.0, shuffle=True, seed=0)):
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

The printed counts are exact for `bert-base-uncased`: 294,912 LoRA parameters (12 layers × 2 projections × (768 × 8 + 8 × 768)) plus the 1,538-parameter classifier. Had the base model not been frozen first, "trainable" would read 100%. One epoch over the 1,800 training reviews takes minutes on a GPU and much longer on a laptop CPU; reduce `max_length` or use a subset if you are on CPU.

## Try It Yourself

**Drill 1.** Apply `LoRALinear` to the Q, K and V projections of a pre-trained BERT model. Fine-tune on IMDB sentiment classification for 3 epochs, report test accuracy, and compare with full fine-tuning.

**Solution:** reuse `build_lora_bert` and `batches` from the worked example.

```python
import time
import torch

def train_and_eval(model, train_df, eval_df, epochs=3, lr=2e-4):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for batch in batches(train_df.sample(fraction=1.0, shuffle=True, seed=epoch)):
            model(**batch).loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    train_seconds = time.perf_counter() - start

    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in batches(eval_df, batch_size=32):
            labels = batch.pop("labels")
            preds = model(**batch).logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
    return correct / eval_df.height, train_seconds

lora_qkv = build_lora_bert(rank=8, targets=("query", "key", "value"))
lora_acc, lora_secs = train_and_eval(lora_qkv, train_df, eval_df)

full_ft = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
full_acc, full_secs = train_and_eval(full_ft, train_df, eval_df, lr=2e-5)  # full FT needs a ~10x lower LR

print(f"LoRA QKV  acc={lora_acc:.3f}  {lora_secs:.0f}s  trainable=443,906")
print(f"Full FT   acc={full_acc:.3f}  {full_secs:.0f}s  trainable=109,483,778")
```

Expect LoRA to land close to full fine-tuning on a 1,800-review training set while training 0.4% of the weights and storing a checkpoint of under 2 MB instead of 440 MB. The time saving per step is smaller than the parameter saving, because the forward and backward passes still run through the whole frozen network.

**Drill 2.** Insert adapter layers into BERT. Compare adapter fine-tuning with LoRA fine-tuning on the same task: accuracy, parameter count, training time.

**Solution:** put an `AdapterLayer` after each layer's feed-forward output block (`layer.output`), which returns a plain hidden-state tensor.

```python
class WithAdapter(nn.Module):
    """Run a frozen sub-layer, then the adapter on its output."""

    def __init__(self, block, d_model=768, bottleneck_dim=64):
        super().__init__()
        self.block = block
        self.adapter = AdapterLayer(d_model, bottleneck_dim)

    def forward(self, *args, **kwargs):
        return self.adapter(self.block(*args, **kwargs))

def build_adapter_bert(bottleneck=64):
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    for p in model.parameters():
        p.requires_grad = False
    for layer in model.bert.encoder.layer:
        layer.output = WithAdapter(layer.output, 768, bottleneck)
    for p in model.classifier.parameters():
        p.requires_grad = True
    return model

adapter_model = build_adapter_bert()
adapter_acc, adapter_secs = train_and_eval(adapter_model, train_df, eval_df, lr=1e-3)
lora_model = build_lora_bert(rank=8)
lora_acc, lora_secs = train_and_eval(lora_model, train_df, eval_df)
print(f"Adapter  acc={adapter_acc:.3f}  {adapter_secs:.0f}s  trainable=1,209,602 (1.09%)")
print(f"LoRA QV  acc={lora_acc:.3f}  {lora_secs:.0f}s  trainable=296,450 (0.27%)")
```

The parameter counts are exact (measured on `bert-base-uncased`); the accuracies and times are yours to report. The adapter trains about four times as many parameters as LoRA on Q and V, and its extra layers stay in the forward pass at inference, whereas LoRA can be merged away.

**Drill 3.** Vary the LoRA rank (1, 2, 4, 8, 16, 32, 64). Plot accuracy vs rank and training time vs rank. What is the optimal rank for the sentiment task?

**Solution:**

```python
import matplotlib.pyplot as plt
import polars as pl

rows = []
for rank in [1, 2, 4, 8, 16, 32, 64]:
    model = build_lora_bert(rank=rank)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    acc, secs = train_and_eval(model, train_df, eval_df, epochs=1)
    rows.append({"rank": rank, "trainable": trainable, "accuracy": acc, "seconds": secs})
sweep = pl.DataFrame(rows)
print(sweep)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(sweep["rank"], sweep["accuracy"], marker="o")
ax1.set(xscale="log", xlabel="LoRA rank r", ylabel="eval accuracy")
ax2.plot(sweep["rank"], sweep["seconds"], marker="o")
ax2.set(xscale="log", xlabel="LoRA rank r", ylabel="training seconds")
plt.tight_layout()
plt.show()
```

The trainable count is exactly $12 \times 2 \times 1536r + 1538$: 38,402 at r = 1, 296,450 at r = 8 and 2,360,834 at r = 64. Binary sentiment is a low-rank task, so accuracy typically plateaus at a small rank; pick the smallest rank on the plateau. Training time barely moves with rank, because the frozen network dominates the compute.

**Drill 4.** Train two LoRA adapters on BERT: one for sentiment and one for a second binary task of your choice. Merge them by task arithmetic (add both weight deltas to the base model). Does the merged model perform both tasks?

**Solution:** each wrapped layer's delta is $\frac{\alpha}{r}(AB)^\top$ in PyTorch's (out, in) weight layout. Fold both deltas into one copy of the base weights:

```python
import copy

def lora_deltas(model):
    """{module path: delta W} for every LoRALinear in the model."""
    return {
        name: (module.lora.lora_A @ module.lora.lora_B * module.lora.scaling).T.detach()
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    }

def merge_task_arithmetic(base, models, weights):
    merged = copy.deepcopy(base)
    all_deltas = [lora_deltas(m) for m in models]
    for name, module in merged.named_modules():
        if isinstance(module, nn.Linear) and name.endswith(("query", "value")):
            key = name                      # same module path as the LoRALinear
            for lam, deltas in zip(weights, all_deltas):
                module.weight.data += lam * deltas[key]
    return merged

base = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
merged = merge_task_arithmetic(base, [sentiment_model, second_task_model], [1.0, 1.0])
```

`sentiment_model` and `second_task_model` are two `build_lora_bert()` models you trained on their own tasks. The merged encoder carries both skills, but each task still needs its own classification head: copy `sentiment_model.classifier` onto the merged model to test sentiment, and the other head for the second task. Expect some loss on each task relative to its own adapter; interference grows when the two deltas push the same weights in opposite directions, which is exactly what TIES's sign election addresses.

**Drill 5.** Explain in five sentences how LoRA relates to SVD from Module 4, Lesson 4.3. What is the "low-rank structure" that LoRA exploits? Why does constraining the rank act as a regulariser?

**Solution:** SVD decomposes a matrix into $\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$, where keeping only the top $r$ singular values gives the best rank-$r$ approximation. LoRA's $\mathbf{B}\mathbf{A}$ decomposition constrains the weight update to rank $r$ — it learns a rank-$r$ update directly rather than truncating a full one. The "low-rank structure" is the observation that fine-tuning changes lie in a low-dimensional subspace of the full parameter space. Constraining the rank acts as a regulariser because it limits the model's capacity to overfit to the fine-tuning data — similar to how PCA with fewer components discards noise. This is why LoRA with $r = 8$ often matches full fine-tuning on tasks with moderate training data.

## Cross-References

- **Lesson 4.3** derived SVD and PCA. LoRA is low-rank factorisation applied to weight updates.
- **Lesson 5.7** introduced transfer learning with frozen-backbone fine-tuning. LoRA and adapters are parameter-efficient alternatives.
- **Lesson 6.3** will use aligned models produced by fine-tuning.

## Reflection

You should now be able to:

- Implement LoRA from scratch and explain the low-rank mathematics.
- Implement adapter layers from scratch.
- Compare all major fine-tuning techniques and select the right one.
- Merge multiple LoRA adapters using task arithmetic, TIES or DARE.
- Choose a quantisation format for the deployment hardware.
- Fine-tune and register an adapter with kailash-align's `AlignmentPipeline` and `AdapterRegistry`.

---

# Lesson 6.3: Preference Alignment — DPO and GRPO

## Why This Matters

A fine-tuned model produces domain-specific outputs, but it may still generate unhelpful, verbose, or harmful responses. Preference alignment trains the model to prefer responses that people prefer. RLHF (Reinforcement Learning from Human Feedback) was the original approach: train a reward model on human preference rankings, then use PPO to optimise the LLM against that reward model. But RLHF is complex — it requires training and maintaining a separate reward model, PPO needs a value network as well, and the whole loop is notoriously sensitive to hyperparameters.

DPO (Direct Preference Optimization) achieves the same goal by bypassing the reward model entirely. It derives a closed-form loss function directly from the preference data. GRPO (Group Relative Policy Optimization), introduced in DeepSeekMath (Shao et al., 2024) and later used to train DeepSeek-R1 (2025), takes a different approach: sample several completions per prompt, score each with a verifier, normalise the scores within the group, and optimise with a PPO-style clipped objective — but without a value network. Both are simpler than full RLHF.

## Core Concepts

### THEORY: From RLHF to DPO — the derivation

RLHF maximises the expected reward while staying close to the reference policy:

$$\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot \mid x)}\left[r(x, y)\right] - \beta \, \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

The optimal solution to this KL-regularised optimisation is:

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{r(x, y)}{\beta}\right)$$

Solving for the reward:

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

### THEORY: The Bradley-Terry preference model

The Bradley-Terry model defines the probability that response $y_w$ is preferred over $y_l$ given prompt $x$:

$$P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$$

where $\sigma$ is the sigmoid function. Substituting the reward expression (with the policy being trained, $\pi_\theta$, in place of $\pi^*$) and noting that $\beta \log Z(x)$ cancels:

$$P(y_w \succ y_l \mid x) = \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)$$

### THEORY: The DPO loss

The DPO loss maximises the log-likelihood of the observed preferences:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$

This is a standard binary classification loss. The model learns to assign higher probability to preferred responses relative to the reference policy. No reward model, no PPO, no RL training loop — just supervised learning on preference pairs.

**What $\beta$ does.** $\beta$ is the KL-penalty coefficient from the RLHF objective, and the optimal policy is $\pi^* \propto \pi_{\text{ref}} \exp(r/\beta)$. A **large** $\beta$ makes the exponent small, so the aligned model stays **close to the reference** (conservative alignment, little risk of degrading general ability). A **small** $\beta$ makes the exponent large, so the model can **drift far** from the reference to chase the preference signal (aggressive alignment, with a higher risk of over-optimising, verbosity, and losing capabilities). Seen from the loss: the implied reward is $\beta \log(\pi_\theta / \pi_{\text{ref}})$, so to express the same preference margin a smaller $\beta$ requires a larger log-ratio — a bigger move away from the reference. Typical values are 0.1–0.5. If a benchmark such as MMLU drops after DPO, the model has drifted too far: **raise** $\beta$ (or train for fewer steps). Drill 2 measures this directly.

### THEORY: GRPO — Group Relative Policy Optimization

GRPO (Shao et al., 2024, DeepSeekMath) works as follows:

1. For each prompt $x$, sample a group of $G$ completions $y_1, \dots, y_G$ from the current policy.
2. Score each completion with a reward $r_i$ — typically a verifier (is the maths answer correct? do the unit tests pass?), not a learned reward model.
3. Normalise within the group: $\hat{A}_i = \dfrac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G) + \epsilon}$.
4. Maximise a PPO-style clipped objective with a KL penalty to the reference policy:

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G} \min\Big(\rho_i \hat{A}_i,\; \text{clip}(\rho_i, 1-\varepsilon, 1+\varepsilon)\,\hat{A}_i\Big) - \beta_{\text{KL}}\, \text{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})\right], \qquad \rho_i = \frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)}$$

(in the paper the ratio and the average are taken per token). The group mean replaces PPO's learned value network as the baseline, which is what makes GRPO cheap. Dividing by the group standard deviation makes the update independent of the reward's scale: rewards of {0, 1} and {0, 10} give identical advantages. Subtracting the mean alone would not — the advantages would be ten times larger. A group in which every completion gets the same reward has zero advantage everywhere and contributes no learning signal.

```python
import torch
from shared.mlfp06.ex_3 import grpo_advantages   # the helper Exercise 6.3 uses

rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0],    # 2 of 4 samples correct
                        [0.0, 0.0, 0.0, 1.0],    # 1 of 4 correct
                        [1.0, 1.0, 1.0, 1.0]])   # all correct: no signal
print(grpo_advantages(rewards))
# tensor([[ 0.8660, -0.8660, -0.8660,  0.8660],
#         [-0.5000, -0.5000, -0.5000,  1.5000],
#         [ 0.0000,  0.0000,  0.0000,  0.0000]])
print(torch.allclose(grpo_advantages(10 * rewards), grpo_advantages(rewards)))  # True
```

The rare correct answer in the second group gets the largest push (1.5), because beating your siblings when most of them fail is the most informative event. GRPO shares DPO's advantage of not needing a learned reward model but keeps the online policy-gradient framework, so it suits tasks where the scoring function is cheap and reliable.

### FOUNDATIONS: LLM-as-Judge

Use one LLM to evaluate another's outputs. The judge LLM rates responses on criteria like helpfulness, factual accuracy, and harmlessness. Known biases:

- **Position bias:** the judge tends to prefer the response shown in a particular position (often the first).
- **Verbosity bias:** the judge prefers longer responses.
- **Self-enhancement bias:** the judge prefers responses similar to its own style.

Mitigations: judge every pair in both orders and only count a win when the two verdicts agree (or average them); control for length (compare length-matched pairs, or penalise length explicitly); use several judge models. Report how often the judge failed to give a parsable verdict — a parse failure is not a tie.

### FOUNDATIONS: Evaluation benchmarks

| Benchmark | Tests                             | Format                  | Metric                |
| --------- | --------------------------------- | ----------------------- | --------------------- |
| MMLU      | Multi-task language understanding | Multiple choice         | Accuracy              |
| HellaSwag | Commonsense reasoning             | Sentence completion     | Accuracy (normalised) |
| HumanEval | Code generation                   | Function implementation | pass@1                |
| MT-Bench  | Multi-turn conversation           | Open-ended + LLM judge  | Judge score 1–10      |

**lm-eval-harness** (EleutherAI) is the standard tool for running the first three with one command, for example `lm_eval --model hf --model_args pretrained=<model> --tasks hellaswag --limit 50`, and running it before and after alignment shows whether DPO cost general capability. Each task reports its own metric key (`acc`/`acc_norm` for multiple choice, `pass@1` for code). MT-Bench is not an lm-eval task; it is run with its own judge-based harness.

## The Kailash Engine: kailash-align (DPO)

This is the pattern of `ex_3/03_dpo_training.py`. Exercise 6.3 trains on **UltraFeedback Binarized** — about 2,000 (prompt, chosen, rejected) triples whose preference labels come from GPT-4 ratings (AI feedback), not from human annotators.

```python
import os
from datasets import Dataset
from kailash_align import AlignmentConfig, AlignmentPipeline, DPOConfig, LoRAConfig
from shared.mlfp06.ex_3 import load_ultrafeedback, split_preferences

config = AlignmentConfig(
    method="dpo",
    base_model_id=os.environ.get("SFT_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
    lora=LoRAConfig(rank=16, alpha=32, target_modules=("q_proj", "v_proj")),
    dpo=DPOConfig(beta=0.1, learning_rate=5e-5, num_train_epochs=2),
)

prefs = load_ultrafeedback()                       # columns: prompt, chosen, rejected
train_pref, eval_pref = split_preferences(prefs)
pref_ds = Dataset.from_dict(
    train_pref.select(["prompt", "chosen", "rejected"]).to_dict(as_series=False)
)

result = await AlignmentPipeline(config).train(
    None, adapter_name="ultrafeedback_dpo_v1", preference_dataset=pref_ds
)
metrics = result.training_metrics                  # the raw TRL metrics dict
print(metrics.get("train_loss"), result.adapter_path)
# Win rate and benchmark deltas are YOUR evaluation step — train() does not compute them.
```

The learning rate is much lower than for SFT: about 5e-5 with LoRA and around 1e-6 for full-parameter DPO, because DPO pushes on log-probability ratios and diverges easily.

## Worked Example: DPO Loss on Sequence Log-Probabilities

DPO needs one number per (prompt, response): the log-probability of the response tokens given the prompt, summed over tokens. Two details are easy to get wrong. The logits at position $t$ predict token $t+1$, so logits and targets must be shifted by one; and only the response tokens should be scored, not the prompt or the padding.

```python
import torch
import torch.nn.functional as F

def sequence_logprob(model, input_ids, attention_mask, prompt_lens):
    """Sum of log p(response tokens | prompt) for each row of the batch."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
    targets = input_ids[:, 1:]                      # logits[t] predicts token t+1
    token_logp = torch.log_softmax(logits, dim=-1).gather(
        -1, targets.unsqueeze(-1)).squeeze(-1)
    positions = torch.arange(targets.shape[1], device=input_ids.device)
    is_response = positions.unsqueeze(0) >= (prompt_lens.unsqueeze(1) - 1)
    mask = attention_mask[:, 1:] * is_response      # drop prompt and padding
    return (token_logp * mask).sum(dim=-1)

def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    chosen_rewards = beta * (policy_chosen - ref_chosen)        # implied rewards
    rejected_rewards = beta * (policy_rejected - ref_rejected)
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

def dpo_step(policy, reference, batch, optimizer, beta=0.1):
    pc = sequence_logprob(policy, batch["chosen_ids"], batch["chosen_mask"], batch["prompt_lens"])
    pr = sequence_logprob(policy, batch["rejected_ids"], batch["rejected_mask"], batch["prompt_lens"])
    with torch.no_grad():                            # the reference is frozen
        rc = sequence_logprob(reference, batch["chosen_ids"], batch["chosen_mask"], batch["prompt_lens"])
        rr = sequence_logprob(reference, batch["rejected_ids"], batch["rejected_mask"], batch["prompt_lens"])
    loss = dpo_loss(pc, pr, rc, rr, beta)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

At the first step the policy equals the reference, every log-ratio is zero, and the loss is exactly $\log 2 \approx 0.6931$ whatever the data. A first-step loss far from 0.693 means the policy and reference are not the same model, or the masking is wrong.

## Try It Yourself

The first two drills use a small synthetic setting that runs on a CPU in seconds, so you can watch DPO and $\beta$ at work before spending GPU time on `ex_3/03`. Each of 50 "prompts" has 4 candidate responses with a hidden true quality; the policy is a table of logits; the preference data are 3,000 noisy Bradley-Terry judgements.

**Drill 1.** Implement the DPO loss from scratch. Verify it decreases during training on a small preference dataset.

**Solution:**

```python
import itertools
import torch
import torch.nn.functional as F
from shared.mlfp06.ex_3 import dpo_loss

torch.manual_seed(0)
N_PROMPTS, K = 50, 4                       # 50 prompts, 4 candidate responses each
quality = torch.randn(N_PROMPTS, K)         # hidden "true" quality (synthetic)
ref_logits = torch.randn(N_PROMPTS, K)      # the frozen reference policy

# 10 noisy judgements per response pair: P(a beats b) = sigmoid(quality_a - quality_b)
rows = []
for x in range(N_PROMPTS):
    for a, b in itertools.combinations(range(K), 2):
        for _ in range(10):
            a_wins = torch.rand(()) < torch.sigmoid(quality[x, a] - quality[x, b])
            rows.append((x, a, b) if a_wins else (x, b, a))
prompt, chosen, rejected = (torch.tensor(c) for c in zip(*rows))

def train_dpo(beta, steps=2000, lr=0.1):
    logits = ref_logits.clone().requires_grad_(True)     # start AT the reference
    opt = torch.optim.Adam([logits], lr=lr)
    ref_logp = F.log_softmax(ref_logits, dim=-1)
    losses = []
    for _ in range(steps):
        logp = F.log_softmax(logits, dim=-1)
        loss = dpo_loss(logp[prompt, chosen], logp[prompt, rejected],
                        ref_logp[prompt, chosen], ref_logp[prompt, rejected], beta=beta)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    with torch.no_grad():
        logp = F.log_softmax(logits, dim=-1)
        kl = (logp.exp() * (logp - ref_logp)).sum(-1).mean().item()
        ratio = logp - ref_logp                  # the implied reward / beta
        pairs = list(itertools.combinations(range(K), 2))
        agree = sum(((ratio[:, a] - ratio[:, b]) * (quality[:, a] - quality[:, b]) > 0)
                    .sum().item() for a, b in pairs) / (N_PROMPTS * len(pairs))
    return losses, kl, agree

losses, kl, agree = train_dpo(beta=0.1)
print([round(losses[i], 4) for i in (0, 99, 499, 999, 1999)])
# [0.6931, 0.5494, 0.5268, 0.5263, 0.5263]
```

The loss starts at exactly $\log 2$ (policy = reference) and falls to a plateau of about 0.526. It does not reach zero because the judgements are noisy — 28% of the judgements prefer the lower-quality response, as with real annotators — so no policy can explain every label.

**Drill 2.** Vary $\beta$ (0.01, 0.05, 0.1, 0.5, 1.0). How does $\beta$ affect how far the aligned model moves from the reference, and does it change which responses it prefers?

**Solution:** reuse `train_dpo` from Drill 1.

```python
for beta in [0.01, 0.05, 0.1, 0.5, 1.0]:
    _, kl, agree = train_dpo(beta)
    print(f"beta={beta:<5} KL(pi||ref)={kl:.3f}  ranking agreement={agree:.2f}")
# beta=0.01  KL(pi||ref)=1.779  ranking agreement=0.88
# beta=0.05  KL(pi||ref)=1.666  ranking agreement=0.89
# beta=0.1   KL(pi||ref)=1.493  ranking agreement=0.88
# beta=0.5   KL(pi||ref)=0.673  ranking agreement=0.89
# beta=1.0   KL(pi||ref)=0.265  ranking agreement=0.89
```

Larger $\beta$ keeps the policy closer to the reference: the KL divergence falls from 1.78 nats at $\beta = 0.01$ to 0.27 at $\beta = 1.0$. The _ranking_ the policy learns (agreement with the hidden quality, about 0.88–0.89) barely changes — it is limited by the noisy data, not by $\beta$. At small $\beta$ the KL levels off because, with only four responses, the policy cannot move further than piling its probability onto the one it prefers. On a real LLM that drift is where general capabilities are lost, which is why you raise $\beta$ when a benchmark drops after DPO. To measure win rates instead, train `ex_3/03` adapters at two $\beta$ values and compare them with the judge from Drill 3.

**Drill 3.** Implement LLM-as-judge evaluation. Measure position bias by judging the same pair of responses in both orderings. Report the bias magnitude.

**Solution:** judge UltraFeedback pairs in both orders. A judge without position bias picks the same underlying response both times.

```python
import re
from shared.mlfp06._ollama_bootstrap import make_delegate, run_delegate_text
from shared.mlfp06.ex_3 import load_ultrafeedback

judge = make_delegate(temperature=0.0)
TEMPLATE = ("Which response answers the prompt better?\n\nPrompt: {p}\n\n"
            "Response A: {a}\n\nResponse B: {b}\n\nReply with exactly one letter: A or B.")

async def verdict(prompt, first, second):
    text, _usage, _secs = await run_delegate_text(
        judge, TEMPLATE.format(p=prompt, a=first[:1500], b=second[:1500]))
    match = re.search(r"\b([AB])\b", text.strip().upper())
    return match.group(1) if match else None         # None = unparsable

async def position_bias(pairs):
    consistent, flipped, unparsed, first_slot_wins = 0, 0, 0, 0
    for prompt, chosen, rejected in pairs:
        v1 = await verdict(prompt, chosen, rejected)   # chosen shown as A
        v2 = await verdict(prompt, rejected, chosen)   # chosen shown as B
        if v1 is None or v2 is None:
            unparsed += 1
            continue
        first_slot_wins += (v1 == "A") + (v2 == "A")
        if (v1 == "A") == (v2 == "B"):                 # same response both times
            consistent += 1
        else:
            flipped += 1                               # the verdict followed the slot
    judged = consistent + flipped
    print(f"consistent {consistent}/{judged}, flipped {flipped}/{judged}, "
          f"unparsed {unparsed}; slot A chosen {first_slot_wins / max(2 * judged, 1):.0%}")

sample = load_ultrafeedback().head(30)
await position_bias(zip(sample["prompt"], sample["chosen"], sample["rejected"]))
```

The flip rate is the bias magnitude: every flipped pair is a verdict decided by position rather than content, and "slot A chosen" far from 50% shows which position the judge favours. The mitigation is built into the measurement — only count a win when both orders agree.

**Drill 4.** Compare DPO with supervised fine-tuning (SFT) on the same dataset. SFT trains only on the preferred responses; DPO trains on both preferred and dispreferred. Which produces better alignment? Why does contrastive learning (DPO) help?

**Solution:** SFT on the chosen responses only raises the likelihood of good answers; it never tells the model which nearby answers are bad, so the probability of the rejected responses can rise too (they share most of their tokens with the chosen ones). DPO trains on the _difference_ between chosen and rejected log-ratios, so it explicitly pushes probability away from the dispreferred behaviour while the reference term limits drift. In practice the two are combined — SFT first to teach the format and domain, then DPO to sharpen preferences (kailash-align's `method="sft_then_dpo"`) — and which is "better" must be measured on your own judge and benchmark, not assumed.

**Drill 5.** Explain GRPO in three sentences. When would you choose GRPO over DPO? When would you choose DPO over GRPO?

**Solution:** GRPO samples several completions per prompt, scores them with a reward function, normalises each score against its group's mean and standard deviation, and updates the policy with a clipped policy-gradient objective plus a KL penalty to the reference. Choose GRPO when you have a cheap, reliable scoring function (code that passes unit tests, a maths answer you can check) and can afford online sampling during training. Choose DPO when you have a fixed dataset of preference pairs, the quality you want is subjective (tone, helpfulness), and you want a simpler, offline, supervised-style training run.

## Cross-References

- **Lesson 5.8** introduced PPO for reinforcement learning. RLHF uses PPO to align LLMs; DPO bypasses PPO.
- **Lesson 6.1** introduced LLM fundamentals and alignment overview.
- **Lesson 6.7** will use PACT governance to enforce alignment boundaries beyond what DPO can learn.

## Reflection

You should now be able to:

- Derive DPO from the RLHF objective via the Bradley-Terry model.
- Implement DPO training from scratch.
- Explain the role of $\beta$: larger $\beta$ keeps the model closer to the reference; raise it when general ability drops.
- Compute GRPO's std-normalised group advantages and explain why they are scale-invariant.
- Evaluate aligned models using LLM-as-judge (measuring position bias) and standard benchmarks via lm-eval-harness.
- Compare DPO, GRPO, and RLHF and know when each is appropriate.

---

# Lesson 6.4: RAG Systems

## Why This Matters

LLMs have a knowledge cutoff — they do not know about events after their training data ends. They hallucinate — they generate confident, plausible-sounding text that is factually wrong. RAG (Retrieval-Augmented Generation) addresses both problems by grounding LLM responses in retrieved documents. Instead of relying on parametric memory (what the model learned during training), RAG uses non-parametric memory (a searchable document store) to provide relevant context. It reduces hallucination rather than eliminating it: the model can still ignore or misread the context, which is why RAG is always evaluated.

## Core Concepts

### FOUNDATIONS: The RAG pipeline

1. **Chunk** documents into manageable pieces (paragraphs, sentences, or semantic units).
2. **Embed** each chunk using an embedding model, producing dense vectors.
3. **Index** the vectors (and, for sparse retrieval, the terms) for fast search.
4. **Retrieve** relevant chunks given a query (dense, sparse, or hybrid retrieval), optionally **re-rank** them.
5. **Generate** a response using the retrieved chunks as context.

**The course corpus.** Exercise 6.4 uses 1,000 documents sampled from the open `neural-bridge/rag-dataset-12000` dataset, loaded with `shared.mlfp06.ex_4.load_rag_corpus()`. Each row has a `section` id, a `text` (a web passage of 600–7,300 characters, 3,400 on average), and a `question` and `answer` generated from that passage. The same table is therefore both the retrieval corpus and a labelled evaluation set: for question $i$, the relevant document is document $i$. The passages are general web text on many topics, not Singapore policy documents.

### FOUNDATIONS: Chunking strategies

Exercise 6.4 (`ex_4/01`) implements four chunkers:

- **Fixed-size:** every $n$ characters, with an overlap (say 100 characters) so a sentence cut at a boundary appears whole in one of the two chunks.
- **Sentence:** group whole sentences up to a size limit — never cuts mid-sentence.
- **Paragraph:** split on blank lines, merging very short paragraphs.
- **Semantic:** split where the topic changes (headings, transition phrases, or a drop in embedding similarity between neighbouring sentences).

Smaller chunks give more precise matches and fit more of them in the prompt, but can split an answer from the sentence that explains it; larger chunks keep context together but dilute the embedding and cost more prompt tokens. Overlap trades index size for robustness at boundaries.

### THEORY: BM25 — sparse retrieval

BM25 (from Lesson 4.6) scores documents using term frequency with saturation and document-length normalisation:

$$\text{BM25}(q, d) = \sum_{t \in q} \text{idf}(t) \cdot \frac{f_{t,d} (k_1 + 1)}{f_{t,d} + k_1 (1 - b + b \cdot |d|/|d_{\text{avg}}|)}$$

with typical $k_1 = 1.5$ (how quickly repeated terms stop adding score) and $b = 0.75$ (how strongly long documents are penalised). BM25 is fast, interpretable, and excels at exact keyword matching — names, codes, rare terms. It fails on paraphrase ("What are the rules for HDB ownership?" will not match a document about "public housing eligibility criteria" unless the words overlap).

### THEORY: Cosine similarity — dense retrieval

Dense retrieval embeds both the query and documents as dense vectors, then finds the most similar:

$$\text{sim}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|}$$

The course embeds with `nomic-embed-text` (768 dimensions) on local Ollama through `make_embedder()`. Dense retrieval captures semantic similarity ("HDB ownership" matches "public housing eligibility") but can miss exact terms such as product codes. A raw dot product equals the cosine only when the vectors are normalised to unit length; otherwise long vectors win regardless of direction.

### FOUNDATIONS: Hybrid retrieval

Combine BM25 and dense retrieval using reciprocal rank fusion (Cormack et al., 2009):

$$\text{RRF}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)}$$

where $k$ is a constant (typically 60). RRF uses ranks, not scores, so it needs no calibration between BM25's unbounded scores and cosine similarities in $[-1, 1]$. A document ranked well by both retrievers rises to the top.

### FOUNDATIONS: Re-ranking

First-stage retrievers are *bi-encoders*: query and document are embedded separately, which makes search fast but approximate. A **cross-encoder** reads the query and a candidate document together and outputs one relevance score — much more accurate, but too slow to run over the whole corpus. The standard pattern is to retrieve 20–50 candidates cheaply and re-rank them:

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")   # downloads once
scores = reranker.predict([(query, doc) for doc in candidates])
top3 = [candidates[i] for i in scores.argsort()[::-1][:3]]
```

Exercise 6.4 (`ex_4/05`) uses the local LLM as the cross-encoder instead (it scores each query–passage pair 0–10), which needs no extra model but is slower.

### FOUNDATIONS: RAGAS evaluation

The RAGAS framework (Es et al., 2024) defines four metrics for RAG quality, each scored in $[0, 1]$:

- **Faithfulness:** is every claim in the answer supported by the retrieved context?
- **Answer relevance:** does the answer address the question?
- **Context relevance:** are the retrieved chunks relevant to the question?
- **Context recall:** does the retrieved context contain the information in the reference answer?

Exercise 6.4 computes all four with an LLM judge on local Ollama (`compute_ragas_metrics` in `ex_4/05`), one judge prompt per metric. Retrieval itself is measured without any LLM: **hit@k** (here equal to recall@k, since each question has one relevant document) is the fraction of questions whose source document appears in the top $k$.

### ADVANCED: HyDE (Hypothetical Document Embeddings)

Generate a hypothetical answer to the query (even if wrong), embed it, and use that embedding for retrieval (Gao et al., 2023). The intuition: a passage-shaped answer is closer in embedding space to real passages than a short question is. It costs one extra LLM call per query, and it can hurt when the model's guess pulls retrieval towards the wrong topic — measure it, do not assume it.

### ADVANCED: Advanced RAG patterns

- **Metadata filtering:** filter chunks by source, date, department or language *before* the similarity search. A question about 2024 rules should never retrieve a 2019 circular, however similar its wording.
- **Multi-hop retrieval:** retrieve, read, form a follow-up query, retrieve again — for questions whose answer spans two documents ("Which company acquired the startup founded by X?"). The follow-up query is a reasoning step, which is why Lesson 6.5 builds multi-hop question answering on HotpotQA as an agent loop.
- **Document summarisation (hierarchical retrieval):** index one summary per document; retrieve the document by its summary first, then search only that document's chunks. This keeps long documents findable when no single chunk resembles the question.

### FOUNDATIONS: Kaizen RAG agents

Kaizen ships ready-made agents for the two most common patterns. `RAGResearchAgent` keeps its own vector store (local sentence-transformer embeddings) and retrieves-then-answers in one call; `MemoryAgent` remembers earlier turns per session. Both run on local Ollama, and their `run()` is synchronous.

```python
from kaizen_agents.agents import MemoryAgent, RAGResearchAgent
from shared.mlfp06._ollama_bootstrap import DEFAULT_CHAT_MODEL

rag = RAGResearchAgent(llm_provider="ollama", model=DEFAULT_CHAT_MODEL, top_k_documents=3)
rag.add_document("d1", "Refund policy", "Refunds are accepted within 30 days of purchase.")
out = rag.run(query="How long is the refund window?")
print(out["answer"], out["sources"], out["confidence"])

mem = MemoryAgent(llm_provider="ollama", model=DEFAULT_CHAT_MODEL)
mem.run("I handle motor insurance claims.", session_id="u42")
reply = mem.run("Which claims do I handle?", session_id="u42")
print(reply["response"])
```

Use them for quick prototypes; build the pipeline yourself (as in the exercise) when you need control over chunking, hybrid retrieval, re-ranking and evaluation.

## Worked Example: Dense RAG on the Course Corpus

```python
import re
from shared.mlfp06.ex_4 import (DenseVectorStore, embed_many, generate_embedding,
                                load_rag_corpus, rag_answer)

def chunk_sentence(text: str, max_chunk_chars: int = 500) -> list[str]:
    """Group whole sentences into chunks of at most ~max_chunk_chars."""
    chunks, current = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if current and len(current) + len(sent) + 1 > max_chunk_chars:
            chunks.append(current.strip())
            current = sent
        else:
            current = f"{current} {sent}" if current else sent
    if current.strip():
        chunks.append(current.strip())
    return chunks

corpus = load_rag_corpus().head(200)          # 200 documents keep embedding quick

# 1. Chunk, remembering each chunk's source document
chunks, owner = [], []
for doc_id, text in zip(corpus["section"], corpus["text"]):
    for chunk in chunk_sentence(text, max_chunk_chars=500):
        chunks.append(chunk)
        owner.append(doc_id)

# 2-3. Embed with nomic-embed-text on Ollama and index
store = DenseVectorStore()
vectors = await embed_many(chunks)
for chunk_id, (chunk, doc_id, vector) in enumerate(zip(chunks, owner, vectors)):
    store.add(chunk, vector, meta={"doc_id": doc_id, "chunk_id": chunk_id})

# 4. Retrieve for a question whose source document we know
question, gold_doc = corpus["question"][0], corpus["section"][0]
hits = store.search(await generate_embedding(question), top_k=5)
print("source document retrieved:", gold_doc in [h["metadata"]["doc_id"] for h in hits])

# 5. Generate a grounded answer and compare with the reference
context = "\n\n".join(h["text"] for h in hits)
print(await rag_answer(question, context))
print("reference:", corpus["answer"][0])
```

Run the retrieval step over many questions, not one: a single query proves nothing about a retriever. Drill 1 does exactly that.

## Try It Yourself

**Drill 1.** Implement BM25 retrieval from scratch. Evaluate it with hit@1 and hit@5 on 100 of the corpus questions, then compare with dense retrieval on the same questions. Which wins on keyword-heavy questions? On paraphrased ones?

**Solution:** the BM25 class below follows `ex_4/03` (same tokeniser, smoothed IDF, $k_1 = 1.5$, $b = 0.75$); `chunk_sentence` is the one from the worked example. Chunk hits are collapsed to their source document before scoring.

```python
import math
import re
from collections import Counter

class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokens = [re.findall(r"\w+", d.lower()) for d in documents]
        self.lengths = [len(t) for t in self.tokens]
        self.avgdl = sum(self.lengths) / len(documents)
        self.tf = [Counter(t) for t in self.tokens]
        df = Counter(term for toks in self.tokens for term in set(toks))
        n = len(documents)
        self.idf = {t: math.log((n - d + 0.5) / (d + 0.5) + 1) for t, d in df.items()}

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        terms = re.findall(r"\w+", query.lower())
        scores = []
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
            s = sum(self.idf.get(t, 0.0) * tf[t] * (self.k1 + 1) / (tf[t] + norm)
                    for t in terms if t in tf)
            scores.append((i, s))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

def hit_at_k(ranked_chunk_ids, owner, gold_doc, k):
    docs = []
    for i in ranked_chunk_ids:              # collapse chunks to documents
        if owner[i] not in docs:
            docs.append(owner[i])
    return gold_doc in docs[:k]

corpus = load_rag_corpus()                  # all 1,000 documents
chunks, owner = [], []
for doc_id, text in zip(corpus["section"], corpus["text"]):
    for chunk in chunk_sentence(text, 500):
        chunks.append(chunk)
        owner.append(doc_id)
bm25 = BM25(chunks)

eval_rows = corpus.head(100)
for k in (1, 5):
    hits = sum(hit_at_k([i for i, _ in bm25.search(q, 50)], owner, gold, k)
               for q, gold in zip(eval_rows["question"], eval_rows["section"]))
    print(f"BM25 hit@{k} = {hits / 100:.2f}")
# BM25 hit@1 = 0.92
# BM25 hit@5 = 0.97
```

BM25 is very strong here (measured: hit@1 0.92, hit@5 0.97 over 8,219 chunks) because the questions were generated from their passages and reuse their words. For the dense side, embed all chunks with `embed_many`, rank with `DenseVectorStore.search`, and compute the same hit@k. Then read the questions BM25 misses: they are the paraphrased ones, and that is where dense retrieval earns its place. On real user questions, which rarely copy the document's wording, the gap usually narrows or reverses.

**Drill 2.** Implement hybrid retrieval using reciprocal rank fusion. Does it outperform both BM25 and dense retrieval individually?

**Solution:**

```python
def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Fuse ranked lists of chunk ids; returns chunk ids by fused score."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)

async def hybrid_ids(question: str, depth: int = 50) -> list[int]:
    sparse = [i for i, _ in bm25.search(question, depth)]
    q_emb = await generate_embedding(question)
    dense = [h["metadata"]["chunk_id"] for h in dense_store.search(q_emb, depth)]
    return reciprocal_rank_fusion([sparse, dense])
```

`dense_store` is a `DenseVectorStore` built exactly as in the worked example, but over the same 1,000-document `chunks` list as `bm25`, so chunk ids match. Score `hybrid_ids` with `hit_at_k` exactly as in Drill 1. Hybrid usually matches the better of the two and fixes some of each one's misses; when one retriever is already near the ceiling (as BM25 is on this corpus), the gain is small.

**Drill 3.** Evaluate your RAG system using RAGAS-style metrics. Compute faithfulness and context relevance for 10 questions.

**Solution:** follow `compute_ragas_metrics` in `ex_4/05`: for each question, retrieve, generate with `rag_answer`, then ask the judge one question per metric and parse a number in $[0, 1]$.

```python
import re
from shared.mlfp06._ollama_bootstrap import make_delegate, run_delegate_text

judge = make_delegate(temperature=0.0)

async def judge_score(prompt: str) -> float | None:
    text, _usage, _secs = await run_delegate_text(judge, prompt + "\nOutput ONLY a number between 0.0 and 1.0.")
    match = re.search(r"\d*\.?\d+", text)
    value = float(match.group()) if match else None
    return value if value is not None and 0.0 <= value <= 1.0 else None   # None = judge failed

async def evaluate(question: str) -> dict:
    hits = store.search(await generate_embedding(question), top_k=3)
    context = "\n\n".join(h["text"] for h in hits)
    answer = await rag_answer(question, context)
    return {
        "faithfulness": await judge_score(
            f"Is every claim in the answer supported by the context?\n\n"
            f"Context: {context[:1500]}\n\nAnswer: {answer}"),
        "context_relevance": await judge_score(
            f"How relevant is this context to the question?\n\n"
            f"Question: {question}\n\nContext: {context[:1500]}"),
    }

results = [await evaluate(q) for q in corpus["question"].head(10)]
```

Average each metric over the questions where the judge returned a valid number, and report the failures separately rather than counting them as zero.

**Drill 4.** Implement HyDE. Compare retrieval quality (hit@5) with and without HyDE.

**Solution:**

```python
from shared.mlfp06._ollama_bootstrap import make_delegate, run_delegate_text

writer = make_delegate(temperature=0.0)

async def hyde_search(question: str, top_k: int = 5) -> list[dict]:
    passage, _usage, _secs = await run_delegate_text(
        writer, f"Write a short paragraph that would answer this question:\n{question}")
    return store.search(await generate_embedding(passage), top_k=top_k)

hyde_hits, plain_hits = 0, 0
for q, gold in zip(corpus["question"].head(50), corpus["section"].head(50)):
    hyde_docs = {h["metadata"]["doc_id"] for h in await hyde_search(q)}
    plain_docs = {h["metadata"]["doc_id"] for h in store.search(await generate_embedding(q), 5)}
    hyde_hits += gold in hyde_docs
    plain_hits += gold in plain_docs
print(f"dense hit@5 {plain_hits / 50:.2f} vs HyDE hit@5 {hyde_hits / 50:.2f}")
```

Compare like with like: the same questions, the same store, the same $k$. On this corpus the questions already share vocabulary with their passages, so HyDE has little room to help and can hurt; it pays off on short, vague or jargon-free questions.

**Drill 5.** Vary the chunk size (250, 500, 1,000 and 2,000 characters) and measure BM25 hit@1 and hit@5. How does chunk size affect retrieval quality?

**Solution:** reuse `chunk_sentence`, `BM25`, `hit_at_k` and `eval_rows` from above.

```python
for size in [250, 500, 1000, 2000]:
    chunks, owner = [], []
    for doc_id, text in zip(corpus["section"], corpus["text"]):
        for chunk in chunk_sentence(text, size):
            chunks.append(chunk)
            owner.append(doc_id)
    bm25 = BM25(chunks)
    h1 = sum(hit_at_k([i for i, _ in bm25.search(q, 50)], owner, g, 1)
             for q, g in zip(eval_rows["question"], eval_rows["section"]))
    h5 = sum(hit_at_k([i for i, _ in bm25.search(q, 50)], owner, g, 5)
             for q, g in zip(eval_rows["question"], eval_rows["section"]))
    print(f"{size:>5} chars  {len(chunks):>6} chunks  hit@1={h1 / 100:.2f}  hit@5={h5 / 100:.2f}")
```

Measured on the course corpus (1,000 documents, first 100 questions):

| Chunk size (chars) | Chunks | hit@1 | hit@5 |
| ------------------ | ------ | ----- | ----- |
| 250                | 16,511 | 0.88  | 0.98  |
| 500                | 8,219  | 0.92  | 0.97  |
| 1,000              | 4,143  | 0.92  | 0.97  |
| 2,000              | 2,222  | 0.93  | 0.99  |

Very small chunks lose hit@1: a single short sentence that happens to share the question's words can outrank the right passage. Beyond 500 characters, document-level retrieval barely changes — but every retrieved chunk now costs four times the prompt tokens at 2,000 characters, and the generator must find the answer in a longer context. Retrieval quality is only half of the chunk-size decision; answer quality and prompt cost (Drill 3) are the other half.

## Cross-References

- **Lesson 4.6** introduced TF-IDF and BM25. BM25 is the sparse retrieval backbone of RAG.
- **Lesson 6.1** covered prompt engineering. RAG is prompt engineering with dynamic context.
- **Lesson 6.5** will use RAG as a tool within agents.

## Reflection

You should now be able to build a complete RAG pipeline, choose a chunking strategy, compare sparse, dense and hybrid retrieval with hit@k on a labelled question set, re-rank candidates, evaluate answers with RAGAS-style metrics, and decide when HyDE, metadata filtering, multi-hop retrieval or a ready-made Kaizen RAG agent is the right tool.

---

# Lesson 6.5: AI Agents — ReAct, Tool Use, and Function Calling

## Why This Matters

An LLM generates text. An agent generates actions. The difference is that an agent can observe the results of its actions and adjust its behaviour accordingly. A ReAct agent follows a thought-action-observation loop: it reasons about the task, takes an action (call a tool, search a database, run code), observes the result, and decides what to do next.

## Core Concepts

### THEORY: ReAct formalisation

ReAct (Reasoning + Acting; Yao et al., 2023) interleaves reasoning traces with actions:

$$\text{Thought}_t \to \text{Action}_t \to \text{Observation}_t \to \text{Thought}_{t+1} \to \ldots$$

The thought is free-form text where the agent reasons about the current state. The action invokes a tool with specific parameters. The observation is the tool's output. The loop continues until the agent produces a final answer — or hits a bound you set (below).

### FOUNDATIONS: Chain-of-thought agents

A chain-of-thought agent reasons step by step *before* answering but takes no actions: it is Lesson 6.1's CoT prompting packaged as an agent. Use it when everything needed is already in the prompt (a policy question with the policy attached, a calculation with all the numbers). Use a ReAct agent when the answer depends on information the agent must fetch or compute.

### FOUNDATIONS: Function calling

Function calling provides structured tool invocation. Each tool is described by a name, a description and a JSON schema for its arguments:

```python
tools = [
    {
        "name": "search_listings",
        "description": "Search HDB resale transactions by town and price range",
        "parameters": {
            "type": "object",
            "properties": {
                "town": {"type": "string"},
                "min_price": {"type": "number"},
                "max_price": {"type": "number"},
            },
            "required": ["town"],
        },
    }
]
```

The LLM selects the appropriate tool and fills in the arguments from the user's natural-language request; your code executes the call and returns the result as the observation. Three controls matter in practice:

- **`tool_choice`:** `auto` (the model decides whether to call a tool), `required` (it must call some tool) or a specific function (it must call that one) — for example, forcing a `search` call before any answer.
- **Parallel function calling:** a model may request several independent calls in one turn (look up two towns at once); your executor runs them and returns all the results.
- **Descriptions are the interface:** the model chooses tools by their descriptions. Two tools with near-identical descriptions are the commonest cause of wrong tool selection.

### FOUNDATIONS: Tools in Kaizen

A Kaizen `Delegate` only calls tools that are registered in a `ToolRegistry`: name, description, JSON-schema parameters and an `async` executor. Passing a plain list of Python functions registers nothing — the agent then answers without tools, silently. Exercise 6.5 builds the registry with `shared.mlfp06.ex_5.build_tool_registry(...)`; the worked example below does it by hand.

### FOUNDATIONS: Bounding an agent — turns, tokens, dollars

Agents can loop: a confused agent may call tools many times before realising it is stuck, and on a priced API every call costs money. Three ceilings exist, and they apply in different places:

| Ceiling | How to set it | Works on local Ollama? |
| ------- | ------------- | ---------------------- |
| Turns   | `make_delegate(tools=..., max_turns=8)` | Yes — the most useful bound in this course |
| Tokens  | Measure per run (`run_delegate_text` usage) and cap prompt/answer sizes | Yes |
| Dollars | `BaseAgentConfig(budget_limit_usd=0.50)`, passed **at construction** | No — local inference is priced at $0, so a dollar cap never trips |

```python
from kaizen import Signature, InputField, OutputField
from kaizen.core.base_agent import BaseAgent, BaseAgentConfig
from shared.mlfp06._ollama_bootstrap import DEFAULT_CHAT_MODEL, OLLAMA_BASE_URL

class BriefingSignature(Signature):
    """Write a three-sentence briefing on a topic."""
    topic: str = InputField(description="The topic")
    briefing: str = OutputField(description="Three sentences")

capped = BaseAgent(
    config=BaseAgentConfig(
        llm_provider="ollama", model=DEFAULT_CHAT_MODEL, base_url=OLLAMA_BASE_URL,
        use_async_llm=True, budget_limit_usd=0.50,
    ),
    signature=BriefingSignature(),
)
print(capped.execution_context.budget_limit)   # 0.5 — the limit actually enforced
# Trap: assigning capped.config.budget_limit_usd AFTER construction changes the
# config object but not the enforced limit. Always pass it at construction.
```

`make_delegate` deliberately sets the Delegate's `budget_usd=None`, because Kaizen's cost estimator would price free local tokens at hosted-API rates. On local Ollama, bound agents by turns and measure tokens; on a priced provider, add the dollar cap as well. Lesson 6.7 cascades budgets across agents through PACT envelopes.

### FOUNDATIONS: Agent design framework

When designing an agent, ask four questions:

1. **What is our goal?** — the task in concrete terms.
2. **What is our thought process?** — the reasoning steps.
3. **What kind of specialist would we hire?** — be precise ("ML data analyst" not "researcher").
4. **What tools do they need?** — versatile, fault-tolerant, with caching.

Three design considerations follow from them:

- **Iterative refinement:** a critic agent reviews the output and returns concrete revisions (`ex_5/04_critic_agent.py` builds an Analyse → Critique → Refine loop).
- **Human-in-the-loop:** pause before irreversible or expensive actions (sending an email, retraining a model) and ask a person to approve.
- **Monitoring and logging:** record every tool call, its arguments and its result, so a wrong answer can be traced to the step that caused it (Lesson 6.8 does this with the Observatory's agent lens).

### FOUNDATIONS: Ready-made Kaizen agents

`kaizen_agents.agents` provides `ReActAgent` and `ChainOfThoughtAgent`. Both take `llm_provider="ollama"` and `model=DEFAULT_CHAT_MODEL`, and their `run()` is synchronous (`ReActAgent.run(task=...)`, `ChainOfThoughtAgent.run(problem=...)`). Custom agents are a `BaseAgent` with your own `Signature` (Lesson 6.1, `ex_5/03`). For tool use, the course's standard pattern is a `Delegate` with a `ToolRegistry`, shown next.

## Worked Example: Data Analysis Agent with Tools

The agent answers questions about the HDB resale data from Module 1 by calling two tools: a Kailash `DataExplorer` profile and a polars group-by.

```python
import polars as pl
from kailash_ml import DataExplorer
from kaizen_agents.delegate.loop import ToolRegistry
from shared import MLFPDataLoader
from shared.mlfp06._ollama_bootstrap import make_delegate, run_delegate_text

df = MLFPDataLoader().load("mlfp01", "hdb_resale.parquet")
# Module 1 planted impossible prices (S$10, S$9M) in this file; drop them first
df = df.filter(pl.col("resale_price").is_between(50_000, 2_000_000))

async def explore_data() -> str:
    """Profile the table with a Kailash engine (truncated for the prompt)."""
    profile = await DataExplorer().profile(df)
    return str(profile.to_dict())[:2000]

async def mean_price_by(column: str) -> str:
    """Mean resale price grouped by one column, highest first."""
    if column not in df.columns:
        return f"Unknown column {column!r}. Columns: {df.columns}"   # tool errors are observations
    out = (df.group_by(column).agg(pl.col("resale_price").mean().round(0))
             .sort("resale_price", descending=True).head(10))
    return str(out)

tools = ToolRegistry()
tools.register(name="explore_data", description="Profile the HDB resale table",
               parameters={"type": "object", "properties": {}},
               executor=explore_data)
tools.register(name="mean_price_by",
               description="Mean resale price grouped by a column such as town or flat_type",
               parameters={"type": "object", "required": ["column"],
                           "properties": {"column": {"type": "string"}}},
               executor=mean_price_by)

agent = make_delegate(tools=tools, max_turns=8)     # model: OLLAMA_CHAT_MODEL
answer, usage, seconds = await run_delegate_text(
    agent, "Which flat type has the highest mean resale price, and by how much?")
print(answer)
print(usage["total_tokens"], f"{seconds:.1f}s")
```

Two details make this robust. The tool returns an error message instead of raising when the model asks for a column that does not exist, so the agent can observe the mistake and retry. And `max_turns=8` guarantees the loop ends even if the model never settles on an answer. To see which tools the agent actually called, capture the run with the Observatory (Lesson 6.8): `await LLMObservatory().agent.capture_run(agent, prompt, run_id="hdb_1")`.

Exercise 6.5 applies the same pattern to multi-hop questions from HotpotQA (`ex_5/01`), bounds a runaway agent (`ex_5/02`), builds a structured BaseAgent (`ex_5/03`) and a critic loop (`ex_5/04`).

## Try It Yourself

**Drill 1.** Build an agent with three tools (data profiler, grouped statistics, and a chart tool that saves a plot to disk and returns its path). Test it on an end-to-end analysis question.

**Solution:** add a third tool to the worked example's registry.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

async def plot_price_by(column: str) -> str:
    """Bar chart of mean price by a column; returns the saved file path."""
    if column not in df.columns:
        return f"Unknown column {column!r}"
    agg = df.group_by(column).agg(pl.col("resale_price").mean()).sort("resale_price")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(agg[column].cast(pl.String).to_list(), agg["resale_price"].to_list())
    ax.set_xlabel("mean resale price (S$)")
    path = f"price_by_{column}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return f"saved {path}"

tools.register(name="plot_price_by", description="Save a bar chart of mean price by a column",
               parameters={"type": "object", "required": ["column"],
                           "properties": {"column": {"type": "string"}}},
               executor=plot_price_by)
agent = make_delegate(tools=tools, max_turns=10)
answer, usage, _ = await run_delegate_text(
    agent, "Profile the data, find the most expensive town, and chart mean price by town.")
```

Check the answer against the data yourself (`mean_price_by("town")`): an agent that answers without calling a tool is guessing, however fluent the prose.

**Drill 2.** Bound a runaway agent. Give it an impossible task and compare `max_turns=3` with `max_turns=20`. Then construct a BaseAgent with `budget_limit_usd` and explain why it never trips on local Ollama.

**Solution:** use the Observatory to count what each run actually did.

```python
from shared.mlfp06.diagnostics import LLMObservatory

obs = LLMObservatory()
impossible = "Find the resale price of a flat in a town that is not in the data, called Atlantis."
for turns in (3, 20):
    bounded = make_delegate(tools=tools, max_turns=turns)
    await obs.agent.capture_run(bounded, impossible, run_id=f"turns_{turns}")
    print(turns, obs.agent.tool_usage(f"turns_{turns}"))
```

The turn ceiling is a hard stop that works on any provider. The dollar cap is enforced against the provider's price per token; local Ollama inference costs nothing, so the spend stays at $0 and the cap never fires. On a priced API the same `budget_limit_usd=0.50` (passed at construction) stops the agent once its cumulative spend reaches 50 cents.

**Drill 3.** Build a function-calling agent with strict structured tool schemas (types, `required`, `enum` for allowed column names). Compare its tool-call error rate with the loose schema of the worked example.

**Solution:** constrain the argument with an `enum`, so the model can only name a real column:

```python
strict = ToolRegistry()
strict.register(
    name="mean_price_by",
    description="Mean resale price grouped by one categorical column",
    parameters={"type": "object", "required": ["column"],
                "properties": {"column": {"type": "string",
                                          "enum": ["town", "flat_type", "flat_model", "storey_range"]}}},
    executor=mean_price_by,
)
```

Run the same ten questions through both registries with `capture_run`, and count tool calls whose result starts with "Unknown column". The enum turns a class of runtime errors into something the model cannot express.

**Drill 4.** Add error handling: if a tool call fails, the agent should retry with different parameters or use an alternative tool.

**Solution:** never let an executor raise into the loop; return an informative message instead, and tell the agent what to do with it:

```python
def safe(executor):
    async def wrapped(**kwargs) -> str:
        try:
            return await executor(**kwargs)
        except Exception as exc:          # the observation names the failure
            return f"TOOL ERROR {type(exc).__name__}: {exc}. Try different arguments or another tool."
    return wrapped

robust = ToolRegistry()
robust.register(name="mean_price_by", description="Mean resale price grouped by a column",
                parameters={"type": "object", "required": ["column"],
                            "properties": {"column": {"type": "string"}}},
                executor=safe(mean_price_by))
agent = make_delegate(tools=robust, max_turns=8,
                      system_prompt="If a tool returns TOOL ERROR, change the arguments or use another tool.")
```

The exception is caught and turned into an observation, which is different from swallowing it: the agent (and your trace) still sees exactly what went wrong.

**Drill 5.** Implement an iterative refinement pattern: after the initial analysis, a critic agent evaluates the result and suggests improvements, then the original agent implements them.

**Solution:** `ex_5/04_critic_agent.py` is the full version. The shape is three typed agents in a bounded loop:

```python
from kaizen import Signature, InputField, OutputField

class Critique(Signature):
    """Review a data analysis for errors and gaps."""
    analysis: str = InputField(description="The analysis to review")
    issues: str = OutputField(description="Concrete problems, one per line")
    should_revise: bool = OutputField(description="True if the analysis needs another pass")

critic = BaseAgent(config={"llm_provider": "ollama", "model": DEFAULT_CHAT_MODEL,
                           "base_url": OLLAMA_BASE_URL, "use_async_llm": True,
                           "response_format": {"type": "json_object"},
                           "structured_output_mode": "explicit"},
                   signature=Critique())

draft, _, _ = await run_delegate_text(agent, "Summarise how flat type drives resale price.")
for _round in range(3):                       # bounded: at most three revisions
    review = await critic.run_async(analysis=draft)
    if not review.get("should_revise"):
        break
    draft, _, _ = await run_delegate_text(
        agent, f"Revise this analysis.\nIssues:\n{review.get('issues')}\n\nAnalysis:\n{draft}")
print(draft)
```

Unlike self-consistency (Lesson 6.1), which samples independent answers and votes, refinement feeds each critique into the next attempt. Bound the loop: a critic can always find something to say.

## Cross-References

- **Lesson 6.1** introduced Kaizen Delegate. Agents extend delegates with tool use and reasoning loops.
- **Lesson 6.6** will orchestrate multiple agents.
- **Lesson 6.7** will govern agents with PACT.

## Reflection

You should now be able to build tool-using agents with a `ToolRegistry`, write structured tool schemas, bound agents with turn ceilings (and dollar caps on priced providers), and apply the four-question design framework with critic refinement, human approval and logging.

---

# Lesson 6.6: Multi-Agent Orchestration and MCP

## Why This Matters

Complex tasks require multiple specialists. A data analysis task might need a data profiler, a feature engineer, a model trainer, and a report writer — each with different tools and expertise. Multi-agent orchestration coordinates these specialists, and MCP (Model Context Protocol) provides the standard for exposing tools to agents at scale.

## Core Concepts

### FOUNDATIONS: Multi-agent patterns

**Supervisor-worker.** One supervisor agent delegates sub-tasks to specialist workers. The supervisor decides which worker to call and aggregates results. Best when the decomposition is dynamic.

**Sequential.** Output of one agent feeds into the next: DataScientist → FeatureEngineer → ModelSelector → ReportWriter. Best for pipeline-like tasks where each stage needs the previous stage's result.

**Parallel.** Multiple agents work simultaneously on independent sub-tasks (fan-out), and the results are aggregated (fan-in). Latency becomes the slowest worker's time instead of the sum — but only if the calls really run concurrently (`asyncio.gather`), not one after another in a loop.

**Handoff.** An agent transfers control — and the conversation so far — to a specialist when the topic leaves its expertise (a general support agent hands a billing dispute to a billing agent).

Decision rule: start with sequential (simplest to debug), add parallelism where sub-tasks are independent, and use a supervisor only when the decomposition must be decided at run time.

**Structured hand-offs.** In every pattern, agents should pass each other *typed* outputs (Signature fields), not free-form chat. A downstream agent that receives `{"claims": [...], "evidence_quality": "high"}` can be validated; one that receives a paragraph cannot.

### FOUNDATIONS: Architecture and security considerations

- **Modularity:** one specialist = one Signature = one responsibility, so a specialist can be swapped or tested alone.
- **Load balancing:** run several replicas of a busy specialist behind a router (round-robin or least-busy), so one slow model call does not stall every request.
- **Dynamic agent creation:** a supervisor may spawn a specialist per sub-task — always inside a hard limit on the number of children and on delegation depth, or a confused planner spawns agents without end.
- **Isolation:** agents must not see data beyond their own authorisation. Every agent's output is untrusted input to the next one (prompt injection travels along the pipeline), and agent A must not get agent B to do what A itself may not do. Lesson 6.7 enforces these boundaries with PACT.

### FOUNDATIONS: A2A (agent-to-agent) communication

When agents run as separate services, they need a protocol, not shared memory. Agent-to-agent protocols have each agent publish an **agent card** (its capabilities, the tasks it accepts, how to reach it) and exchange typed task messages with a lifecycle — submitted → working → input-required → completed — with streaming for long tasks. MCP (below) connects an agent to *tools*; A2A connects an agent to *other agents*.

### FOUNDATIONS: Agent memory

| Memory      | Holds                                              | Lifetime              | Typical store                    |
| ----------- | -------------------------------------------------- | --------------------- | -------------------------------- |
| Short-term  | The current conversation (a sliding window of turns) | One session          | The prompt / an in-process list  |
| Long-term   | Facts and insights worth keeping across sessions   | Persistent            | A file, a database, a vector store searched by similarity |
| Entity      | Structured facts about specific people, datasets, projects | Persistent      | A key-value store keyed by entity |

Never rely on the context window alone in production: it is short-term memory with a hard size limit. Exercise 6.6 (`ex_6/05_memory_and_security.py`) implements all three and probes each with questions whose answers are known — including one with no stored answer, where a correct memory must say it does not know.

### FOUNDATIONS: MCP (Model Context Protocol)

MCP standardises how tools are exposed to agents: a server publishes tools with JSON schemas; any MCP client — an agent framework, an IDE assistant, a desktop chat app — discovers them with the JSON-RPC method `tools/list` and invokes them with `tools/call`. The transport is chosen when the server is built: `stdio` (the client launches the server as a subprocess and talks over stdin/stdout — no network port at all) or HTTP/SSE (a remote server).

```python
import polars as pl
from kailash_mcp import MCPServer
from shared.mlfp06.ex_6 import load_squad_corpus

passages = load_squad_corpus()                     # SQuAD 2.0 passages (title, text, question, ...)
server = MCPServer(name="ml-tools", transport="stdio")

@server.tool()                                     # note the parentheses
def search_corpus(query: str, top_k: int = 3) -> str:
    """Return passages containing the query text.

    Args:
        query: Text to search for.
        top_k: Maximum number of passages to return.
    """
    hits = passages.filter(pl.col("text").str.contains(query, literal=True))
    return "\n\n".join(hits["text"].head(top_k).to_list())

@server.tool()
def get_corpus_stats() -> str:
    """Number of passages and distinct article titles in the corpus."""
    return f"{passages.height} passages from {passages['title'].n_unique()} articles"

if __name__ == "__main__":
    server.run()                                   # serves until the client disconnects
```

The tool's name, description and argument schema are published from the function name, docstring and type hints. A client then discovers and calls the tools:

```python
import sys
from kailash_mcp import MCPClient

server_config = {"transport": "stdio", "command": sys.executable, "args": ["ml_tools_server.py"]}
client = MCPClient()
tools = await client.discover_tools(server_config, timeout=60)   # tools/list
print([t["name"] for t in tools])                                # ['search_corpus', 'get_corpus_stats']
stats = await client.call_tool(server_config, "get_corpus_stats", {})   # tools/call
print(stats["success"], stats["content"])                       # True 300 passages from 35 articles
```

## Worked Example: Supervisor-Worker and Sequential Pipelines

Exercise 6.6 builds both patterns on SQuAD 2.0 passages with three typed specialists (factual, semantic, structural analysis) and a synthesis agent from `shared.mlfp06.ex_6`, all running on `OLLAMA_CHAT_MODEL`.

```python
import asyncio
import json
from shared.mlfp06.ex_6 import (InterpretationAgent, build_specialists, build_synthesis,
                                load_squad_corpus, run_checked)

factual, semantic, structural = build_specialists()   # each a BaseAgent with its own Signature
supervisor = build_synthesis()

async def supervisor_worker(doc: str, question: str) -> dict:
    # Fan-out: the three specialists run concurrently
    f, s, st = await asyncio.gather(
        run_checked(factual, document=doc, question=question),
        run_checked(semantic, document=doc, question=question),
        run_checked(structural, document=doc, question=question),
    )
    # Fan-in: the supervisor reads three STRUCTURED outputs
    return await run_checked(
        supervisor, document=doc, question=question,
        factual_analysis=json.dumps(f, default=str),
        semantic_analysis=json.dumps(s, default=str),
        structural_analysis=json.dumps(st, default=str),
    )   # -> unified_answer, confidence, reasoning_chain

async def sequential(doc: str, question: str) -> dict:
    claims = await run_checked(factual, document=doc, question=question)        # stage 1
    interpreted = await run_checked(InterpretationAgent(),                       # stage 2
                                    factual_claims=str(claims["factual_claims"]),
                                    document=doc, question=question)
    return await run_checked(supervisor, document=doc, question=question,         # stage 3
                             factual_analysis=str(interpreted["interpreted_facts"]),
                             semantic_analysis=str(interpreted["relevance_ranking"]),
                             structural_analysis=f"Evidence quality: {claims['evidence_quality']}")

row = load_squad_corpus().row(0, named=True)
result = await supervisor_worker(row["text"], row["question"])
print(result["unified_answer"], result["confidence"])
```

`run_checked` raises if an agent's LLM call failed instead of passing an error dict downstream, so a broken stage stops the pipeline loudly. Compare the two patterns on the same passages: the supervisor-worker run takes roughly the slowest specialist's time; the sequential run takes the sum of its stages, but stage 2 can use stage 1's claims.

## Try It Yourself

**Drill 1.** Implement the 4-agent sequential pipeline DataScientist → FeatureEngineer → ModelSelector → ReportWriter on the HDB resale data. Verify that each agent's output is consumed by the next.

**Solution:** give each role its own Signature, and feed each agent the previous agent's typed fields.

```python
import polars as pl
from kaizen import Signature, InputField, OutputField
from kaizen.core.base_agent import BaseAgent
from shared import MLFPDataLoader
from shared.mlfp06._ollama_bootstrap import DEFAULT_CHAT_MODEL, OLLAMA_BASE_URL
from shared.mlfp06.ex_6 import run_checked

class DataScientistSig(Signature):
    """Describe a dataset and the most promising signals for the target."""
    data_summary: str = InputField(description="Column types and summary statistics")
    target: str = InputField(description="Column to predict")
    findings: str = OutputField(description="Key patterns, data issues and candidate predictors")

class FeatureEngineerSig(Signature):
    """Propose features from a data scientist's findings."""
    findings: str = InputField(description="Findings from the data scientist")
    features: str = OutputField(description="Feature list, one per line, with a reason each")

class ModelSelectorSig(Signature):
    """Choose a model family for the proposed features."""
    findings: str = InputField(description="Data findings")
    features: str = InputField(description="Proposed features")
    model_choice: str = OutputField(description="Model family and why")
    evaluation_plan: str = OutputField(description="Split, metric and baseline")

class ReportWriterSig(Signature):
    """Write a short report for a non-technical manager."""
    findings: str = InputField(description="Data findings")
    features: str = InputField(description="Proposed features")
    model_choice: str = InputField(description="Chosen model")
    evaluation_plan: str = InputField(description="Evaluation plan")
    report: str = OutputField(description="Five-sentence report")

def agent(sig: Signature) -> BaseAgent:
    return BaseAgent(config={"llm_provider": "ollama", "model": DEFAULT_CHAT_MODEL,
                             "base_url": OLLAMA_BASE_URL, "use_async_llm": True,
                             "response_format": {"type": "json_object"},
                             "structured_output_mode": "explicit"}, signature=sig)

df = MLFPDataLoader().load("mlfp01", "hdb_resale.parquet")
summary = str(df.describe())

ds = await run_checked(agent(DataScientistSig()), data_summary=summary, target="resale_price")
fe = await run_checked(agent(FeatureEngineerSig()), findings=ds["findings"])
ms = await run_checked(agent(ModelSelectorSig()), findings=ds["findings"], features=fe["features"])
report = await run_checked(agent(ReportWriterSig()), findings=ds["findings"], features=fe["features"],
                           model_choice=ms["model_choice"], evaluation_plan=ms["evaluation_plan"])
print(report["report"])
```

To verify the hand-offs, check that every stage's input fields are non-empty strings taken from the previous stage's output — not that the prose sounds plausible.

**Drill 2.** Build an MCP server exposing three tools. Connect a client and verify it can discover and call them.

**Solution:** add a third `@server.tool()` to the server above (for example `count_passages(title: str) -> str`), save it as `ml_tools_server.py`, and run the client code: `discover_tools` must list all three names, and `call_tool` must return each tool's result. `ex_6/04_mcp_server.py` does exactly this, including a call with an argument outside the tool's `Literal` type, which the server rejects before the handler runs.

**Drill 3.** Implement supervisor-worker orchestration where the supervisor decides which specialists to call.

**Solution:** let a router Signature pick the workers, then fan out to only those:

```python
class RouterSig(Signature):
    """Decide which analyses a question needs."""
    question: str = InputField(description="The user's question")
    analyses: str = OutputField(description="Comma-separated subset of: factual, semantic, structural")

router = agent(RouterSig())
workers = {"factual": factual, "semantic": semantic, "structural": structural}

async def routed(doc: str, question: str) -> dict:
    plan = await run_checked(router, question=question)
    chosen = [w.strip() for w in str(plan["analyses"]).split(",") if w.strip() in workers]
    chosen = chosen or list(workers)                 # an empty plan falls back to all three, visibly
    outputs = await asyncio.gather(*(run_checked(workers[w], document=doc, question=question)
                                     for w in chosen))
    return {"plan": chosen, "outputs": dict(zip(chosen, outputs))}
```

Log `plan` with every answer: when the final answer is wrong, the first question is whether the supervisor routed it to the right specialists.

**Drill 4.** Add long-term memory to an agent using a JSON file store. The agent should remember insights from previous sessions.

**Solution:**

```python
import json
from pathlib import Path
from shared.mlfp06._ollama_bootstrap import make_delegate, run_delegate_text

class JsonMemory:
    def __init__(self, path: str = "agent_memory.json"):
        self.path = Path(path)
        self.facts: dict[str, str] = json.loads(self.path.read_text()) if self.path.exists() else {}

    def remember(self, key: str, fact: str) -> None:
        self.facts[key] = fact
        self.path.write_text(json.dumps(self.facts, indent=2))

    def recall(self, query: str) -> list[str]:
        words = set(query.lower().split())
        return [f for k, f in self.facts.items() if words & set(k.lower().split("_"))]

memory = JsonMemory()
memory.remember("most_expensive_flat_type", "MULTI-GENERATION flats have the highest mean resale price.")
context = "\n".join(memory.recall("most expensive flat type")) or "No stored facts."
answer, _, _ = await run_delegate_text(
    make_delegate(), f"Known facts:\n{context}\n\nQuestion: Which flat type is most expensive?")
```

Keyword recall is enough to see persistence across sessions; a production long-term memory embeds the facts and retrieves by similarity, exactly like Lesson 6.4's dense retriever.

**Drill 5.** Implement parallel execution: launch two agents simultaneously and aggregate their results. Measure the speed-up over running them one after the other.

**Solution:**

```python
import time

row = load_squad_corpus().row(1, named=True)
start = time.perf_counter()
await run_checked(factual, document=row["text"], question=row["question"])
await run_checked(semantic, document=row["text"], question=row["question"])
sequential_s = time.perf_counter() - start

start = time.perf_counter()
await asyncio.gather(run_checked(factual, document=row["text"], question=row["question"]),
                     run_checked(semantic, document=row["text"], question=row["question"]))
parallel_s = time.perf_counter() - start
print(f"sequential {sequential_s:.1f}s, parallel {parallel_s:.1f}s")
```

On one local Ollama server the speed-up depends on whether it serves requests concurrently (`OLLAMA_NUM_PARALLEL`); if it queues them, "parallel" agents take as long as sequential ones. That is the load-balancing point above, measured.

## Cross-References

- **Lesson 6.5** built single agents. This lesson coordinates multiple agents.
- **Lesson 6.7** will govern multi-agent systems with PACT.

## Reflection

You should now be able to implement supervisor-worker, sequential, parallel and handoff patterns with typed hand-offs, build and call an MCP server, explain where A2A fits, configure short-term, long-term and entity memory, and name the isolation risks that Lesson 6.7 governs.

---

# Lesson 6.7: AI Governance Engineering

## Why This Matters

An ungoverned AI agent is a liability. It can access data it should not see, spend more money than budgeted, take actions outside its intended scope, and produce no audit trail of its decisions. PACT is the Terrene Foundation's governance framework for AI agent organisations (the `kailash-pact` package, imported as `pact`); it turns these risks into engineering constraints. Governance is not philosophy — it is code. Access controls you implement, operating envelopes you define, and budget cascading you test.

## Core Concepts

### THEORY: PACT D/T/R addressing

PACT locates every actor in an organisation with a positional address built from three kinds of unit:

- **D (Department):** a top-level organisational unit (e.g. "ML Engineering", "Risk & Compliance").
- **T (Team):** a team nested inside a department.
- **R (Role):** the role that heads a department or team — a person or an agent.

An address is a dash-delimited path like `D1-R1-T1-R1`, read left to right: Department 1 → its head role R1 → Team 1 inside it → that team's head role R1. The grammar has one rule: every `D` or `T` is immediately followed by exactly one `R`. Who *delegates* authority to whom is a separate concept — the envelope (below) records a defining role and a target role.

The organisation is defined in YAML and compiled in two steps. `GovernanceEngine(loaded.org_definition)` compiles the **structure only** (departments, teams, roles). The YAML's clearances and envelopes take effect only after `apply_governance_specs(engine, loaded)`. The course helper `shared.mlfp06.ex_7.compile_governance()` does both for the course organisation — three departments, six teams, nine roles.

```python
from kailash.trust.pact.yaml_resolvers import apply_governance_specs
from pact import Address, GovernanceEngine, load_org_yaml
from shared.mlfp06.ex_7 import write_org_yaml

loaded = load_org_yaml(write_org_yaml())          # the course org YAML
engine = GovernanceEngine(loaded.org_definition)  # structure only
apply_governance_specs(engine, loaded)            # clearances + envelopes now enforced

addr = Address.parse("D1-R1-T1-R1")               # Data Analyst: dept 1 -> head -> team 1 -> head

verdict = engine.verify_action("D1-R1-T1-R1", "read_data", context={"cost": 0.10})
print(verdict.allowed, verdict.level, verdict.reason)
# True auto_approved Action 'read_data' is within all constraint dimensions
```

`verify_action(role_address, action, context)` is the single decision call. It returns a `GovernanceVerdict` whose `.level` is one of `auto_approved`, `flagged`, `held` or `blocked`; `.allowed` is `True` for `auto_approved` and `flagged`; `.reason` names the rule that decided.

### FOUNDATIONS: The default — fail-open for roles without an envelope

You must know what the engine does when it has nothing to check against. In the installed `kailash-pact` (0.14.1), **a role with no attached envelope, and an address that is not in the organisation at all, are auto-approved**:

```python
from shared.mlfp06.ex_7 import compile_governance

bare, _org = compile_governance(apply_specs=False)        # structure only, no envelopes
v = bare.verify_action("D1-R1-T1-R1", "delete_customer_data")
print(v.allowed, v.level, v.reason)
# True auto_approved No envelope constraints -- action permitted

engine, org = compile_governance()                        # envelopes applied
v = engine.verify_action("D99-R99-T99-R99", "read_data")  # not in the org
print(v.allowed, v.level)
# True auto_approved
```

This is a *fail-open* default: absence of policy means permission. A deny path exists only where an envelope is attached. So the engineering rules are: attach an envelope to every role that can act; reject unknown addresses in your own code (for example, in the API handler) before calling the engine; and write a test that pins this default, so that you notice if an upgrade changes it.

### FOUNDATIONS: Operating envelopes

A `ConstraintEnvelopeConfig` defines the boundaries of what a role can do across **five dimensions** — Financial, Operational, Temporal, Data Access and Communication — plus a `confidentiality_clearance` and a `max_delegation_depth` cap. A `RoleEnvelope` attaches one to a role: a *defining* role (the supervisor) sets it for a *target* role (the direct report).

- **Operational:** the allowed action surface (`allowed_actions=["read_data", "summarise_data"]`). An action outside the list is `blocked`.
- **Financial:** a spending cap (`max_spend_usd=20.0`). An action whose `context={"cost": ...}` exceeds it is `blocked`.
- **Confidentiality clearance:** PACT's ladder, lowest to highest, is `PUBLIC < RESTRICTED < CONFIDENTIAL < SECRET < TOP_SECRET`. "Restricted" is the **second-lowest** tier, not the top — giving your most privileged role "restricted" gives it *less* access than "confidential". (`GovernedSupervisor(data_clearance="internal")` is an alias for restricted.) In the course organisation, department heads hold `secret`.
- **Monotonic tightening:** an envelope can only get stricter, never looser, as you descend the delegation tree. `RoleEnvelope.validate_tightening(parent_envelope=..., child_envelope=...)` checks this structurally and raises `MonotonicTighteningError` naming every dimension the child widens. Its arguments are **keyword-only**; a positional call raises `TypeError`, not the governance error.

```python
from pact import (
    CommunicationConstraintConfig, ConfidentialityLevel, ConstraintEnvelopeConfig,
    DataAccessConstraintConfig, FinancialConstraintConfig, MonotonicTighteningError,
    OperationalConstraintConfig, RoleEnvelope, TemporalConstraintConfig,
)

def envelope(env_id, clearance, max_spend, actions):
    return ConstraintEnvelopeConfig(
        id=env_id, description=env_id,
        confidentiality_clearance=clearance,
        financial=FinancialConstraintConfig(max_spend_usd=max_spend),
        operational=OperationalConstraintConfig(allowed_actions=actions),
        temporal=TemporalConstraintConfig(), data_access=DataAccessConstraintConfig(),
        communication=CommunicationConstraintConfig(), max_delegation_depth=3,
    )

parent = envelope("analyst_env", ConfidentialityLevel.CONFIDENTIAL, 10.0, ["analyse", "delegate"])
child_ok = envelope("worker_env", ConfidentialityLevel.RESTRICTED, 3.0, ["analyse"])
child_bad = envelope("rogue_env", ConfidentialityLevel.SECRET, 12.0, ["analyse", "deploy"])

RoleEnvelope.validate_tightening(parent_envelope=parent, child_envelope=child_ok)   # passes
try:
    RoleEnvelope.validate_tightening(parent_envelope=parent, child_envelope=child_bad)
except MonotonicTighteningError as exc:
    print(exc)   # lists the financial, operational and confidentiality violations

# Attach an envelope to the BARE engine from the previous block: the department
# head (D1-R1) defines it for the analyst role. The deny path appears only now.
bare.set_role_envelope(RoleEnvelope(
    id="analyst_env", defining_role_address="D1-R1",
    target_role_address="D1-R1-T1-R1", envelope=parent,
))
print(bare.verify_action("D1-R1-T1-R1", "delete_customer_data").level)   # blocked
print(bare.verify_action("D1-R1-T1-R1", "analyse").level)                # auto_approved
```

### FOUNDATIONS: Budget cascading

Budgets cascade through the financial dimension. A parent role's envelope caps spend (`max_spend_usd=10.00`); each child's cap must be at most the parent's, which `validate_tightening` enforces; and `verify_action(..., context={"cost": c})` blocks any single action whose cost exceeds the role's cap. In the course organisation, for example, the Data Analyst's cap is $20, so a $25 action is refused:

```python
v = engine.verify_action("D1-R1-T1-R1", "read_data", context={"cost": 25.0})
print(v.allowed, v.level, v.reason)
# False blocked Action cost ($25.00) exceeds financial limit ($20.00)
```

The envelope bounds each action; it does not keep a running total. Cumulative spend per child — allocate, spend, refuse the overspend — is a ledger the supervisor keeps (Exercise 6.7 part 3 builds one with `TeachingBudgetTracker`). Each agent can also carry its own cap (`BaseAgentConfig.budget_limit_usd`, Lesson 6.5). On local Ollama the dollar amounts are notional, so the exercise uses them as a teaching currency.

### FOUNDATIONS: GovernedSupervisor

`kaizen_agents.GovernedSupervisor` is the governed agent entry point. Three knobs — `budget_usd`, `tools` (the allowed action list) and `data_clearance` — become its envelope. It does not wrap an existing agent: `await supervisor.run(objective, execute_node=...)` plans the task and governs each step, and **your** `execute_node` callback runs the model call and reports its cost and tokens. Every step is written to a hash-chained audit trail.

```python
from kaizen_agents import GovernedSupervisor
from shared.mlfp06._ollama_bootstrap import DEFAULT_CHAT_MODEL
from shared.mlfp06.ex_7 import make_llm_executor

governed = GovernedSupervisor(
    model=DEFAULT_CHAT_MODEL,          # from OLLAMA_CHAT_MODEL
    budget_usd=2.00,
    tools=["answer_question", "search_faq"],
    data_clearance="confidential",
)
print(governed.envelope.financial.max_spend_usd)        # 2.0
print(governed.envelope.operational.allowed_actions)    # ['answer_question', 'search_faq']
print(governed.envelope.confidentiality_clearance.name) # CONFIDENTIAL

result = await governed.run(
    objective="Summarise the refund policy for a customer",
    execute_node=make_llm_executor(),   # calls the local model; returns result, cost, tokens
)
print(result.success, result.budget_consumed)

for record in governed.audit.to_list()[-5:]:
    print(record["record_type"], record["action"], record["record_hash"][:12])
assert governed.audit.verify_chain()    # False if any record was altered
```

### FOUNDATIONS: Governance testing

Governance without tests is governance theatre. Test that allowed actions succeed, that denied actions stay denied, that envelopes cannot be loosened, and that the default you rely on is the default you have. Every deny test must run on an engine whose roles **have** envelopes — on a bare engine "analyst cannot delete" fails.

```python
import pytest
from pact import MonotonicTighteningError, RoleEnvelope
from shared.mlfp06.ex_7 import compile_governance

@pytest.fixture
def engine():
    eng, _org = compile_governance()       # YAML envelopes ATTACHED
    return eng

def test_analyst_can_read_data(engine):
    assert engine.verify_action("D1-R1-T1-R1", "read_data").allowed

def test_analyst_cannot_delete(engine):
    v = engine.verify_action("D1-R1-T1-R1", "delete_customer_data")
    assert not v.allowed and v.level == "blocked"

def test_analyst_cannot_train(engine):
    assert not engine.verify_action("D1-R1-T1-R1", "train_model").allowed

def test_trainer_can_train(engine):
    assert engine.verify_action("D1-R1-T2-R1", "train_model").allowed

def test_spend_over_cap_is_blocked(engine):
    assert not engine.verify_action("D1-R1-T1-R1", "read_data", context={"cost": 25.0}).allowed

def test_unknown_address_is_auto_approved(engine):
    # Pins pact 0.14's fail-open default. If an upgrade changes it, this fails
    # and tells you; production code rejects unknown addresses itself.
    assert engine.verify_action("D99-R99-T99-R99", "read_data").allowed

def test_envelope_cannot_loosen():
    with pytest.raises(MonotonicTighteningError):
        RoleEnvelope.validate_tightening(parent_envelope=parent, child_envelope=child_bad)
```

`parent` and `child_bad` are the envelopes built above. All seven tests pass against the course organisation.

## Worked Example: Governed Analyst and Trainer

The course organisation gives the Data Analyst (`D1-R1-T1-R1`, clearance restricted) the actions read, summarise and report, and the Model Trainer (`D1-R1-T2-R1`, clearance confidential) train, evaluate and read. Check every request with the engine *before* running the agent, and run the agent under a supervisor whose envelope matches the role.

```python
from kaizen_agents import GovernedSupervisor
from shared.mlfp06._ollama_bootstrap import DEFAULT_CHAT_MODEL
from shared.mlfp06.ex_7 import compile_governance, make_llm_executor

engine, org = compile_governance()
ROLES = {"analyst": org.address_of("data_analyst"),     # D1-R1-T1-R1
         "trainer": org.address_of("model_trainer")}    # D1-R1-T2-R1
agents = {
    "analyst": GovernedSupervisor(model=DEFAULT_CHAT_MODEL, budget_usd=2.00,
                                  tools=["read_data", "summarise_data", "generate_report"],
                                  data_clearance="restricted"),
    "trainer": GovernedSupervisor(model=DEFAULT_CHAT_MODEL, budget_usd=5.00,
                                  tools=["train_model", "evaluate_model", "read_data"],
                                  data_clearance="confidential"),
}

async def governed_request(role: str, action: str, objective: str) -> dict:
    if role not in ROLES:                                 # never fall back on an unknown role
        return {"allowed": False, "reason": f"unknown role {role!r}"}
    verdict = engine.verify_action(ROLES[role], action)
    if not verdict.allowed:
        return {"allowed": False, "level": verdict.level, "reason": verdict.reason}
    result = await agents[role].run(objective=objective, execute_node=make_llm_executor())
    return {"allowed": True, "success": result.success, "spent": result.budget_consumed}

print(engine.verify_action(ROLES["analyst"], "train_model").level)   # blocked
print(engine.verify_action(ROLES["trainer"], "train_model").level)   # auto_approved
out = await governed_request("analyst", "summarise_data", "Summarise last quarter's churn drivers")
```

The two verdict lines are deterministic (verified against the course organisation); the agent run needs Ollama. Note the order of checks: unknown role refused by your code, then the organisation's envelope, then the supervisor's own envelope during the run.

## Try It Yourself

**Drill 1.** Write a small org YAML with 2 departments, 3 teams and their head roles. Compile it, print every role's address, and test five access rules including the boundary cases.

**Solution:** follow the structure of `shared.mlfp06.ex_7.ORG_YAML` (sections `departments`, `teams`, `roles` with `heads` and `reports_to`, `clearances`, `envelopes`), then:

```python
from kailash.trust.pact.yaml_resolvers import apply_governance_specs
from pact import GovernanceEngine, load_org_yaml

loaded = load_org_yaml("my_org.yaml")
engine = GovernanceEngine(loaded.org_definition)
apply_governance_specs(engine, loaded)
for address, node in sorted(engine.get_org().nodes.items()):
    print(address, node.node_type.name, node.name)

cases = [  # (address, action, context, expected allowed) — adjust to your YAML
    ("D1-R1-T1-R1", "read_data", {}, True),                  # inside the envelope
    ("D1-R1-T1-R1", "delete_customer_data", {}, False),      # outside the action list
    ("D1-R1-T1-R1", "read_data", {"cost": 1_000.0}, False),  # over the financial cap
    ("D1-R1", "delete_customer_data", {}, True),             # head role with no envelope
    ("D9-R9", "read_data", {}, True),                        # address not in the org
]
for address, action, context, expected in cases:
    v = engine.verify_action(address, action, context=context)
    print(f"{address:12s} {action:22s} {v.level:13s} {'OK' if v.allowed == expected else 'MISMATCH'}")
```

Write down why the last two come back allowed (the fail-open default), and what your API layer must do about it.

**Drill 2.** Attach operating envelopes for two agents. Verify that the analyst is blocked from training models and the trainer is allowed.

**Solution:** the worked example's two `verify_action` lines are the test; turn them into assertions (`level == "blocked"` and `allowed`). Then build the same pair of envelopes yourself with `envelope(...)` from the envelopes section and attach them with `engine.set_role_envelope(...)` on a bare engine (`compile_governance(apply_specs=False)`) to see the deny path appear only after attachment.

**Drill 3.** Implement budget cascading: a supervisor with $10 allocates $3 to each of 3 workers. Verify that each worker stops at its budget limit without consuming the supervisor's remaining budget.

**Solution:**

```python
from shared.mlfp06.ex_7 import TeachingBudgetTracker

ledger = TeachingBudgetTracker(total_budget=10.0)
for worker in ("w1", "w2", "w3"):
    assert ledger.allocate(worker, 3.0)
assert not ledger.allocate("w4", 3.0)        # only $1 left unallocated
assert ledger.spend("w1", 2.5)
assert not ledger.spend("w1", 1.0)           # would exceed w1's $3 allocation
print(ledger.summary())                      # w1 has $0.50 left; w2, w3 untouched

# The envelope side: each worker's cap must tighten the supervisor's
RoleEnvelope.validate_tightening(
    parent_envelope=envelope("sup", ConfidentialityLevel.CONFIDENTIAL, 10.0, ["analyse", "delegate"]),
    child_envelope=envelope("w1", ConfidentialityLevel.CONFIDENTIAL, 3.0, ["analyse"]),
)
```

**Drill 4.** Write governance tests that verify: (a) denied access stays denied, (b) envelopes cannot be loosened, (c) a cost above the cap is blocked, (d) the fail-open default is what you think it is.

**Solution:** the test module in "Governance testing" covers all four; run it with `pytest`. Then delete the `apply_governance_specs` step from the fixture and watch (a) and (c) fail — that is the proof that your envelopes, not luck, produce the denials.

**Drill 5.** Generate an audit trail for a multi-agent workflow. The trail should log every access decision (who, what, when, allowed/denied, reason).

**Solution:** log the engine's verdicts yourself, and keep each supervisor's hash-chained trail for what happened inside the run:

```python
from datetime import datetime, timezone
import polars as pl

decisions = []
for role, action in [("analyst", "read_data"), ("analyst", "train_model"),
                     ("trainer", "train_model"), ("trainer", "deploy_model")]:
    v = engine.verify_action(ROLES[role], action)
    decisions.append({"when": datetime.now(timezone.utc).isoformat(), "who": ROLES[role],
                      "action": action, "allowed": v.allowed, "level": v.level, "reason": v.reason})
print(pl.DataFrame(decisions))

for name, sup in agents.items():
    print(name, len(sup.audit.to_list()), "records, chain intact:", sup.audit.verify_chain())
```

Exercise 6.7 part 4 (`ex_7/04_runtime_audit.py`) runs the governed agents for real and audits both layers.

## Cross-References

- **Lesson 6.5** and **6.6** built agents without governance. This lesson adds the safety layer.
- **Lesson 6.8** will deploy governed agents to production.
- **Module 3, Lesson 3.6** covered fairness and model cards. PACT governance is the production enforcement of those principles.

## Reflection

You should now be able to:

- Implement PACT governance with D/T/R addressing.
- Define and enforce operating envelopes for agents.
- Implement budget cascading across agent hierarchies.
- State PACT's fail-open default for envelope-less roles and unknown addresses, and attach envelopes before relying on a deny path.
- Order clearance levels correctly: public < restricted < confidential < secret < top_secret.
- Test that governance rules are enforced (denied access stays denied).
- Generate audit trails for compliance.

---

# Lesson 6.8: Capstone — Full Production Platform

## Why This Matters

This is the last lesson of the MLFP programme. Everything you have learned converges here: data pipelines from Module 1, statistics from Module 2, the ML pipeline from Module 3, unsupervised learning from Module 4, deep learning architectures from Module 5, and LLMs, agents, and governance from Module 6.

In this lesson you will deploy a complete, governed AI system using Nexus — Kailash's multi-channel deployment platform. One registered handler is served as a REST API, a CLI command and an MCP tool. It has authentication, rate limiting, CORS, governance enforcement and drift monitoring. This is not a toy — it is the architecture of a real production AI application.

## Core Concepts

### FOUNDATIONS: Nexus multi-channel deployment

Nexus exposes one handler on three interfaces:

- **API:** REST endpoint (`POST /workflows/<name>/execute`) for programmatic access.
- **CLI:** `nexus execute <name>` for operators.
- **MCP:** an MCP tool other AI agents can discover and call.

A handler is an ordinary `async` function registered with `app.handler_extract(name, func)`; Nexus derives the input schema from its parameters. Functionality is added with **plugins** (`app.add_plugin(...)`): `NexusAuthPlugin` adds JWT authentication, role-based access and rate limiting; CORS is configured on the app itself.

```python
import os
import secrets
from kailash.trust.auth.jwt import JWTConfig
from kailash.trust.rate_limit.config import RateLimitConfig
from nexus import Nexus, NexusAuthPlugin
from starlette.requests import Request

# Never hardcode a signing secret: read it from the environment
jwt_config = JWTConfig(secret=os.environ.get("MLFP_JWT_SECRET") or secrets.token_urlsafe(48),
                       algorithm="HS256")

app = Nexus(api_port=8000,
            cors_origins=["https://intranet.example.sg"],   # CORS allow-list
            enable_durability=False)
app.add_plugin(NexusAuthPlugin(
    jwt=jwt_config,                                          # 401 without a valid token
    rate_limit=RateLimitConfig(requests_per_minute=10, burst_size=5),   # 429 when exceeded
))

async def echo_role(question: str, request: Request) -> dict:
    """Return the caller's verified role (set by the JWT middleware)."""
    return {"question": question, "roles": list(request.state.user.roles)}

app.handler_extract("echo_role", echo_role, description="Echo the caller's role")
# app.start()   # serve API on :8000 (CLI and MCP from the same registration)
```

A request to the API passes through layers, outermost first: rate limit (429), JWT (401 for a missing, forged or expired token), then your handler. The JWT layer is HTTP middleware: CLI and MCP callers are authenticated differently, so do not assume the 401 behaviour carries over to them.

### FOUNDATIONS: Authentication and authorisation

Nexus owns **authentication** (who are you?). PACT owns **authorisation** (what may you do?). The handler reads the role from the *verified* token — never from the request body, where any client could write `"role": "admin"` — maps it to a PACT address, and asks `engine.verify_action(...)` before doing any work. A governance refusal is returned as a normal response marked `blocked: true` with the reason; the 401 comes from the JWT layer before the handler runs.

```python
import httpx
from kailash.trust.auth.jwt import JWTValidator

# Your identity provider issues tokens; in the exercise we play that role
token = JWTValidator(jwt_config).create_access_token("analyst1", roles=["qa"])

transport = httpx.ASGITransport(app=app.fastapi_app)   # in-process, no network port
async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
    ok = await client.post("/workflows/echo_role/execute",
                           headers={"Authorization": f"Bearer {token}"},
                           json={"inputs": {"question": "Who am I?"}})
    anonymous = await client.post("/workflows/echo_role/execute",
                                  json={"inputs": {"question": "Who am I?"}})
print(ok.status_code, ok.json()["outputs"]["handler"])   # 200 {'question': 'Who am I?', 'roles': ['qa']}
print(anonymous.status_code)                              # 401
```

### FOUNDATIONS: Full platform integration

The complete stack, with the module where you first met each piece:

| Step | Package          | Purpose                                     | Met in |
| ---- | ---------------- | ------------------------------------------- | ------ |
| 1    | kailash-ml       | Train and register a model                  | M3     |
| 2    | kailash-dataflow | Persist data and results                    | M3     |
| 3    | kailash-align    | Fine-tune and register an adapter           | 6.2–6.3 |
| 4    | kailash-kaizen   | Wrap the model in an agent                  | 6.5–6.6 |
| 5    | kailash-pact     | Govern the agent                            | 6.7    |
| 6    | kailash-nexus    | Deploy on API + CLI + MCP                   | 6.8    |
| 7    | kailash-ml       | Monitor inputs for drift (`DriftMonitor`)   | M3.8, 6.8 |

Exercise 6.8 runs this chain for the module's QA system: it loads the adapters trained in Exercises 6.2 and 6.3 (`ex_8/01`), builds a three-tier governed stack (`ex_8/02`), serves it through Nexus with JWT auth, rate limiting and CORS (`ex_8/03`), monitors question drift and debugs agent runs (`ex_8/04`), and writes a compliance audit (`ex_8/05`).

### FOUNDATIONS: Production monitoring with DriftMonitor

`DriftMonitor` (Lesson 3.8) compares each production batch with a stored reference distribution, per feature, with PSI and the KS test. It persists references and reports through a `ConnectionManager`, so it is constructed with a connection and a tenant id, and its methods are `async`. For an LLM service, monitor features of the *inputs* — question length, language, topic — because the model's own weights do not change while the questions do.

```python
import numpy as np
import polars as pl
from pathlib import Path
from kailash.db.connection import ConnectionManager
from kailash_ml import DriftMonitor

conn = ConnectionManager(f"sqlite:///{Path('drift.db').resolve()}")   # absolute sqlite path
await conn.initialize()
monitor = DriftMonitor(conn, tenant_id="mlfp_demo", psi_threshold=0.2)

rng = np.random.default_rng(0)
reference = pl.DataFrame({"question_length": rng.normal(60, 15, 1_000)})
await monitor.set_reference_data("capstone_qa_model", reference, ["question_length"])

for name, mean_length in [("same distribution", 60), ("longer questions", 90)]:
    batch = pl.DataFrame({"question_length": rng.normal(mean_length, 15, 300)})
    report = await monitor.check_drift("capstone_qa_model", batch)
    psi = report.feature_results[0].psi
    print(f"{name}: drift={report.overall_drift_detected} "
          f"severity={report.overall_severity} PSI={psi:.3f}")
# same distribution: drift=False severity=none PSI=0.038
# longer questions: drift=True severity=severe PSI=5.063
await conn.close()
```

(The data here are synthetic, to make the two cases unambiguous; Exercise 6.8 uses question lengths from the QA traffic.) The usual PSI reading is: below 0.1 stable, 0.1–0.2 moderate, above 0.2 act — investigate, and retrain or re-tune if the shift persists.

### FOUNDATIONS: Debugging agent reasoning

When an agent gives a wrong answer, you need the chain of steps that produced it: which tools it called, with what arguments, what came back, and where it went in circles. The course Observatory's agent lens captures a real run of a Delegate as a trace:

```python
from shared.mlfp06.diagnostics import LLMObservatory

obs = LLMObservatory()
trace = await obs.agent.capture_run(agent, "Which flat type costs most?", run_id="debug_1")
for event in trace.events:                         # token / tool_start / tool_end / complete / error ...
    print(event.kind, event.tool or "", (event.content or event.error or "")[:60])
print(obs.agent.tool_usage("debug_1"))             # calls per tool
print(obs.agent.detect_loops("debug_1"))           # repeated identical tool calls

for record in governed.audit.to_list()[-5:]:       # governance decisions: the supervisor's chain
    print(record["record_type"], record["action"])
```

`agent` is the tool-using Delegate from Lesson 6.5 and `governed` any `GovernedSupervisor` from Lesson 6.7.

| Symptom                | Likely cause                                   |
| ---------------------- | ---------------------------------------------- |
| Agent loops            | Ambiguous goal, or no tool can produce the answer |
| Wrong tool selected    | Tool descriptions too similar                  |
| Answer without tool use | Tools not registered (a bare list of functions) |
| Governance blocked     | Action missing from the role's envelope        |
| Turn ceiling reached   | Too many retries; simplify the task            |
| Incoherent reasoning   | Context window overflow                        |

### FOUNDATIONS: Testing agentic systems

Test five things, cheapest first: each **tool** returns the right output for known input (plain unit tests, no LLM); **governance** blocks what it should (Lesson 6.7's tests, no LLM); **bounds** hold (the run stops at its turn ceiling); **tool selection** is right for representative prompts; and **end-to-end** answers are correct on a small labelled set. The last two need the model, so they assert on the trace, not the prose:

```python
import pytest

@pytest.mark.asyncio
async def test_agent_uses_grouping_tool():
    obs = LLMObservatory()
    await obs.agent.capture_run(make_delegate(tools=tools, max_turns=8),
                                "Which flat type is most expensive?", run_id="t1")
    used = obs.agent.tool_usage("t1")
    assert "mean_price_by" in used["tool"].to_list()
    assert obs.agent.detect_loops("t1").height == 0
```

### FOUNDATIONS: Inference optimisation for serving (brief)

- **KV-cache and continuous batching** (Lesson 6.1) are what serving engines are built around.
- **FlashAttention** computes exact attention in tiles that stay in fast on-chip memory, avoiding the full $n \times n$ attention matrix in GPU memory: faster and far less memory for long contexts, same result.
- **vLLM** is an open-source serving engine with PagedAttention (the KV-cache managed in pages, like virtual memory) and continuous batching; it serves many concurrent users from one GPU. Ollama (llama.cpp underneath) targets single-machine and laptop use with quantised GGUF models.
- **Quantisation** (Lesson 6.2) trades a little quality for 2–4× less memory.

### FOUNDATIONS: Multimodal LLMs (awareness)

Vision-language models (open ones such as LLaVA, and commercial assistants) feed image patches through a vision encoder into the language model's token stream, so the same transformer answers questions about images, charts and documents. Everything in this module — prompting, RAG, agents, governance and deployment — applies unchanged; the input simply contains images as well as text.

## Worked Example: Deploying the Governed QA System

This is the shape of `ex_8/02`–`ex_8/03`, built from the shared capstone helpers: three governance tiers (qa: public clearance, $1; admin: confidential, $10; audit: secret, $50), each a `GovernedSupervisor`, behind one Nexus handler.

```python
import os
import secrets
from kailash.trust.auth.jwt import JWTConfig
from kailash.trust.rate_limit.config import RateLimitConfig
from nexus import Nexus, NexusAuthPlugin
from starlette.requests import Request
from shared.mlfp06.ex_8 import build_capstone_stack, compile_capstone_governance, handle_qa

# 1. Governance: org + envelopes, then one governed supervisor per tier
engine, loaded_org = compile_capstone_governance()
agents_by_role, tiers = build_capstone_stack(engine)
for tier in tiers:
    print(f"{tier.role:6s} {tier.address}  ${tier.budget_usd:>5.1f}  {tier.clearance}")

# 2. The one handler: role from the VERIFIED token, then PACT, then the tier's agent
async def serve_qa(question: str, request: Request) -> dict:
    roles = list(getattr(request.state.user, "roles", None) or [])
    role = next((r for r in roles if r in agents_by_role), "")
    return await handle_qa(question, role=role, agents_by_role=agents_by_role, engine=engine)
    # handle_qa refuses an unknown role, returns {"blocked": True, ...} on a
    # blocked verdict, and otherwise runs the tier's supervisor on local Ollama

# 3. Deploy: JWT + rate limit plugin, CORS allow-list, one registration -> 3 channels
jwt_config = JWTConfig(secret=os.environ.get("MLFP_JWT_SECRET") or secrets.token_urlsafe(48),
                       algorithm="HS256")
app = Nexus(api_port=8000, cors_origins=["https://intranet.example.sg"], enable_durability=False)
app.add_plugin(NexusAuthPlugin(jwt=jwt_config,
                               rate_limit=RateLimitConfig(requests_per_minute=10, burst_size=5)))
app.handler_extract("capstone_serve_qa", serve_qa, description="Governed capstone QA")
# app.start()

# 4. Governance check per tier, before any model call
qa_address = next(t.address for t in tiers if t.role == "qa")
print(engine.verify_action(qa_address, "update_model").level)      # blocked
print(engine.verify_action(qa_address, "generate_answer").level)   # auto_approved
```

Add the drift monitor from the previous section to the handler (record each question's length, check a batch every N requests) and you have the full loop: authenticate → authorise → answer → monitor.

## Try It Yourself

**Drill 1.** Deploy the governed QA handler via Nexus and call the API channel in-process with valid, missing and forged tokens.

**Solution:** reuse the worked example's `app` and the in-process client from the authentication section. Issue tokens with `JWTValidator(jwt_config).create_access_token(user, roles=[...])`; a forged token is one signed with a *different* secret. Expect 200 for valid tokens and 401 for missing or forged ones. The CLI (`nexus execute capstone_serve_qa`) and MCP (`workflow_capstone_serve_qa`) channels come from the same registration; `ex_8/03` registers them but only exercises the API channel in-process.

**Drill 2.** Verify that unauthenticated requests are rejected with 401, and that an authenticated caller asking for an action outside its tier is refused by governance.

**Solution:** the 401 is the JWT layer (Drill 1). The refusal is not an HTTP error: `handle_qa` returns `{"blocked": True, "verdict": ..., "role": ...}` with status 200. Test both, and also send a qa token with `{"role": "audit"}` in the body — the answer must still come from the qa tier, because the role is read from the signed token.

**Drill 3.** Integrate DriftMonitor. Send a batch that should trigger drift (for example, much longer questions) and verify the report says so.

**Solution:** the DriftMonitor block above is the pattern. Use real question lengths from the QA data as the reference (`question.str.len_chars()` on the questions you served), then a batch of long, multi-part questions. Assert `report.overall_drift_detected` on the shifted batch and `not report.overall_drift_detected` on a held-out sample of the reference distribution — a drift test that cannot fail proves nothing.

**Drill 4.** Add governance enforcement at the Nexus level. Verify that the qa tier can answer but cannot update the model, and the admin tier can do both.

**Solution:**

```python
admin_address = next(t.address for t in tiers if t.role == "admin")
for address, action, expected in [(qa_address, "generate_answer", True),
                                  (qa_address, "update_model", False),
                                  (admin_address, "generate_answer", True),
                                  (admin_address, "update_model", True)]:
    v = engine.verify_action(address, action)
    assert v.allowed == expected, (address, action, v.level, v.reason)
```

Then call `handle_qa(question, role="qa", agents_by_role=agents_by_role, engine=engine, action="update_model")` and check it returns `blocked: True` without running the model.

**Drill 5.** Generate a complete audit trail for one request that flows through authentication → governance check → answer → drift monitoring → response. Every step should be logged.

**Solution:** log one structured record per layer, keyed by a request id: the JWT result (user and roles from `request.state.user`, or the 401), the `verify_action` verdict (address, action, level, reason), the supervisor's audit records for the run (`agents_by_role[role].audit.to_list()`, with `verify_chain()`), the drift report for the batch the question joined, and the response status. `ex_8/05_compliance_audit.py` assembles exactly this kind of report.

## Cross-References

- **Module 3, Lesson 3.8** introduced drift monitoring with PSI and KS tests. This lesson deploys it in production.
- **Lesson 6.7** defined governance rules. This lesson enforces them at the API level.
- **Lesson 6.6** built MCP servers. This lesson exposes them through Nexus.

## Reflection

You should now be able to:

- Deploy a complete AI system with Nexus (API + CLI + MCP).
- Authenticate callers with JWT in Nexus and authorise every request with PACT inside the handler.
- Integrate drift monitoring in production.
- Enforce governance at the deployment level.
- Debug agent reasoning chains from captured traces, and test agentic systems from tools up to end-to-end.

This is the end of the MLFP programme. You started in Module 1 not knowing what a variable was. You are ending Module 6 having deployed a governed, multi-channel AI system that trains models, aligns them with human preferences, grounds them in retrieved knowledge, coordinates multiple specialist agents, enforces access controls, monitors for drift, and serves predictions via API, CLI, and MCP simultaneously.

The distance you have covered is not measured in modules or lines of code. It is measured in the questions you can now ask — and answer. When someone shows you a model, you ask: how was it trained? When they show you a prediction, you ask: how confident is it, and has the input distribution shifted? When they show you an agent, you ask: what are its operating envelopes, and who audits its decisions?

Those are the questions of an ML engineer. Welcome to the profession.

---

# Chapter Summary

Module 6 covered the complete journey from a trained model to a governed production system:

| Lesson | Topic            | Key Concept                           |
| ------ | ---------------- | ------------------------------------- |
| 6.1    | LLM Fundamentals | Prompt engineering, structured output |
| 6.2    | Fine-Tuning      | LoRA, adapters, PEFT landscape        |
| 6.3    | Alignment        | DPO, GRPO, Bradley-Terry              |
| 6.4    | RAG              | Retrieval-augmented generation        |
| 6.5    | Agents           | ReAct, tool use, turn and cost bounds |
| 6.6    | Multi-Agent      | Orchestration, MCP, memory            |
| 6.7    | Governance       | PACT D/T/R, operating envelopes       |
| 6.8    | Capstone         | Nexus deployment, full integration    |

The progression mirrors the real-world deployment pipeline: understand the model (6.1), customise it (6.2), align it (6.3), ground it (6.4), give it tools (6.5), coordinate specialists (6.6), govern the system (6.7), and ship it (6.8).

## The complete MLFP arc

| Module | Theme                           | Key Skill                                        |
| ------ | ------------------------------- | ------------------------------------------------ |
| M1     | Data foundations                | Polars, visualisation, profiling                 |
| M2     | Statistics and probability      | Inference, hypothesis testing, regression        |
| M3     | Supervised ML                   | Full pipeline, evaluation, deployment            |
| M4     | Unsupervised ML + Neural bridge | Clustering, PCA, embeddings, backpropagation     |
| M5     | Deep learning architectures     | CNN, RNN, Transformer, GAN, GNN, RL              |
| M6     | LLMs and production systems     | Fine-tuning, RAG, agents, governance, deployment |

You have traversed the Feature Engineering Spectrum from manual features to learned features to semantic features. You have built models from linear regression to transformers to governed multi-agent systems. You have deployed with Kailash's full stack: Core SDK, DataFlow, ML, Kaizen, PACT, Nexus, and Align.

The programme is complete. The learning continues.

---

# Glossary

**Adapter layer.** A small bottleneck module inserted between transformer layers for parameter-efficient fine-tuning.

**Agent.** A system that reasons about tasks, takes actions, observes results, and iterates. Distinct from a model, which simply maps inputs to outputs.

**Alignment.** Training a model to produce outputs that are helpful, harmless, and honest, typically through RLHF or DPO.

**BM25.** A sparse retrieval scoring function that extends TF-IDF with term frequency saturation and document length normalisation.

**Bradley-Terry model.** A probabilistic model for pairwise preferences: $P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l))$.

**Budget cascading.** Allocating cost budgets from parent agents to child agents: each child's envelope cap is at most its parent's, and a ledger tracks what each child has spent.

**Clearance level.** PACT's confidentiality ladder, lowest to highest: public < restricted < confidential < secret < top_secret.

**Chain-of-thought (CoT).** A prompting technique that instructs the model to reason step by step before answering.

**Chunking.** Splitting documents into smaller pieces for retrieval in a RAG system.

**Cosine similarity.** A similarity measure between vectors based on the cosine of the angle between them.

**D/T/R addressing.** PACT's three-part access control structure: Department, Team, Role.

**Delegate.** Kaizen's streaming LLM interface (text, tool calls, token usage). M6 builds every Delegate with `make_delegate()`, which points it at local Ollama.

**Dense retrieval.** Finding similar documents using vector embeddings and cosine similarity.

**DPO (Direct Preference Optimization).** An alignment method that bypasses the reward model by deriving a loss function directly from preference data.

**Fail-open default.** In the installed PACT (0.14.1), a role with no attached envelope, or an address not in the organisation, is auto-approved. Deny paths exist only where an envelope is attached; the opposite policy — deny unless explicitly permitted — is called fail-closed.

**Few-shot prompting.** Providing examples in the prompt to guide the model's output format and quality.

**Function calling.** Structured tool invocation where the LLM selects a tool and fills in parameters based on natural language input.

**GovernanceEngine.** PACT's core component that compiles organisational structures and evaluates access requests.

**GRPO (Group Relative Policy Optimization).** A policy-gradient alignment method that scores several completions per prompt, normalises each reward by its group's mean and standard deviation, and optimises a clipped objective with a KL penalty. Introduced in DeepSeekMath (2024), later used for DeepSeek-R1.

**HyDE (Hypothetical Document Embeddings).** A RAG technique that generates a hypothetical answer and uses its embedding for retrieval.

**InferenceServer.** Kailash ML engine for serving model predictions in production.

**KV-cache.** Stored key and value matrices from previous attention computations, avoiding redundant calculation during autoregressive generation.

**LLM-as-Judge.** Using one LLM to evaluate another's outputs on criteria like helpfulness and accuracy.

**LoRA (Low-Rank Adaptation).** A parameter-efficient fine-tuning method that decomposes weight updates into low-rank matrices.

**MCP (Model Context Protocol).** A protocol for exposing tools to AI agents at scale with standardised schemas.

**Model merging.** Combining multiple fine-tuned model weights (TIES, DARE, SLERP, task arithmetic).

**Monotonic tightening.** The principle that operating envelopes can only become stricter, never looser, down the delegation tree; checked by `RoleEnvelope.validate_tightening`.

**Nexus.** Kailash's multi-channel deployment platform (API + CLI + MCP simultaneously).

**Operating envelope.** Defined boundaries for what an agent can do, enforced by PACT.

**PACT.** The Terrene Foundation's governance framework for AI agent organisations (`kailash-pact`): D/T/R addressing, operating envelopes, clearances, `verify_action` verdicts and audit chains.

**GovernedSupervisor.** A two-layer agent from `kaizen_agents` that plans the task while a caller-supplied `execute_node` callback runs the LLM. The envelope (budget, action surface, clearance) is attached at construction and enforced on every step. This is the modern pact-governed agent entry point.

**Preference alignment.** Training a model to prefer outputs that humans prefer, using methods like DPO or RLHF.

**Prompt engineering.** Designing input prompts to control LLM output quality and format.

**QLoRA.** Quantising the base model to 4-bit precision, then applying LoRA for fine-tuning.

**Quantisation.** Reducing model weight precision (e.g., from FP16 to INT4) to reduce memory and compute requirements.

**RAG (Retrieval-Augmented Generation).** Grounding LLM responses in retrieved documents to reduce hallucination and incorporate current knowledge.

**RAGAS.** An evaluation framework for RAG systems measuring faithfulness, relevance, and recall.

**RBAC (Role-Based Access Control).** Granting permissions based on the user's role in the organisation.

**ReAct.** A reasoning framework where agents interleave thought, action, and observation steps.

**Reference policy.** The original, unaligned model used as a baseline in DPO to prevent the aligned model from deviating too far.

**Self-consistency.** Sampling multiple reasoning paths and taking the majority vote to reduce variance.

**Signature.** Kaizen's typed input/output schema for structured LLM interaction.

**Sparse retrieval.** Finding documents using keyword matching (BM25, TF-IDF).

**Speculative decoding.** Using a small model to draft tokens that a larger model verifies, accelerating generation.

**Structured output.** LLM output parsed into typed fields rather than free-form text.

**Temperature.** A parameter controlling the randomness of LLM output. Low temperature = deterministic; high = random.

**Tool use.** An agent's ability to invoke external functions (APIs, databases, code execution) based on reasoning.

**ToolRegistry.** Kaizen's registry of tools a Delegate may call: name, description, JSON-schema parameters and an async executor. A plain list of functions registers nothing.

**Verdict level.** The outcome of PACT's `verify_action`: `auto_approved`, `flagged`, `held` or `blocked`; `.allowed` is true for the first two.

**Zero-shot prompting.** Providing only a task description with no examples.

---

# Further Reading

**On LLMs and prompt engineering**

- Brown, T., et al. "Language Models are Few-Shot Learners." _NeurIPS_, 2020. The GPT-3 paper introducing few-shot prompting.
- Wei, J., et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." _NeurIPS_, 2022.
- Wang, X., et al. "Self-Consistency Improves Chain of Thought Reasoning in Language Models." _ICLR_, 2023.
- Kojima, T., et al. "Large Language Models are Zero-Shot Reasoners." _NeurIPS_, 2022. The "Let's think step by step" paper.
- Hoffmann, J., et al. "Training Compute-Optimal Large Language Models." _NeurIPS_, 2022. The Chinchilla scaling law.

**On fine-tuning**

- Hu, E., et al. "LoRA: Low-Rank Adaptation of Large Language Models." _ICLR_, 2022. The original LoRA paper.
- Houlsby, N., et al. "Parameter-Efficient Transfer Learning for NLP." _ICML_, 2019. The adapter layers paper.
- Aghajanyan, A., Gupta, S., and Zettlemoyer, L. "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning." _ACL_, 2021.
- Dettmers, T., et al. "QLoRA: Efficient Finetuning of Quantized Language Models." _NeurIPS_, 2023.

**On preference alignment**

- Rafailov, R., et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." _NeurIPS_, 2023. The DPO paper.
- Shao, Z., et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." _arXiv:2402.03300_, 2024. Introduces GRPO.
- Ouyang, L., et al. "Training language models to follow instructions with human feedback." _NeurIPS_, 2022. The InstructGPT/RLHF paper.

**On RAG**

- Lewis, P., et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." _NeurIPS_, 2020. The original RAG paper.
- Gao, L., et al. "Precise Zero-Shot Dense Retrieval without Relevance Labels." _ACL_, 2023. The HyDE paper.
- Cormack, G., Clarke, C., and Büttcher, S. "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods." _SIGIR_, 2009.
- Es, S., et al. "RAGAS: Automated Evaluation of Retrieval Augmented Generation." _EACL (demonstrations)_, 2024.

**On AI agents**

- Yao, S., et al. "ReAct: Synergizing Reasoning and Acting in Language Models." _ICLR_, 2023. The ReAct paper.
- Schick, T., et al. "Toolformer: Language Models Can Teach Themselves to Use Tools." _NeurIPS_, 2023.

**On AI governance**

- Terrene Foundation. `kailash-pact` package documentation (the PACT governance framework used in Lesson 6.7).
- Mitchell, M., et al. "Model Cards for Model Reporting." _FAT\*_, 2019.

**On production ML systems**

- Sculley, D., et al. "Hidden Technical Debt in Machine Learning Systems." _NeurIPS_, 2015. The classic paper on ML system complexity.
- Paleyes, A., Urma, R.-G., and Lawrence, N. "Challenges in Deploying Machine Learning: A Survey of Case Studies." _ACM Computing Surveys_, 2022.

---
