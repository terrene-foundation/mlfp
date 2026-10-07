# Module 6: Machine Learning with Language Models and Agentic Workflows — Speaker Notes

Total time: ~180 minutes (3 hours)
Audience: working professionals; instructors must scaffold for both novices and LLM practitioners.

This is the final module of MLFP. By the end of today, students will have built, fine-tuned, aligned, grounded, governed, and deployed a production AI system. Pace the room accordingly: treat the capstone as the destination, and treat every earlier lesson as a tool they will pick up in the final hour.

The deck opens with a 15-slide Observatory section (slides 6–20): six diagnostic lenses that every later lesson reuses. Do not skip it — it is what turns "prompt play" into engineering for the rest of the day.

---

## Slide 1: Machine Learning with Language Models and Agentic Workflows

**Time**: ~2 min
**Talking points**:

- Welcome to the final module of MLFP. Read the provocation aloud: "The best LLM application is not a chatbot. It is an engineer that reasons, acts, and governs itself."
- Let it sit. This frames the entire module: we are not here to prompt a chatbot, we are here to build systems.
- Emphasise the pivot: every previous module was about models that take input and produce output. Today we build systems that think, plan, and act autonomously.
- Ask for a show of hands: "Who has used a chat assistant — ChatGPT or Claude?" Most will raise their hands. "You have used an LLM. Today you learn to build one."
- "If beginners look confused": "If you have only ever typed into a chat box, today you see the engineering underneath. We will go slowly through the fundamentals before we build anything."
- "If experts look bored": "We implement LoRA from scratch, derive the DPO loss term by term, and build a governed multi-agent system with PACT. The theory track will earn its keep."
  **Transition**: "Let me show you exactly what you will be able to do by the end of today."

---

## Slide 2: What You Will Learn

**Time**: ~2 min
**Talking points**:

- Walk through the eight outcomes on the left column. These are concrete capabilities, not abstract concepts.
- Introduce the three depth layers: FOUNDATIONS (green, everyone), THEORY (blue, stretch), ADVANCED (purple, experts).
- "A product manager and a research engineer sit in the same classroom. Both leave having learned something new. This module is designed for both."
- "If beginners look confused": "Follow the green markers. You will come out able to build RAG systems and governed agents. You can skip the derivations — they are labelled Theory."
- "If experts look bored": "The DPO derivation slide and the GRPO formula slides are purple. That is where you will find new depth."
  **Transition**: "Here is the roadmap for our 8 lessons today."

---

## Slide 3: Your Journey: 8 Lessons

**Time**: ~1 min
**Talking points**:

- Walk through the table once. Do not dwell on each cell.
- Emphasise the arc: understand LLMs (6.1), customise them (6.2), align them (6.3), ground them (6.4), give them agency (6.5), coordinate them (6.6), govern them (6.7), ship them (6.8).
- "Each lesson produces runnable code. By the end, you have a deployed AI platform."
- "If beginners look confused": "The first four lessons teach you what LLMs do. The last four teach you how to ship them safely."
- "If experts look bored": "6.2 and 6.3 are the densest. Focus your attention there."
  **Transition**: "To build all of that, you will use four Kailash packages."

---

## Slide 4: Kailash Engines You Will Meet

**Time**: ~2 min
**Talking points**:

- Introduce the four packages briefly. Kaizen: agent framework and LLM calls (Delegate, BaseAgent, Signature). kailash-align: fine-tuning and preference alignment (AlignmentPipeline, AdapterRegistry). kailash-pact: governance and access control (GovernanceEngine, GovernedSupervisor from kaizen_agents). kailash-nexus: multi-channel deployment.
- Mention kailash-mcp briefly — it is the MCP server library we will use in 6.6.
- "Every exercise uses at least one of these. By the end of 6.8, you will have used all of them in a single deployed system."
- "If beginners look confused": "Think of these as specialised tools. Kaizen is the one you will touch most often today."
- "If experts look bored": "AdapterRegistry, ConstraintEnvelopeConfig, and GovernedSupervisor are the three classes most experts have not seen before."
  **Transition**: "Before we dive in, let us locate ourselves in the MLFP journey."

---

## Slide 5: Where We Are

**Time**: ~2 min
**Talking points**:

- Walk through the table: M1 data, M2 statistics, M3 supervised ML, M4 unsupervised and NLP, M5 deep learning and transformers.
- "Today we take the transformer architecture from M5 and give it the ability to reason, act, learn from feedback, and govern itself. Everything you built before becomes a tool an agent can wield."
- "If beginners look confused": "Think of M5 as building the engine. M6 is building the car around it."
- "If experts look bored": "The interesting bit is how LoRA ties back to M4's SVD. We will reach that in 6.2."
  **Transition**: "Before Lesson 6.1, one more foundation: the six diagnostic lenses you will carry into every exercise from here."

**[PAUSE FOR QUESTIONS — 1 min]**

---

## Slide 6: Six Lenses

**Time**: ~3 min (45 sec story + 2 min pair share)
**Talking points**:

- Hook (say it in these words): "Picture this — a hypothetical, but a very common shape of failure: a team ships a customer chatbot that cheerfully invents a refund policy, and it runs for three weeks before anyone notices — the unit tests were green the whole time. A 10-line faithfulness check would have caught it on day one. That is what we are preventing today."
- Be explicit that the story is a hypothetical composite, not a report about a named company.
- Key question for the room: "Why do most ML engineers stop instrumenting the moment they switch from training to prompting?" Expected wrong answer: "because the model is a black box". Right answer: "because re-prompting feels faster than instrumenting — until the clock hits hour six and production is still hallucinating."
- Run the pair share (2 min): name a time you deployed an LLM and something weird happened — and how you figured out what. Self-study students jot it in their notebook.
- "If behind schedule": cut the pair share; keep the 45-second story — it is the emotional anchor for the whole section.
  **Transition**: "You just named symptoms. Now let me give you the six lenses to diagnose them."

---

## Slide 7: The Six Lenses

**Time**: ~3 min
**Talking points**:

- Hook: "Six questions, one per lens. You carry all six into every M6 exercise from here."
- Walk the split: the first four lenses DIAGNOSE the running system (Output, Attention, Retrieval, Agent Trace). The fifth EVALUATES the training signal (Alignment). The sixth is the organisational contract (Governance).
- Key question: "If the chatbot returns a plausible but wrong answer, which lens am I reading first?" Expected wrong answer: "Attention — check the heads". Right answer: "Output lens — faithfulness against retrieved context. Attention comes later, only if Output does not settle it."
- "If behind schedule": skim five of the six icons, spend time on the diagnose/evaluate/govern split — that is the mental model.
  **Transition**: "Lens by lens now. First the Output lens — M5's stethoscope, evolved for generations."

---

## Slide 8: Lens 1: Output — The Stethoscope, Evolved

**Time**: ~3 min
**Talking points**:

- Hook: "A fluent wrong answer looks identical to a fluent right answer. That is why we measure."
- Walk the asks: is the generation coherent, factual, on-task? Bounded tasks get perplexity/ROUGE/BLEU/BERTScore. Open-ended tasks get LLM-as-judge — pairwise, with swapped positions and normalised length. Hallucination gets self-consistency plus retrieval grounding. Safety gets refusal calibration: refuse harmful prompts, do NOT over-refuse benign ones.
- Key question: "Why not just eyeball a sample of 20 outputs and ship?" Expected wrong answer: "not enough coverage". Right answer: "human eyeballing is biased toward fluency — judge-models plus faithfulness scoring catch the confident-but-wrong class that humans rate as good."
- Point at the faithful-vs-hallucinated pair: one answer cites the retrieved policy verbatim, the other invents a 30-day window. Ship one, raise on the other.
- "If behind schedule": skip the refusal calibration row; hallucination vs faithful is the headline.
  **Transition**: "Here is what that looks like in one screen of code."

---

## Slide 9: Lens 1: Output — Code

**Time**: ~3 min
**Talking points**:

- Hook: "One object, a handful of method calls, one dashboard. You never hand-roll a judge."
- Point out: the judge is a Kaizen Delegate on the local Ollama model — no raw provider SDK anywhere in the course.
- Stress the signatures: `self_consistency` does NOT call the model — you generate the five samples (as in Exercise 6.1), it measures agreement (`agreement`, `is_outlier` columns). The refusal metrics score the model's RESPONSES, not the prompts.
- Key question: "When we compare TWO answers (Exercise 6.3), why swap their positions and normalise length?" Expected wrong answer: "fairness". Right answer: "because unmitigated LLM judges favour the first-shown and the LONGER answer; judging both orders and averaging cancels position bias, and length normalisation stops verbosity inflating the win rate."
- Budget note: every judge call is an LLM call; the observatory caps them (`max_judge_calls=50` by default). On local Ollama the cost is compute time, not dollars.
- "If behind schedule": skip self-consistency; faithfulness + judge is the core pair.
  **Transition**: "Output tells you the surface. Lens 2 tells you what is happening inside the model."

---

## Slide 10: Lens 2: Attention — The X-Ray

**Time**: ~3 min
**Talking points**:

- Hook: "If Output says 'wrong', Attention says 'why'. It is the only lens that looks through the model's skull."
- Walk the four reads: attention heatmaps (token-to-token weights per head), logit lens (the prediction at every layer), sparse autoencoders (read Gemma Scope features), linear probes (is a concept linearly readable at layer L?).
- Mech interp in 2026: we read FEATURES, not neurons. The course default for SAE work is Gemma-2-2B because Gemma Scope ships pre-trained SAEs for every layer — students read features, they do not train them.
- Key question: "Why Gemma-2-2B as the default, not Llama?" Expected wrong answer: "smaller". Right answer: "no comparable pre-trained SAE suite exists for Llama-3 at course-friendly size."
- Hosted API-only models cannot be X-rayed: the lens reports "not applicable" and falls back to Output + Agent Trace.
- "If behind schedule": skip probes — heatmap + logit lens is enough for the intuition. Circuit tracing / activation patching is a research topic beyond the course toolkit.
  **Transition**: "Here is the method surface in five lines."

---

## Slide 11: Lens 2: Attention — Code

**Time**: ~3 min
**Talking points**:

- Hook: "Four method calls cover most mech-interp questions a practitioner asks. Note the model is chosen ONCE, on the observatory — the methods take only the prompt."
- This wraps transformer_lens + sae_lens. Open-weight models only; API models report `not_applicable`.
- Walk the calls: `attention_heatmap(prompt, layer=8, head=3)`, `logit_lens(prompt, top_k=5)` (columns: layer, rank, token, prob), `sae_features(prompt, layer=12, top_k=10)`, `probe(prompts, labels, layer=6)`.
- Key question: "In the logit lens output, what does it mean if 'Paris' appears strongly at layer 5 but the model outputs 'Lyon'?" Expected wrong answer: "bug". Right answer: "the knowledge is in the residual stream but a later layer overrode it — now you know exactly where to intervene."
- Non-goal: we do NOT train SAEs in this course. SAE training is research; SAE reading is an M6 skill.
- "If behind schedule": skip probe; heatmap + logit lens carry the intuition.
  **Transition**: "Lens 3 leaves the model and looks at what we fed it."

---

## Slide 12: Lens 3: Retrieval — The RAG Diagnostic

**Time**: ~3 min
**Talking points**:

- Hook: "Most RAG post-mortems stop at recall@5. That is why most RAG post-mortems are wrong."
- Walk the per-stage metrics: recall@k / precision@k / MRR at the retriever, chunk relevance at the ranker, faithfulness AND context utilisation at the generator. The hidden failure: retrieval can be perfect AND the answer still wrong, because the generator ignored the chunks.
- Key question on the illustrative leaderboard: "Hybrid wins recall by 5 points but loses faithfulness by 15. How could that happen?" Expected wrong answer: "hybrid is worse". Right answer: "hybrid retrieves MORE chunks — the generator latches onto weaker ones and ungrounds. Classic over-retrieval pattern."
- Say clearly that the leaderboard numbers are illustrative — Exercise 6.4 fills in the real table from the student's own run.
- "If behind schedule": skip MRR — the recall-vs-faithfulness tension is the insight.
  **Transition**: "Code surface for the retrieval lens is short and scriptable."

---

## Slide 13: Lens 3: Retrieval — Code

**Time**: ~2 min
**Talking points**:

- Hook: "`compare_retrievers` is one call. In 6.4 you score four retrievers with it without hand-writing a single metric loop."
- A retriever is any callable `(query, k) → [(doc_id, text, score)]` — students plug their own BM25/dense/hybrid/HyDE functions straight in.
- Walk the calls: `recall_at_k(retrieved_ids, gold_ids, k=5)`, the `compare_retrievers` bake-off on an eval set, `context_utilisation(answer, chunks)` (a token-overlap heuristic — no LLM call), and `evaluate(...)` for end-to-end scores (ragas if usable, judge otherwise).
- Key question: "What happens if context_utilisation is 0.3? Which number do I change?" Expected wrong answer: "try a bigger model". Right answer: "tighten the prompt to cite chunks, OR re-rank so irrelevant chunks do not crowd the context window."
- First use: Exercise 6.4 (ex_4/05) — hit@k leaderboard for BM25 / dense / hybrid / HyDE on a 1,000-document sample of the open neural-bridge/rag-dataset-12000.
- "If behind schedule": skip evaluate — the bake-off is the headline.
  **Transition**: "Lens 4 is the one most teams never build — until the first 3am agent incident."

---

## Slide 14: Lens 4: Agent Trace — What Did The Agent Actually Do?

**Time**: ~3 min
**Talking points**:

- Hook: "Without this lens an agent failure is 'it got stuck'. With it, failure is 'step 4 opened a loop that burned 78% of budget in 10 seconds'. Same bug, two reports, one actionable."
- Walk the four capabilities: run capture (every token/tool_start/tool_end/error event, timestamped), per-tool usage (calls, errors, mean latency, cost), loop detection (repeated tool+arguments flagged as cycles), and the cost timeline (where did the turns go?).
- Key question: "Why is the optional trace backend a self-hosted, open-source one?" Expected wrong answer: "features". Right answer: "independence and data custody — a self-hosted backend keeps agent traces (which contain prompts and tool outputs) inside your own infrastructure."
- The $ figures on the diagram are illustrative — on local Ollama every call prices at $0, so watch turns and tokens.
- "If behind schedule": skip the across-runs view; loop detection is the highest-value single capability.
  **Transition**: "Lens 4 reads what an agent DID. Lens 5 reads whether training TAUGHT it the right thing."

---

## Slide 15: Lens 4: Agent Trace — Code

**Time**: ~2 min
**Talking points**:

- Hook: "`capture_run` runs the agent for you and records every event. No bespoke logging."
- Point at the ToolRegistry line: tools must live in a ToolRegistry — a bare list of functions registers nothing. The Delegate is built with `make_delegate(tools=…, max_turns=8)`; the model comes from OLLAMA_CHAT_MODEL.
- Walk the three reads: `tool_usage(run_id)` (per-tool calls, errors, latency, cost), `detect_loops(run_id)` (repeated signatures), `plot_trace` / `cost_breakdown` / `report`.
- Key question: "Why look at detect_loops when the agent already returned an answer?" Expected wrong answer: "it finished, so it is fine". Right answer: "a run can finish AND waste most of its turns repeating the same tool call — the trace tells you where the turns went, the answer does not."
- "If behind schedule": skip cost_breakdown; it recurs in 6.5.
  **Transition**: "Now the lens for training: did our fine-tuning reward the right thing?"

---

## Slide 16: Lens 5: Alignment — Is The Training Signal Honest?

**Time**: ~3 min
**Talking points**:

- Hook: "Reward goes up. Held-out quality goes down. KL spikes. That is the reward hacking signature — three curves tell you in one glance."
- Walk the reads: KL from the frozen reference, DPO reward margin distribution, bias-mitigated win rate (position swap + length normalisation), benchmark deltas before-vs-after, and the reward-hacking scan (high proxy score + failed held-out rubric).
- Key question: "If reward climbs AND held-out climbs AND KL climbs, is that hacking?" Expected wrong answer: "yes, KL is high". Right answer: "no — that is successful alignment. KL alone is not a failure signal; it is a signal to CHECK held-out. Pattern, not single number."
- GRPO extension: group-relative reward variance and group-collapse detection — GRPO's failure shape is unlike PPO's.
- "If behind schedule": skip GRPO — reward margin + KL + bench delta is the core triad.
  **Transition**: "Lens 6 is organisational: policy, envelope, audit chain."

---

## Slide 17: Lens 6: Governance — PACT Audit

**Time**: ~3 min
**Talking points**:

- Hook: "Every allowed action AND every denied action appends to the same hash-linked log. A broken chain means tampering — and you can name the exact index."
- Walk the reads: the five-dimension envelope snapshot per role, budget consumption against each dimension, audit chain verification, D/T/R addressing on every verdict, and the negative verdict drill (assert the DENY set stays denied).
- Key question: "Why audit DENIALS, not just executions?" Expected wrong answer: "completeness". Right answer: "the pattern of denials is how you detect an agent that is probing its envelope — systematic deny-then-retry patterns are a leading indicator of compromise or mis-aligned optimisation."
- Caveat to state now, plainly: in the installed PACT (0.14.x) a role with NO envelope — or an address not in the org — is auto-approved. The "no" branch only exists for roles whose envelope you have attached. We prove this in 6.7.
- Native to Kailash: zero external dependencies; the lens inspects GovernanceEngine, not a third-party policy engine.
- "If behind schedule": skip D/T/R; envelope + chain is the headline.
  **Transition**: "Six lenses, six responsibilities. Here is the modern stack that powers each."

---

## Slide 18: Modern LLM Observability Stack (2026)

**Time**: ~2 min
**Talking points**:

- Hook: "The observatory wraps these behind one object, so you rarely import them directly. None of them is a mystery SDK — they are open tools the LLM engineering profession runs on."
- Walk the table by layer: Output eval (Kaizen judge, RAGAS), interpretability (TransformerLens, SAELens + Gemma Scope), retrieval eval (RAGAS, TruLens), agent trace (self-hosted, open-source backends), alignment (TRL DPO/GRPO, lm-evaluation-harness), governance (PACT, Foundation-native).
- Key question: "Why does the Output row list TWO tools (a Kaizen judge AND RAGAS)?" Expected wrong answer: "redundancy". Right answer: "the judge handles open-ended criteria on any response; RAGAS is RAG-specific (faithfulness to retrieved context). Different specialists. The observatory composes, it does not pick winners."
- Why this stack: every tool is open-source or Foundation-native and runs on your own machine — traces, prompts and judge calls never leave it. Independence is a design constraint, not a suggestion.
- "If behind schedule": skip the callout; the table is the artefact.
  **Transition**: "Now for the map: which lens powers which lesson."

---

## Slide 19: How the Observatory Threads Through M6

**Time**: ~2 min
**Talking points**:

- Hook: "The six lenses stay the same for the next eight lessons. What changes is which lens takes the lead."
- Walk two or three rows: 6.1 leads Output (Attention preview only), 6.4 leads Retrieval with Output on faithfulness, 6.7 leads Governance with Agent Trace on deny patterns. The capstone uses ALL SIX.
- Key question: "Why does 6.1's attention lens say 'preview only'?" Expected wrong answer: "too advanced". Right answer: "the chat model is reached through Ollama's HTTP API, which exposes text, not activations. The attention lens needs the weights loaded in-process (LLMObservatory(attention_model='google/gemma-2-2b')) — an optional, advanced extension."
- Every lesson's first exercise begins with: "Run the observatory. Report the relevant lens readings. Then build."
- "If behind schedule": skim 6.2–6.7; the headline is that the capstone uses ALL SIX.
  **Transition**: "Last slide of the section: the protocol you run before every M6 build."

---

## Slide 20: Your Observatory Workflow

**Time**: ~5 min (3 min walk-through + 2 min sticky-note commitment)
**Talking points**:

- Hook: "Five checks, one decision. Every LLM system, every iteration, for the rest of M6."
- Walk the protocol: 10 probe prompts (smoke test) → Output lens (faithfulness + judge) → Retrieval lens if RAG (recall@5 + context utilisation) → Agent Trace if agentic (loop detection + cost) → Governance if PACT wired (verify_chain + envelope). All pass → full eval. Any fail → open that lens's deep-dive; change ONE thing; re-run.
- Key question: "Why five checks and not one omnibus eval?" Expected wrong answer: "thoroughness". Right answer: "each lens surfaces a DIFFERENT failure shape — an omnibus number averages them, so a passing omnibus hides a governance breach behind a strong output score."
- Cardinal rule: one change per iteration. Change prompt + retriever + judge model simultaneously and you cannot attribute the fix.
- "If behind schedule": cut the walk-through, run ONLY the sticky-note activity. Students leave with the five-step protocol in their own handwriting.
  **Transition**: "Put your sticky note next to your laptop. Now, Lesson 6.1 — prompt engineering. First exercise: run the observatory on your first prompted system."

---

## Slide 21: Lesson 6.1: LLM Fundamentals, Prompt Engineering & Structured Output

**Time**: ~1 min
**Talking points**:

- State the four learning objectives on screen. Read them aloud.
- "This lesson covers the most immediately practical LLM skill: getting good output from a model. We start with how they work, then focus on prompt engineering, then structured output with Kaizen Delegate."
- Prerequisite check: "Everyone completed M5? The transformer architecture is assumed knowledge. If anything feels unfamiliar, flag it at the break."
  **Transition**: "Let us start with how LLMs learn in the first place."

---

## Slide 22: How LLMs Learn: Pre-training

**Time**: ~3 min
**Talking points**:

- Explain the GPT-style objective: "Given a sequence of tokens, predict the next one. That is literally all it does during pre-training. Yet from this simple objective, it learns grammar, facts, and reasoning."
- Walk through the loss function slowly. Point at the summation: "We are minimising the negative log probability of the correct next token across the whole corpus."
- Contrast with BERT: masked language modelling is bidirectional — better for classification, worse for generation.
- "Pre-training does not teach the model to follow instructions. It teaches it to complete text. Alignment is what makes it helpful — we get to that in 6.3."
- "If beginners look confused": "Imagine being given a sentence with the last word missing, over and over, trillions of times. You would get very good at language. That is what the model is doing."
- "If experts look bored": "The cross-entropy form is standard; the interesting bit is the data scale. Chinchilla argues the field over-prioritised parameters at the expense of data — we see that next."
  **Transition**: "How big should these models be, and how much data do they need?"

---

## Slide 23: Scaling Laws: Parameters, Data, Compute

**Time**: ~3 min
**Talking points**:

- Core message: "Bigger models plus more data equals better performance, and the relationship is predictable."
- Explain Kaplan et al. (2020): loss falls as a power law in parameters, data and compute. Chinchilla (Hoffmann et al., 2022) corrected the allocation: "Many early models were undertrained — too many parameters, not enough data. Scale parameters and tokens roughly equally."
- Walk the formula on screen: L(N, D) ≈ E + A/N^α + B/D^β. "The E term is the irreducible loss — more parameters and data shrink only the last two terms. Loss never goes to zero. This is the Chinchilla form."
- Walk through the notable models table briefly. Do not dwell. Point to the diversity: Mixture of Experts, constitutional training, open weights, sliding window attention, knowledge distillation, high-quality data.
- Be precise about GPT-4: "Parameter count is undisclosed — reported at roughly 1.8T in a mixture-of-experts configuration, but not confirmed." Never state it as fact.
- "If beginners look confused": "You do not need to memorise these. The point is the landscape moves fast. The engineering patterns we teach today apply to every model in this table."
- "If experts look bored": "Ask yourselves: is Phi-3 at 3.8B the most important data point here? Data quality beats parameter count, and the whole curve bends with it."
  **Transition**: "Pre-training gives us a fluent model. But fluency is not helpfulness. How do we make it actually useful?"

---

## Slide 24: From Prediction to Helpfulness: RLHF

**Time**: ~3 min
**Talking points**:

- Walk through the flow: pre-trained LLM, supervised fine-tuning (SFT), reward model training, PPO against the reward model, aligned LLM.
- "Pre-training optimises prediction. SFT teaches format — how to follow instructions. RLHF teaches quality — how to be helpful, harmless, and honest."
- Flag the complexity: two models in memory, PPO instability, reward model drift. "This is why only big labs did RLHF for years."
- Connect to M5: "Remember PPO from M5? It is used here to optimise the LLM against a reward model. In Lesson 6.3 we will see DPO, which skips the reward model entirely."
- "If beginners look confused": "Think of RLHF as teaching the model what 'good' looks like by showing it examples of human preferences."
- "If experts look bored": "The interesting question is whether the reward model is learning the right signal. DPO argues you can skip that learning step and go directly from preferences to policy."
  **Transition**: "Before we get to alignment, let us master the cheapest and most powerful LLM skill: prompt engineering."

---

## Slide 25: Prompt Engineering: Zero-shot & Few-shot

**Time**: ~3 min
**Talking points**:

- Walk through zero-shot: "Task description only, no examples. Works well for simple, well-defined tasks."
- Walk through few-shot: "Provide a couple of examples to set the pattern. Works much better for structured or domain-specific tasks."
- "Few-shot is in-context learning: the model infers the task from examples without any weight updates. No training, just examples in the prompt."
- If you can do a live demo, run both prompts against the same input. Show the difference.
- "If beginners look confused": "Zero-shot is asking someone to do a task with just instructions. Few-shot is showing them examples first. Most of us perform better with examples — so do LLMs."
- "If experts look bored": "The in-context learning phenomenon is still not fully understood theoretically. There are several competing hypotheses in the literature."
  **Transition**: "For reasoning tasks, there is a single phrase that changes everything."

---

## Slide 26: Chain-of-Thought Prompting

**Time**: ~3 min
**Talking points**:

- Walk through standard CoT: few-shot with reasoning included in each example.
- Walk through zero-shot CoT: just append "Let's think step by step" — no examples needed.
- Quote the Kojima et al. (2022) result precisely: "Adding the five words 'Let's think step by step' lifted zero-shot accuracy on MultiArith from 17.7% to 78.7%, and on GSM8K from 10.4% to 40.7% (text-davinci-002)." Do not swap the benchmarks — MultiArith is the dramatic one.
- "CoT forces the model to show its work, which prevents it from jumping to wrong conclusions."
- "If beginners look confused": "It is like asking a student to show their working on a maths exam. They get partial credit — and they also get the right answer more often."
- "If experts look bored": "The interesting theoretical question: is CoT activating a latent reasoning capability, or is it just spreading computation across more tokens? The answer seems to be both."
  **Transition**: "CoT is good. Self-consistency on top of CoT is better."

---

## Slide 27: Self-Consistency & Structured Prompting

**Time**: ~3 min
**Talking points**:

- Self-consistency: "Sample multiple CoT paths with temperature > 0, take the majority vote. More robust than a single CoT call because it averages out noise."
- Walk through the code: five samples, extract answer from each, majority vote. Point out the fresh `make_delegate(temperature=0.7)` per sample — fresh delegate, independent sample.
- Structured prompting: "In production, free-form text is unparseable. You need to specify a schema — JSON, a table, typed fields."
- "Kaizen Signature enforces this at the framework level, so you do not have to parse LLM output with regex. We get to Signature on the next slide."
- "If beginners look confused": "Self-consistency is like asking the same question to a room full of experts and taking the majority answer. Each individual might be wrong sometimes, but the majority is usually right."
- "If experts look bored": "Self-consistency is strictly better than CoT on every reasoning benchmark, but it multiplies the token count by N. The engineering tradeoff is when the stakes justify the spend."
  **Transition**: "Let us put all of these techniques on one slide."

---

## Slide 28: Prompt Engineering: The Complete Toolkit

**Time**: ~2 min
**Talking points**:

- Walk through the table once. Do not read every row.
- The gain column is a rough, task-dependent illustration — Exercise 6.1 measures the real numbers on SST-2 with the course model.
- Emphasise the engineering rule at the bottom: "Start with zero-shot. Add few-shot if accuracy is insufficient. Add CoT if reasoning is needed. Add self-consistency only if the cost is justified by the stakes."
- "Self-consistency multiplies your token usage by N. On local Ollama that is compute time; on a priced provider it is money. Use it when you are making a high-stakes decision, not when you are classifying news articles."
- "If beginners look confused": "This is the decision tree. You never need all of them at once. Start simple, escalate when needed."
- "If experts look bored": "The accuracy gains in the table are median values from benchmarks. Your mileage will vary by task. The point is the ordering, not the exact numbers."
  **Transition**: "Now let us see how Kaizen wraps all of this in a type-safe interface."

---

## Slide 29: Kaizen: Delegate & Signature

**Time**: ~3 min
**Talking points**:

- Walk through the code. A Signature class defines the inputs and outputs as typed fields. The agent renders the schema into the prompt and parses the reply into typed fields.
- Point at the config: a local Ollama model named by OLLAMA_CHAT_MODEL (never a hard-coded hosted model), `use_async_llm=True` (required for `run_async`), and the JSON structured-output mode.
- "This is the Kailash bridge: everything we just learned about prompting is wrapped in a type-safe interface. No more parsing JSON from raw text."
- On accounting: "Students should always know what each call consumes. On local Ollama that is tokens and seconds, not dollars — a dollar cap such as budget_limit_usd only bites on a priced provider."
- "A missing field is a real failure — check for it, never fill a placeholder. If the LLM fails to return the schema, that is a bug to surface, not to paper over."
- "If beginners look confused": "Think of Signature as a form. The LLM has to fill in the fields. If it does not, you find out immediately."
- "If experts look bored": "The Delegate supports streaming, structured decoding, function calling, and cost tracking out of the box. It is a production wrapper around the raw API."
  **Transition**: "A quick aside on inference optimisation, then we get to the exercise."

---

## Slide 30: Inference Optimisation (Brief)

**Time**: ~2 min
**Talking points**:

- Three optimisations everyone should know: KV-cache (caches previously computed tokens), speculative decoding (draft model proposes, large model verifies), continuous batching (requests join the batch as slots free up).
- "You will not implement these yourself. But understanding them explains why some APIs are faster than others, and why batch pricing exists."
- "Lesson 6.8 covers vLLM, which is the production serving framework that bundles all of these."
- "If beginners look confused": "Skip the details. The point is that calling an LLM API is not the same as the API running fast. There is a lot of engineering behind the scenes."
- "If experts look bored": "Flash Attention is the one missing from this slide; it shows up in 6.8. PagedAttention is the interesting vLLM contribution."
  **Transition**: "Time for the first exercise."

---

## Slide 31: Exercise 6.1: Prompt Engineering Showdown

**Time**: ~2 min
**Talking points**:

- Walk through the exercise skeleton. Six technique files (ex_1/01–06): zero-shot, few-shot, CoT, zero-shot CoT, self-consistency, and Signature structured output — all on the same SST-2 eval docs, all through a Kaizen Delegate on local Ollama.
- "Measure accuracy, tokens and latency for each technique. On local Ollama the dollar cost is zero, so the exercise reports tokens and converts them at one illustrative hosted price — that accounting is the engineering differentiator: this is not a toy exercise, it is a production evaluation."
- "Files 02–05 change only the prompt; file 06 swaps in a Signature. The table at the end is the deliverable."
- "If beginners look confused": "You only need to write small prompt functions. The shared helpers and the Delegate do the heavy lifting."
- "If experts look bored": "Stretch task: add the observatory's Output lens to your comparison — faithfulness and judge scores on the same docs."
  **Transition**: "Lesson 6.2. Fine-tuning. This is the mathematical heart of the module."

---

## Slide 32: Lesson 6.2: LLM Fine-tuning — LoRA, Adapters & the Technique Landscape

**Time**: ~1 min
**Talking points**:

- Read the learning objectives. Five outcomes, two from-scratch implementations.
- "This lesson is the most mathematically dense in the module. If you only follow the green foundations slides here, that is fine — you will still come out knowing what LoRA and adapters are and when to use each."
- Connect to M4: "Remember SVD from M4.3? LoRA is literally SVD applied to weight updates. If you followed that, you will follow this."
  **Transition**: "Let us start with LoRA theory."

---

## Slide 33: LoRA: Low-Rank Adaptation

**Time**: ~4 min
**Talking points**:

- Core idea: "Pre-trained weights W_0 are frozen. Instead of updating all d by d parameters, we learn two small matrices B (d by r) and A (r by d). Rank r is much smaller than d — typically 4, 8, or 16."
- Walk through the forward equation: h = W_0 x + BA x. "The original output plus a low-rank adaptation."
- Point at the parameter savings box: "For d=4096 and r=8, the full update is 16.7 million parameters. LoRA is 65 thousand. That is a 256x reduction."
- M4 connection: "LoRA IS SVD applied to weight updates. The low-rank matrices B and A approximate the full update delta-W just like truncated SVD approximates a matrix."
- "If beginners look confused": "Think of it as summarising a 1000-page book into a 4-page summary. You lose some detail but capture the essence. And you only need to store the 4-page summary, not the 1000 pages."
- "If experts look bored": "The interesting question is the rank selection. Ranks 4 to 16 are typical, but the optimal rank depends on the task. For some tasks, rank 1 is enough."
  **Transition**: "Now let us implement it from scratch."

---

## Slide 34: LoRA: From-Scratch Implementation

**Time**: ~4 min
**Talking points**:

- Walk through every line of the LoRALayer class.
- Key points: the original linear layer has requires_grad=False — it is frozen. Only A and B are trainable.
- Initialisation matters: A is initialised with small random values, B with zeros. "At initialisation, B times A equals zero, so the model starts as the original pre-trained model. Training gradually learns the adaptation. This is why LoRA is safe: you cannot make the model worse at the start."
- Forward pass: compute original + adaptation, add them.
- "If beginners look confused": "The original model does its thing. The LoRA layer adds a small correction on top. At the start, the correction is zero."
- "If experts look bored": "Notice how the adaptation is computed as (x @ A.T) @ B.T. That is the order that matters for efficiency — never materialise the full delta-W."
  **Transition**: "LoRA was not the first parameter-efficient method. Adapters came before."

---

## Slide 35: Adapter Layers: Bottleneck Modules

**Time**: ~3 min
**Talking points**:

- Architecture: down-project to a bottleneck, apply a non-linearity, up-project back. Residual connection around the whole thing.
- Historical note: "Adapters predate LoRA — Houlsby et al. 2019. Same core idea: freeze the big model, train small additions."
- The bottleneck forces the adapter to learn a compressed representation of the task-specific knowledge.
- Init trick: up-projection weights and bias are initialised to zero. "Just like LoRA's B=0 trick, this means the adapter output is zero at init. Training starts from the original behaviour."
- "If beginners look confused": "Think of adapters as tiny specialist modules plugged into the side of the big model."
- "If experts look bored": "Adapters are sequential — they add computation to every forward pass. That is their main disadvantage vs LoRA, which we cover on the next slide."
  **Transition**: "Head-to-head comparison."

---

## Slide 36: LoRA vs Adapters: Head-to-Head

**Time**: ~2 min
**Talking points**:

- Walk through the table quickly. The key differentiator is inference latency: LoRA can be merged into the base weights after training, so inference is exactly the same speed as the original model. Adapters add sequential computation at every forward pass.
- "In production, LoRA dominates because merged weights mean zero inference overhead. Adapters remain useful when you need to swap between many tasks without reloading the model."
- Modularity trade-off: adapters are separate modules you can hot-swap. LoRA can be merged (fast) or left separate (flexible).
- "If beginners look confused": "LoRA is the default. Use adapters only if you need to swap between many tasks at serving time."
- "If experts look bored": "The inference-latency argument is why LoRA won the production battle. But for multi-task serving with many small tasks, adapters can still win."
  **Transition**: "LoRA and adapters are just two of ten techniques. Let us look at the full landscape."

---

## Slide 37: The Fine-tuning Landscape: 10 Techniques

**Time**: ~4 min
**Talking points**:

- Do not deep-dive every technique. Walk through the table once.
- Highlights: Prefix Tuning (K/V prefix vectors), Prompt Tuning (learnable soft prompts), LLRD (lower learning rate for earlier layers — they are more general), Progressive Freezing (unfreeze top-down), Distillation (teacher-student), DP-SGD (gradient noise for privacy), EWC (Fisher Information to prevent catastrophic forgetting).
- Point at the decision rule: "Start with LoRA. Full fine-tuning only if the domain is very different. Distillation for constrained hardware. DP-SGD when privacy is a legal requirement."
- "If beginners look confused": "You do not need to memorise these. The point is: LoRA is almost always the right starting point. Only escalate when you have a specific reason."
- "If experts look bored": "EWC is the interesting one for continual learning scenarios. The Fisher Information penalty prevents the model from forgetting previous tasks when you add new ones."
  **Transition**: "Once you have multiple fine-tuned models, you can combine them."

---

## Slide 38: Model Merging: Combining Fine-tuned Models

**Time**: ~3 min
**Talking points**:

- Four techniques: TIES (trim, elect sign, merge), DARE (drop and rescale), SLERP (spherical linear interpolation), Task Arithmetic (add/subtract task vectors).
- Core insight: "Fine-tuned weights form 'task vectors' that can be added and subtracted like arithmetic. You can combine skills without retraining."
- Application: "Train separate LoRA adapters for sentiment analysis and summarisation. Merge them with TIES to get a model that does both — without retraining."
- "If beginners look confused": "Think of it as mixing paint colours. Each adapter is a colour; merging creates a new shade."
- "If experts look bored": "Task arithmetic is the most theoretically interesting: subtract a bad behaviour vector to remove a capability, or add a good behaviour vector to gain one. The linear algebra of fine-tuning."
  **Transition**: "Brief aside on running these big models on small hardware."

---

## Slide 39: Quantisation: Running Large Models on Small Hardware

**Time**: ~2 min
**Talking points**:

- Four methods: GPTQ, AWQ, GGUF, bitsandbytes. All reduce precision from 16-bit or 32-bit to 4-bit or 8-bit.
- QLoRA is the practical combination: "Quantise the base model to 4-bit NF4. Add LoRA adapters in full BF16. Train LoRA while the base stays quantised. Result: fine-tune a 65B model on a single 48GB GPU."
- Dettmers et al. (2023) showed QLoRA matches full 16-bit fine-tuning quality with 4x less memory.
- "If beginners look confused": "Quantisation is like compressing a photo. You lose a tiny bit of quality but the file is 4x smaller. QLoRA means you can fine-tune large models on consumer hardware."
- "If experts look bored": "The NF4 datatype is specifically designed for normally distributed weights. It quantises the outliers differently from the bulk. The math is elegant."
  **Transition**: "Now let us see how Kailash wraps all of this in one pipeline."

---

## Slide 40: kailash-align: Fine-tuning with AlignmentPipeline

**Time**: ~2 min
**Talking points**:

- Walk through the code. `AlignmentConfig(method="sft", base_model_id=…, lora=LoRAConfig(rank=8, alpha=16, target_modules=("q_proj","v_proj")), sft=SFTConfig(num_train_epochs=3, learning_rate=2e-4))`. The base model id comes from SFT_BASE_MODEL (course default Qwen/Qwen2.5-0.5B-Instruct) — never hardcoded.
- Three API facts to stress: (1) `train()` is async and REQUIRES an `adapter_name`; (2) the dataset is a HuggingFace Dataset with a "text" column — convert polars with `Dataset.from_dict(frame.to_dict(as_series=False))`; (3) losses come from `result.training_metrics`, not attributes on the result.
- "AdapterRegistry stores versioned adapters: `await AdapterRegistry().register_adapter(name=…, adapter_path=result.adapter_path, signature=AdapterSignature(...))`. The capstone loads from this registry."
- "The Kailash bridge: everything we learned about LoRA, adapters, and quantisation is wrapped in AlignmentPipeline. Students implement from scratch first (ex_2/01–02), then run this in ex_2/06."
- "If beginners look confused": "You do not start with this. You start with the from-scratch implementation. AlignmentPipeline is what you use once you understand what it is doing."
- "If experts look bored": "For scale intuition: rank-8 LoRA on q_proj and v_proj of an 8B model trains about 3.4 million parameters — roughly 0.04% of the model. The savings ratio is the point, not the absolute number."
  **Transition**: "Time for the second exercise."

---

## Slide 41: Exercise 6.2: From-Scratch Fine-tuning

**Time**: ~2 min
**Talking points**:

- Walk through the six files: LoRALayer from scratch with a rank sweep (ex_2/01), AdapterLayer from scratch and the comparison (02), the landscape decision tree (03), TIES and SLERP merging on synthetic task vectors (04), INT8 quantisation from scratch (05), and the full SFT + LoRA run on IMDB instruction pairs through AlignmentPipeline with a registered adapter (06).
- "The from-scratch implementations are the core. Merging (file 04) is required but works on synthetic task vectors — merging two TRAINED adapters end-to-end is an extension for advanced students."
- Flag the common mistake: forgetting to freeze the original weights. "If W_0.requires_grad is True, you are doing full fine-tuning, not LoRA. Check this."
- "If beginners look confused": "The first two tasks are the important ones. Get LoRA and Adapter working with the expected parameter counts. The pipeline run in 06 is mostly provided."
- "If experts look bored": "The merging file is where it gets interesting. TIES will give you different results from naive averaging. Measure and explain."
  **Transition**: "Lesson 6.3. Preference alignment. This is where we stop teaching the model what to say and start teaching it what is good."

---

## Slide 42: Lesson 6.3: Preference Alignment — DPO & GRPO

**Time**: ~1 min
**Talking points**:

- Read the four learning objectives.
- "DPO is the breakthrough: it removes the reward model from RLHF, making alignment accessible to anyone who can fine-tune. GRPO is the 2024–25 extension from DeepSeekMath — even more efficient on verifiable tasks."
- "This lesson is about making models not just capable, but aligned with human preferences."
  **Transition**: "Let us derive DPO."

---

## Slide 43: DPO: Direct Preference Optimization

**Time**: ~5 min
**Talking points**:

- The derivation sketch: "RLHF objective is maximise reward minus KL penalty to reference. Rafailov et al. (2023) showed the optimal policy has a closed-form solution. Rearranging, we can express the reward in terms of the policy directly. The reward model drops out of the equation entirely."
- Walk through the DPO loss term by term. Point at the log ratio pi/pi_ref: "This measures how much the policy has changed from the reference for each response."
- The sigma (sigmoid) converts the difference into a probability — the probability that the policy prefers the chosen response over the rejected one.
- Beta is the temperature, and get the direction right: "β is the KL-penalty strength. LARGE β keeps the policy close to the reference — conservative alignment. SMALL β is a weak anchor — the policy can drift further. If your general-capability benchmarks regress after DPO, the fix is to RAISE β, not lower it."
- Bradley-Terry model: "The assumption that preferences follow a logistic function of reward differences. This is where the sigmoid comes from."
- "If beginners look confused": "DPO says: given two answers, make the good one more likely and the bad one less likely, but do not change too much from the original model. That is it."
- "If experts look bored": "The key insight is that the reward model never existed in DPO — the policy IS the reward model, up to a constant. This is elegant and it is also why DPO is so much more stable than PPO."
  **Transition**: "Now the implementation."

---

## Slide 44: DPO: Training Loop

**Time**: ~3 min
**Talking points**:

- Walk through the code. The dpo_loss function is four lines of math. That is the entire algorithm.
- Key operational points: "You need two models in memory — the policy (which is trained) and the reference (which is frozen). You do NOT need a reward model."
- The ref model is typically the SFT checkpoint. "It anchors the training — the policy cannot wander too far."
- "In practice, you compute log probs by running both models on the chosen and rejected sequences, then plug into the loss."
- "If beginners look confused": "Four lines of math. No reward model. No PPO. No instability. That is why DPO spread so quickly."
- "If experts look bored": "The implementation is clean, but the memory cost is still 2x a single model. QLoRA fixes this by quantising the reference model."
  **Transition**: "DPO was 2023. Let us look at the 2024–25 evolution."

---

## Slide 45: GRPO: Group Relative Policy Optimization

**Time**: ~4 min
**Talking points**:

- Provenance, stated correctly: "GRPO was introduced by Shao et al. (2024) in the DeepSeekMath paper, and later used to train DeepSeek-R1 (2025)."
- Walk through the algorithm: sample G completions per prompt, score each with a verifier (not a learned reward model), compute advantage relative to the group, update with a PPO-style clipped objective plus a KL penalty to the reference. No value network.
- Walk through the advantage formula: "Reward minus group mean, divided by group standard deviation (plus epsilon). The std-normalisation is what makes it invariant to the reward's scale — mean-subtraction alone is not."
- Write the full objective on the board: J = E[ min(ρ_i Â_i, clip(ρ_i, 1±ε) Â_i) ] − β·KL(π_θ ‖ π_ref), with ρ_i = π_θ(y_i|x) / π_old(y_i|x). "The clip bounds each update; the KL term keeps the policy from collapsing."
- DPO vs GRPO table: "DPO uses preference pairs; GRPO uses single prompts with multiple completions. DPO is better for subjective quality; GRPO is better for objective tasks like code and math where you can verify correctness."
- "If beginners look confused": "Imagine giving 10 students the same maths problem, marking all their answers, then telling each student how they did compared to the group average. Students above average get positive feedback; students below get negative."
- "If experts look bored": "The verifier is the interesting part. For code, it is 'does it compile and pass the test suite'. For math, it is 'is the final answer correct'. For subjective tasks, you cannot use GRPO — that is when DPO shines."
  **Transition**: "Once you have a fine-tuned model, how do you evaluate it?"

---

## Slide 46: LLM-as-Judge Evaluation

**Time**: ~3 min
**Talking points**:

- The idea: use one LLM to rate another LLM's outputs. Walk through the judge prompt.
- Three known biases: position bias (whichever response is shown first gets a boost), verbosity bias (longer responses rated higher), self-enhancement (a model rates its own outputs higher).
- Mitigations are mechanical: swap positions and average, normalise by length, use a different model as judge.
- "Never trust a single judge call. Always swap positions, run multiple times, aggregate. LLM-as-judge is a statistical estimator, not a deterministic oracle."
- Course-specific discipline: "Count parse failures and ties separately — never score them as 'consistent'. A judge that always fails would otherwise look 100% consistent. Exercise 6.3 ships a swap-averaged judge for exactly this reason."
- "If beginners look confused": "It is like asking a human to rate two essays. If you always show essay A first, essay A gets a slight unfair advantage. So you flip the order and average."
- "If experts look bored": "MT-Bench uses a strong hosted model as judge. The self-enhancement bias is real and measurable — judge models tend to favour answers written in their own family's style. That is why the course judge is a different model from the one under test where practical."
  **Transition**: "LLM-as-judge is fast but biased. For standard benchmarks, we use established evaluation sets."

---

## Slide 47: Evaluation Benchmarks

**Time**: ~2 min
**Talking points**:

- Quick tour: MMLU (multi-task language understanding, 57 subjects), HellaSwag (commonsense), HumanEval (code), MT-Bench (LLM-as-judge multi-turn), GSM8K (grade-school math).
- lm-eval-harness is the unified framework. One command, many benchmarks.
- Critical concept: alignment tax. "Run benchmarks before and after alignment. If MMLU drops more than 2%, your alignment is degrading general capability. You have paid too much for the alignment you gained."
- "If beginners look confused": "Benchmarks are standardised tests for LLMs. They tell you what capabilities the model has."
- "If experts look bored": "lm-eval-harness supports 200+ benchmarks. The interesting question is benchmark contamination — when the test set is in the training data."
  **Transition**: "Let us see how Kailash wraps DPO in one call."

---

## Slide 48: kailash-align: DPO Training

**Time**: ~2 min
**Talking points**:

- Walk through the code. `AlignmentConfig(method="dpo", base_model_id=…, lora=LoRAConfig(rank=16, alpha=32, …), dpo=DPOConfig(beta=0.1, learning_rate=5e-5, num_train_epochs=2))`. Beta lives on DPOConfig, not on the top-level config.
- "The preference triples go in `preference_dataset`, not the positional dataset slot: `await pipeline.train(None, adapter_name=…, preference_dataset=prefs)` — and prefs is a HuggingFace Dataset with prompt/chosen/rejected columns."
- Be explicit about what the pipeline does NOT do: "The pipeline trains. It does NOT evaluate. `result.training_metrics` gives you the raw TRL metrics (train_loss, rewards/margins). Win rate (LLM-as-judge) and safety/benchmark checks are separate steps YOU run — ex_3/03 does the safety eval, ex_3/04 does the judge."
- On the learning rate: "DPO needs a much smaller LR than SFT — roughly 1e-6 for full fine-tuning, about 5e-5 with LoRA (what ex_3/03 uses), versus the 2e-4 LoRA-SFT default."
- "If beginners look confused": "You are not writing DPO from scratch in production. You write it once to understand, then you use the pipeline."
- "If experts look bored": "The interesting production pattern is tracking the reward margin from training_metrics alongside an LLM-judge win rate and a benchmark delta. You want margin and win rate up with the benchmark delta near zero."
  **Transition**: "Time for the third exercise."

---

## Slide 49: Exercise 6.3: Preference Alignment

**Time**: ~1 min
**Talking points**:

- Four files: load UltraFeedback Binarized preference pairs (ex_3/01), implement the DPO loss and sweep β (02), train a DPO adapter with AlignmentPipeline and compare base vs aligned refusal rates on adversarial prompts (03), and GRPO advantages plus LLM-as-judge bias tests with a swap-averaged judge (04).
- Say it plainly: "UltraFeedback's chosen/rejected labels are AI feedback — GPT-4 ratings, not human raters. Know what your preference data actually is."
- "Students must demonstrate that alignment changed behaviour (refusal rate up on the adversarial set) AND know how they would check it did not destroy general capability: lm-eval before and after is the recommended extension."
- "If beginners look confused": "The hardest part is measuring the judge biases. Start with position bias: run each evaluation twice, once with each order, and average."
- "If experts look bored": "Stretch: implement the bias correction as a post-hoc adjustment, then run a small lm-eval task (hellaswag, limit 50) before and after DPO and report the alignment tax."
  **Transition**: "Lesson 6.4. RAG. The most-deployed LLM pattern in production."

---

## Slide 50: Lesson 6.4: RAG Systems

**Time**: ~1 min
**Talking points**:

- Read the four learning objectives.
- "RAG is the most deployed LLM pattern in production. It solves the biggest LLM limitation: hallucination. By grounding the model in retrieved documents, we get factual, verifiable answers."
- Prerequisites: 6.1 prompting plus M4 Lesson 6 embeddings.
  **Transition**: "Let us start with the concept."

---

## Slide 51: RAG: Retrieval-Augmented Generation

**Time**: ~3 min
**Talking points**:

- Walk through the pipeline diagram. Offline: chunk documents, embed, index. Online: embed query, retrieve similar chunks, generate with query + chunks.
- Why RAG beats fine-tuning for knowledge: "LLMs have a knowledge cutoff. LLMs hallucinate. RAG grounds the model in actual documents. And updating documents is much cheaper than retraining weights."
- "RAG is an open-book exam for the LLM: it can look up the answer instead of relying on memory."
- "If beginners look confused": "Think of RAG as giving the AI a reference book. Instead of guessing, it looks up the answer. And you can update the reference book without retraining the AI."
- "If experts look bored": "The interesting question is when to RAG vs when to fine-tune. Rule of thumb: RAG for facts, fine-tune for style. RAG for updatable knowledge, fine-tune for fixed capabilities."
  **Transition**: "The most underrated decision in RAG: chunking."

---

## Slide 52: Chunking Strategies

**Time**: ~3 min
**Talking points**:

- Four strategies: fixed-size (simple, breaks mid-sentence), sentence (coherent but variable), paragraph (preserves context but can be large), semantic (expensive but meaning-preserving).
- Overlap: 10-20% overlap between chunks. "Without overlap, a fact split across two chunks is lost to both."
- Chunk size: 256-512 tokens is the sweet spot for most use cases. Smaller means precise but noisy; larger means more context but less precise.
- "If beginners look confused": "Imagine cutting a textbook into note cards. Too small and each card is useless. Too big and you cannot find what you need. Around 300-400 words per card is usually right."
- "If experts look bored": "Semantic chunking is the current frontier. Embed every sentence, cluster by similarity, use cluster boundaries as chunk boundaries. Slow but it produces the best retrieval."
  **Transition**: "Once you have chunks, how do you find the relevant ones?"

---

## Slide 53: Retrieval: Dense, Sparse, and Hybrid

**Time**: ~4 min
**Talking points**:

- Three methods:
  - Dense: embed query and documents, cosine similarity. Captures semantic meaning. The course embedder is nomic-embed-text via Ollama (`make_embedder()`, 768-dim).
  - Sparse (BM25): term frequency + inverse document frequency. Exact keyword matching, fast, no GPU, interpretable. Students build BM25 from scratch in Exercise 6.4 (ex_4/03).
  - Hybrid: reciprocal rank fusion combining both. Best of both worlds.
- "In practice, hybrid almost always wins. Dense captures 'what you mean', sparse captures 'what you say', and some queries need both."
- Walk through the RRF code briefly. "It is a one-liner: 1/(k+rank) summed across methods. RRF fuses RANKS, so BM25 and cosine scores never need a common scale."
- "If beginners look confused": "Dense is like asking a librarian who understands your topic. Sparse is like searching the index at the back of the book. Hybrid is doing both and combining the results."
- "If experts look bored": "The reciprocal rank fusion k=60 is a magic number from the IR literature. It is surprisingly robust across domains."
  **Transition**: "Two advanced techniques: re-ranking and HyDE."

---

## Slide 54: Re-ranking & HyDE

**Time**: ~3 min
**Talking points**:

- Re-ranking: "Two stages. First stage is fast but approximate — bi-encoder retrieval. Second stage is slow but accurate — cross-encoder re-ranking. The cross-encoder sees query and document together, so it scores relevance much better."
- "This is the standard production pattern: retrieve 50 candidates fast, re-rank to top 3 accurately."
- Be precise about the course variant: "The slide shows a dedicated cross-encoder model; Exercise 6.4 (ex_4/05) uses the local LLM as a stand-in cross-encoder — score each query–passage pair 0–10 — so no extra model download is needed."
- HyDE: "Query and document are in different 'language spaces'. A question looks different from an answer. So we ask the LLM to generate a hypothetical answer, and we embed that. The hypothetical answer does not need to be correct — it just needs to be in the right neighbourhood."
- "If beginners look confused": "Re-ranking is like having a fast librarian grab a stack of candidates, then a slow expert look through the stack carefully. HyDE is more surprising — you ask the AI to guess an answer, then use the guess to find real answers."
- "If experts look bored": "HyDE works because embedding distances are shorter between answers than between questions and answers. The hypothetical answer bridges the gap."
  **Transition**: "How do you know if your RAG system is actually good?"

---

## Slide 55: RAGAS: RAG Evaluation Framework

**Time**: ~3 min
**Talking points**:

- Four metrics: Faithfulness (is the answer supported by the context?), Answer Relevance (does it address the question?), Context Relevance (are the retrieved chunks relevant?), Context Recall (did we retrieve everything needed?).
- Faithfulness is the most important: "If the answer is not grounded in the retrieved documents, you have a hallucinating RAG system. Faithfulness is the hallucination detector."
- Be precise about the course implementation: "The course computes the four metrics as LLM-as-judge prompts on local Ollama (`compute_ragas_metrics` in ex_4/05) — the ragas package's own metrics need extra provider dependencies that are not part of the course environment. Same metric definitions, judge-based scoring."
- One discipline: "Judge parse failures must RAISE, never default to a score. A default score of 5 out of 10 hides a broken judge behind a plausible number."
- "If beginners look confused": "Faithfulness asks: did the AI make this up, or did it actually find it in the documents? That is the only question that matters for RAG."
- "If experts look bored": "The four metrics are two pairs: quality of retrieval (context relevance, context recall) and quality of generation given retrieval (faithfulness, answer relevance). Diagnose failures by separating the two."
  **Transition**: "Beyond the one-shot pipeline: three patterns that fix the commonest RAG failures."

---

## Slide 56: Advanced RAG Patterns & Kaizen RAG Agents

**Time**: ~3 min
**Talking points**:

- Two halves. Left: three patterns that fix the commonest RAG failures.
  - Metadata filtering: "Filter chunks by source, date or department BEFORE similarity search — precision you cannot get from embeddings alone. 'Only notices issued after 2024' is a WHERE clause, not an embedding problem."
  - Multi-hop retrieval: "Retrieve, read, form a follow-up query, retrieve again — for questions whose answer spans two documents (HotpotQA-style, used in Exercise 6.5). Multi-hop is where RAG meets agents: the follow-up query is a reasoning step, which is why 6.5 builds it as a ReAct loop."
  - Document summarisation: "Index a summary per document alongside its chunks; route broad questions to summaries, narrow ones to chunks. Keeps long documents findable."
- Right: Kaizen ships ready-made agents for the two most common shapes. "RAGResearchAgent keeps its own small vector store and answers with sources and a confidence. MemoryAgent keeps per-session conversation history. Both run on the course's local Ollama model."
- Point out the trade-off: "The packaged agents are quick to start, but the hand-built pipeline from the previous slides is what you tune and evaluate. Exercise 6.4 builds the pipeline by hand so every stage is visible."
- "If beginners look confused": "Metadata filtering is like telling the librarian 'only the 2024 shelf' before you ask the question."
  **Transition**: "Time for the fourth exercise."

---

## Slide 57: Exercise 6.4: Build a RAG System

**Time**: ~1 min
**Talking points**:

- Five files: chunking strategy comparison (ex_4/01), dense retrieval with nomic-embed-text (02), BM25 from scratch (03), hybrid with RRF (04), and the LLM re-ranker + RAGAS-style judge metrics + HyDE + full pipeline (05).
- Describe the data accurately: "The corpus is a 1,000-document sample of the open neural-bridge/rag-dataset-12000 — context, question, answer triples, downloaded on first run and cached. Every question has a known source document, so hit@k is a real measurement, not a vibe."
- "The deliverable is the hit@k leaderboard: BM25 vs dense vs hybrid vs HyDE, same eval questions. Stress that HyDE is not guaranteed to win — students measure it on the same hit@k as dense and report what they find."
- "If beginners look confused": "Start with dense retrieval only. Get the pipeline end-to-end, then add BM25, then hybrid, then HyDE. Incremental is fine."
- "If experts look bored": "Stretch: experiment with chunking sizes on the same dataset. You will find that chunk size matters more than retrieval method."
  **Transition**: "Lesson 6.5. Agents. This is where LLMs become autonomous."

---

## Slide 58: Lesson 6.5: AI Agents — ReAct, Tool Use & Function Calling

**Time**: ~1 min
**Talking points**:

- Read the four learning objectives.
- "We move from passive LLM usage (prompting, fine-tuning, RAG) to active LLM usage: agents that reason, take actions, and observe results. This is where LLMs become autonomous."
- Prerequisite: 6.1 Kaizen Delegate. "You already know how to call an LLM. Now you will let the LLM decide what to do next."
  **Transition**: "What exactly is an agent?"

---

## Slide 59: What Is an AI Agent?

**Time**: ~3 min
**Talking points**:

- Core loop: observe, think, act, observe again. Repeat until the task is done or the budget is gone.
- "An agent is an LLM that can use tools. Instead of just generating text, it can search the web, run code, query databases, call APIs."
- Pipeline vs agent table: pipelines are deterministic with fixed steps; agents are dynamic and decide their own next step. Pipelines are fast; agents are flexible.
- Engineering rule: "If a pipeline can solve it, do not use an agent. Agents add complexity and cost. Use agents when the problem requires reasoning about which tools to use."
- "If beginners look confused": "A pipeline is like a recipe. An agent is like a chef who decides what to cook based on what is in the fridge. Recipes are fast and predictable; chefs are flexible and creative."
- "If experts look bored": "The interesting design question: how much agency to grant. Too little and you might as well use a pipeline. Too much and the agent wanders. The right answer is usually less than you think."
  **Transition**: "The ReAct pattern is the foundational agent loop."

---

## Slide 60: ReAct: Reasoning + Acting

**Time**: ~4 min
**Talking points**:

- Walk through the example trace step by step. Thought, Action, Observation, repeat. Each thought plans the next action; each observation updates the plan.
- "Without 'Thought' steps, agents make random tool calls. The reasoning step forces the model to plan before acting, which dramatically improves tool selection."
- Quote Yao et al. (ICLR 2023) precisely: "On HotpotQA, ReAct beat act-only prompting — 27.4 vs 25.7 exact match on PaLM-540B — and in their error analysis, hallucination drove most chain-of-thought failures but almost none of ReAct's. Reasoning constrains the action space."
- "The 'Thought' step is not overhead; it is the mechanism. It is also the debugging artifact — you can read the trace and see why the agent did what it did."
- "If beginners look confused": "It is like thinking out loud before doing something. You plan, then act, then check the result. Humans do this all the time; ReAct makes the LLM do it too."
- "If experts look bored": "The interesting thing is that pure act-only agents hallucinate MORE than ReAct agents. The reasoning step grounds the next action in the last observation."
  **Transition**: "The mechanism for actually doing things is function calling."

---

## Slide 61: Function Calling: Structured Tool Use

**Time**: ~3 min
**Talking points**:

- JSON schema defines a tool: name, description, parameters with types and descriptions, required fields.
- Tool choice parameter: auto (model decides), required (must call something), specific (must call a named tool).
- Parallel function calling: "Models can invoke multiple tools simultaneously when the calls are independent. Reduces latency for multi-tool queries."
- "Function calling is the bridge between LLM reasoning and real-world actions. The LLM generates a structured JSON call; your code executes it and returns the result."
- "If beginners look confused": "Think of it as the AI filling out a form to request an action, and your code processes the form. The schema is the form template."
- "If experts look bored": "Parallel function calling is a significant latency win for multi-step agents. The tradeoff is you lose sequential context between tool calls."
  **Transition**: "How do you design a good agent in the first place?"

---

## Slide 62: Agent Design: The Hiring Framework

**Time**: ~3 min
**Talking points**:

- Four questions: What is our goal? What is our thought process? What specialist would we hire? What tools do they need?
- "The key insight: vague agents produce vague results. Specific agents produce specific results."
- "You would not hire a 'general analyst'. You would hire a 'financial fraud investigator with experience in transaction pattern analysis'. The more specific your agent's role, the better it performs."
- Design considerations: iterative refinement (add a critic agent), human-in-the-loop (pause for validation on high-stakes decisions), monitoring (track intermediate outputs), and bounded runs (turn ceilings and, on priced providers, a dollar cap).
- "If beginners look confused": "Before building an agent, pretend you are writing a job description. The more precise the description, the better the candidate."
- "If experts look bored": "The sharpest heuristic is the specialist specificity. If you cannot name the agent's role in five words, the role is not specific enough."
  **Transition**: "Let us see how Kaizen implements all of this."

---

## Slide 63: Kaizen: Building Agents

**Time**: ~3 min
**Talking points**:

- Walk through the ToolRegistry code. Three API facts to stress:
  1. "Tools must be registered in a ToolRegistry with a name, description, JSON-schema parameters and an async executor — passing a plain list of functions silently registers NOTHING. This is the number-one course bug."
  2. "Build the Delegate with make_delegate(tools=…, max_turns=8) and NO model= argument — the model comes from OLLAMA_CHAT_MODEL."
  3. "max_turns is the hard ceiling on the Thought–Action loop. It works on every provider, including free local Ollama."
- "Students should notice the tools wrap a Kailash engine (DataExplorer). Wrapping TrainingPipeline works the same way, but its constructor needs a feature store and model registry (M3), so keep that tool for the capstone."
- Packaged agents: "kaizen_agents also ships ReActAgent and ChainOfThoughtAgent — step-by-step reasoning before answering, no tools needed. Both take llm_provider='ollama', and their run() is synchronous."
- "If beginners look confused": "You are wrapping functions you already know as tools. The agent decides WHEN to call them; you decide WHAT they do."
- "If experts look bored": "A Delegate with a ToolRegistry IS a ReAct loop — the same pattern from the previous slide, with the framework managing the Thought/Action/Observation cycle."
  **Transition**: "A word on why bounding agent spend is non-negotiable."

---

## Slide 64: Cost Budgets: Preventing Runaway Spending

**Time**: ~2 min
**Talking points**:

- The problem: "A confused agent can make hundreds of LLM calls before realising it is stuck. At an illustrative hosted price of $0.003 per call, 500 calls equals $1.50 per request. A production system with 10K daily users equals $15,000/day." The dollar figures are illustrative hosted prices; the SHAPE of the risk is real.
- Three ceilings, named precisely:
  - Turns — `max_turns`: the hard ceiling on loop length. Works on every provider, including local Ollama.
  - Dollars — `budget_limit_usd`: pass it AT CONSTRUCTION (`BaseAgentConfig(budget_limit_usd=0.50)` or `GovernedSupervisor(budget_usd=…)`). Setting it later changes the config object, not the enforcement. On local Ollama every call prices at $0, so this cap never trips here — configure it correctly anyway for when you move to a priced provider.
  - Tokens — reported per run; tokens × price = real spend.
- "Non-negotiable in production: every agent gets a turn ceiling, and on a priced provider a dollar cap set at construction. An unbounded agent is a financial liability."
- Course reality: "Exercise 6.5 (ex_5/02) bounds the loop with max_turns, reports tokens, and shows the dollar cap configured correctly — because on Ollama the dollar cap can never fire."
- "If beginners look confused": "It is like giving a contractor a budget. They can spend up to that amount, then they stop and report what they accomplished."
- "If experts look bored": "The interesting engineering is graceful degradation. A well-designed agent that hits its ceiling should return its best partial answer, not an error."
  **Transition**: "Time for the fifth exercise."

---

## Slide 65: Exercise 6.5: Build a Data Analysis Agent

**Time**: ~1 min
**Talking points**:

- Four files: a ReAct agent with four tools over HotpotQA multi-hop questions in a ToolRegistry, with the real trace captured and read (ex_5/01); bounded loops — max_turns ceilings, a dollar cap set at construction, token accounting (02); a structured agent — BaseAgent + Signature with typed, validated outputs (03); and a critic agent — Analyse → Critique → Refine (04).
- "The tools in ex_5/01 search and summarise a HotpotQA sample; the previous slide showed the same pattern wrapping a Kailash engine (DataExplorer), which students can try as an extension."
- "The reasoning chain is the key deliverable: students must demonstrate, from the captured trace, that the agent thinks before it acts."
- "If beginners look confused": "Start with one tool. Get the agent to call it correctly. Then add the others."
- "If experts look bored": "Stretch: add a critic agent that reviews the data scientist agent's output and suggests improvements. That is iterative refinement."
  **Transition**: "Lesson 6.6. One agent is good. Multiple specialists coordinating is better."

---

## Slide 66: Lesson 6.6: Multi-Agent Orchestration & MCP

**Time**: ~1 min
**Talking points**:

- Read the four learning objectives.
- "Single agents are powerful but limited. Real systems need multiple specialists working together: one for data, one for modelling, one for reporting. This lesson covers the coordination patterns."
- "MCP is the interoperability standard that makes all of this work across frameworks."
  **Transition**: "Four coordination patterns."

---

## Slide 67: Multi-Agent Patterns

**Time**: ~3 min
**Talking points**:

- Four patterns with concrete examples:
  - Supervisor-worker: "A project manager delegating to a data scientist and a report writer."
  - Sequential: "Data cleaning agent feeds feature engineering agent feeds model training agent."
  - Parallel: "Search agent and compute agent run at the same time, results aggregated."
  - Handoff: "Customer support bot transfers to a billing specialist when the topic changes."
- Decision rule: "Start with sequential (simplest). Add parallelism if sub-tasks are independent. Use supervisor-worker only when task decomposition is dynamic."
- Architecture checklist on the slide: modularity (one specialist = one Signature, swappable), load balancing (spread requests across agent replicas behind a router), dynamic agent creation (spawn a specialist per sub-task, inside a budget and depth limit), isolation (no data leaks between agents — that is 6.7).
- "If beginners look confused": "These are organisation charts for AI teams. Sequential is an assembly line. Parallel is a research lab. Supervisor-worker is a management hierarchy. Handoff is a customer service transfer."
- "If experts look bored": "The interesting question is when coordination overhead exceeds the parallelisation benefit. For small tasks, sequential always wins."
  **Transition**: "Let us implement supervisor-worker in Kaizen."

---

## Slide 68: Multi-Agent: Supervisor-Worker in Kaizen

**Time**: ~3 min
**Talking points**:

- Walk through the code. Three BaseAgent specialists (factual, semantic, structural), each with its OWN typed Signature, running on the course Ollama model. The supervisor is a fourth BaseAgent that reads three STRUCTURED outputs, not chat.
- "Here the supervisor's plan is fixed — all three specialists, then synthesis. In a dynamic supervisor the 'tools' it chooses between are other agents."
- "asyncio.gather makes fan-out latency roughly the SLOWEST specialist, not the sum. Exercise 6.6 part 3 measures this."
- Routing, stated precisely: "Kaizen also has Pipeline.router(), but it routes between BaseAgent instances — not Delegates — and its run() is synchronous. Exercise 6.6 (ex_6/03) builds an explicit LLM router instead and compares it against a keyword router."
- "If beginners look confused": "Think of the supervisor as an agent whose only tools are 'ask the factual specialist', 'ask the semantic specialist', 'ask the structural specialist'."
- "If experts look bored": "Recursive pattern: a supervisor can also be a specialist in a higher-level team. Every hand-off is a typed Signature output, which is what makes the audit trail readable."
  **Transition**: "For agents to remember things across sessions, they need memory."

---

## Slide 69: Agent Memory

**Time**: ~2 min
**Talking points**:

- Three memory types:
  - Short-term: current conversation context, in the LLM's context window, lost when the session ends.
  - Long-term: persistent knowledge across sessions, stored in a vector database, retrieved by semantic similarity.
  - Entity: structured knowledge about specific people/projects/datasets, key-value with entity extraction.
- Production pattern: "Short-term for the current task. Long-term for domain knowledge. Entity for user-specific context. Never rely on context window alone for production agents."
- "If beginners look confused": "Short-term is what you remember during a meeting. Long-term is what you write in your notes. Entity is your contacts list."
- "If experts look bored": "Entity memory is the underused one. Most production agents fail at 'who are you and what were we talking about last time' even though the fix is trivial."
  **Transition**: "Let us talk about the tool protocol that makes all of this interoperable."

---

## Slide 70: MCP: Model Context Protocol

**Time**: ~3 min
**Talking points**:

- What MCP is: "Standardised protocol for exposing tools to AI agents. Tool registration with JSON schemas. Transport via stdio or HTTP/SSE. Any agent from any framework can use your MCP server."
- Why MCP matters: "Without MCP, every agent framework has its own tool format. With MCP, one server, any client. Like REST APIs for humans, MCP is APIs for AI agents."
- Walk through the kailash_mcp code with the three API traps called out:
  1. "@server.tool() needs the parentheses — it is a decorator factory. Without them you bind your function to the wrong argument."
  2. "The transport goes in the CONSTRUCTOR: MCPServer(name='ml-tools', transport='stdio')."
  3. "server.run() takes NO arguments — it serves until the client disconnects."
- "The type hints and docstring become the tool's published JSON schema. Clients discover tools with the JSON-RPC tools/list method and invoke them with tools/call. In Exercise 6.6 (ex_6/04) a real MCP client launches the server over stdio, discovers the tools and calls them — including a tool that runs a specialist agent."
- "If beginners look confused": "Think of MCP as a USB port for AI tools. Any device that speaks USB can plug in. Any agent that speaks MCP can use your tools."
- "If experts look bored": "The interesting design tension is stateful vs stateless tools. MCP supports both, but stateful tools complicate the server implementation considerably."
  **Transition**: "MCP is how agents talk to tools. A2A is how agents talk to each other."

---

## Slide 71: A2A: Agent-to-Agent Communication

**Time**: ~2 min
**Talking points**:

- A2A: agents exchange typed messages, not free-form text. Agent Cards describe capabilities. Task lifecycle: submit, working, input-required, completed.
- Security concerns:
  - Data leakage: agents must not share data beyond their authorisation.
  - Prompt injection: one agent's output becomes another's input — sanitise.
  - Escalation attacks: Agent A asks Agent B to do something Agent A is not allowed to do.
- "Lesson 6.7 solves all three with PACT governance."
- "If beginners look confused": "When agents talk, every message is a potential trust boundary. You cannot assume the agent on the other side will protect your data."
- "If experts look bored": "The interesting question is whether agent-to-agent trust should be peer-to-peer or mediated by a central governance layer. PACT takes the second approach."
  **Transition**: "Time for the sixth exercise."

---

## Slide 72: Exercise 6.6: Multi-Agent ML Pipeline

**Time**: ~1 min
**Talking points**:

- Five files: supervisor-worker fan-out/fan-in on SQuAD 2.0 passages (ex_6/01), a sequential extract → interpret → synthesise pipeline (02), parallel execution plus an LLM router vs a keyword router (03), an MCP server with three typed tools called from a real MCP client (04), and short/long/entity memory plus multi-agent security (05).
- "The spec's DataScientist → FeatureEngineer → ModelSelector → ReportWriter chain is the same sequential pattern as ex_6/02 with ML-flavoured roles — a good extension for fast finishers."
- "Each agent stays within its specialisation. If the DataScientist starts writing reports, you have built a monolith, not a multi-agent system."
- "If beginners look confused": "Start with the sequential pipeline. Each agent has one job. Test that each one works alone before chaining."
- "If experts look bored": "Stretch: add a critic agent that reviews the ReportWriter's output and sends it back for revision if it fails quality checks."
  **Transition**: "Lesson 6.7. Governance. This is what separates toy projects from production."

---

## Slide 73: Lesson 6.7: AI Governance Engineering

**Time**: ~1 min
**Talking points**:

- Read the five learning objectives.
- Design principle: "This is ENGINEERING. Students implement access controls, test them, and verify they work. No philosophical discussion of AI ethics. The code IS the governance."
- "Governance without tests is governance theatre. If you cannot write a test that proves your access control works, it does not work."
  **Transition**: "PACT starts with addressing: who is asking for what."

---

## Slide 74: PACT: D/T/R Addressing

**Time**: ~3 min
**Talking points**:

- D/T/R, expanded once and never differently: Department / Team / Role. Never "Delegator/Task/Responsible", never "Decides/Trusts/Responds".
- Grammar: "A dash-delimited path. Every D or T MUST be immediately followed by exactly one R — its head role. D1-R1-T1-R1 reads: Department 1, its head Role R1, Team 1, that team's head Role R1."
- "Every agent, every API call, every data access has an address. The GovernanceEngine checks the address against the rules."
- Stress the two-step build: "GovernanceEngine(loaded.org_definition) compiles only the org STRUCTURE. The YAML's clearances and envelopes take effect only after apply_governance_specs — the course helper compile_governance() does both. Skip the second step and every verify_action call is auto-approved."
- The decision API: "verify_action is the single decision call. It returns a GovernanceVerdict with .allowed, .level and .reason. The four levels are auto_approved, flagged, held and blocked — allowed is True for auto_approved and flagged."
- "If beginners look confused": "D/T/R is like a postal address for permissions. The system checks your address to decide what you can access."
- "If experts look bored": "Every verdict carries a reason. A governance system that cannot explain its decisions is not a governance system — it is a random gate."
  **Transition**: "Once you can identify the requester, you need rules about what they can do."

---

## Slide 75: Operating Envelopes

**Time**: ~3 min
**Talking points**:

- A ConstraintEnvelopeConfig defines the boundaries of what a role can do: five canonical dimensions — Financial, Operational, Temporal, Data Access, Communication — plus a confidentiality clearance and a delegation-depth cap.
- Say the default out loud, plainly: "The installed PACT (0.14) is fail-OPEN for roles with no envelope and for addresses not in the org: verify_action auto-approves them with 'No envelope constraints — action permitted'. The envelope is what CREATES the deny path. Attach an envelope to every role with engine.set_role_envelope(...) before you rely on any denial."
- Monotonic tightening: "Child envelopes must be ≤ parent on every dimension. They can only add restrictions, never loosen. This prevents privilege escalation."
- One API trap: "RoleEnvelope.validate_tightening is KEYWORD-ONLY. Call it positionally and you get TypeError, not MonotonicTighteningError — and inside pytest.raises that TypeError escapes."
- "If beginners look confused": "An operating envelope is like a job description with hard limits. The agent can do anything within the envelope, but nothing outside it."
- "If experts look bored": "The lattice of envelopes forms a partial order under tightening. The framework catches violations at construction time via validate_tightening — structural, not runtime, enforcement."
  **Transition**: "Budgets are cost envelopes. They cascade."

---

## Slide 76: Budget Cascading

**Time**: ~2 min
**Talking points**:

- How it works: parent role's envelope caps spend (max_spend_usd=10.00); each child's cap must be ≤ the parent's — validate_tightening rejects a wider child; verify_action with a cost context blocks an action that exceeds the role's cap.
- "Two layers: the envelope CAPS, checked structurally and per action by PACT, and a running LEDGER of what each child has actually spent — Exercise 6.7 part 3 builds one: allocate, spend, overspend denied."
- "Reallocation — moving one child's unused allocation to another — is a ledger operation the supervisor performs; it can never lift a child above the parent's cap."
- Course reality: "On local Ollama the dollar amounts are notional — every call prices at $0 — so the exercise uses them as a teaching currency. The mechanics are identical on a priced provider."
- "If beginners look confused": "It is like a project manager splitting a budget across team members, then reallocating surplus from under-spenders to over-runners."
- "If experts look bored": "The interesting question is whether budgets should be hierarchical or flat. PACT uses hierarchical because it matches D/T/R structure."
  **Transition**: "Let us see the whole governance stack in one class."

---

## Slide 77: GovernedSupervisor: Governance Built In

**Time**: ~3 min
**Talking points**:

- Walk through the code. GovernedSupervisor is the governed agent entry point: three knobs — budget_usd, tools (the allow-list), and data_clearance — become its envelope.
- "It runs a two-layer contract: the supervisor PLANS the task, and YOUR execute_node callback runs the real LLM for each step, returning result, cost and token counts. It does NOT wrap an existing BaseAgent — the callback is where your model call lives."
- "The governance check happens before every step, and every step is written to its hash-chained audit trail. result.success, result.budget_consumed, result.audit_trail."
- On the verify_action line at the bottom: "This is the organisation-level check from the engine. It blocks here ONLY because compile_governance() attached the YAML envelopes — on a bare engine the same call is auto-approved. Same fact as the envelopes slide, now visible in code."
- "If beginners look confused": "You construct the supervisor with the envelope. Every action it tries to take is checked against the rules first. If it is not allowed, it is blocked."
- "If experts look bored": "The two-layer contract separates planning from execution: the same governed supervisor can drive a real LLM in production and an offline executor in tests — the envelope enforcement is identical."
  **Transition**: "And you must test this, or it does not exist."

---

## Slide 78: Governance Testing: Proving Safety

**Time**: ~3 min
**Talking points**:

- Testing imperative: "Governance without tests is governance theatre. Test that allowed actions succeed. Test that denied actions are blocked. Test that envelopes tighten correctly. Test that budget enforcement works."
- Note the fixture: "Every deny test runs on an engine whose roles HAVE envelopes — compile_governance() attaches them. On a bare engine 'analyst cannot delete' would FAIL, because pact 0.14 auto-approves envelope-less roles."
- The third test is deliberate: "test_unknown_address_is_auto_approved PINS the fail-open default. If a PACT upgrade changes the default to fail-closed, this test tells you on upgrade day — not in a post-incident review. Production code should reject unknown addresses itself before calling the engine."
- "If the test for 'analyst cannot delete data' fails, you have a security vulnerability. Red team your own governance."
- "If beginners look confused": "We write tests to prove the locks on the doors actually work. Pushing on a locked door is the test."
- "If experts look bored": "Property-based testing is the natural extension. Generate random envelopes and verify monotonic tightening as an invariant."
  **Transition**: "Governance also requires audit trails."

---

## Slide 79: Audit Trails & Clearance Levels

**Time**: ~2 min
**Talking points**:

- Audit trail: every access decision is logged with timestamp, requester, action, decision, reason. Immutable append-only, hash-chained; verify_chain() returns False the moment any record was altered. Required for regulatory compliance, incident investigation, accountability.
- Clearance levels — teach the ladder the way the installed PACT orders it, lowest to highest: PUBLIC < RESTRICTED < CONFIDENTIAL < SECRET < TOP_SECRET.
- The common trap, named explicitly: "Students read 'restricted' as the most secret level. In PACT it sits just above public — giving your most privileged role 'restricted' gives it LESS access than 'confidential'. 'internal' is an alias for restricted. The course org gives department heads SECRET, and an escalation test asks for secret or top_secret."
- "Clearance levels map to role addresses. A data analyst at D1-R1-T1-R1 carries restricted; a department head carries secret; a public support bot carries public."
- "If beginners look confused": "An audit trail is like CCTV for your AI system. It records every decision for review. If something goes wrong, you can reconstruct exactly what happened."
- "If experts look bored": "The interesting compliance question is retention policy. How long do you keep the audit trail? That is a legal question, not a technical one."
  **Transition**: "Time for the seventh exercise."

---

## Slide 80: Exercise 6.7: Governed Multi-Agent System

**Time**: ~1 min
**Talking points**:

- Four files: define the D/T/R org in YAML and compile it (ex_7/01); operating envelopes plus monotonic tightening (02); a budget cascading ledger plus ten allow/deny verify_action cases — envelopes attached FIRST (03); and GovernedSupervisor at three clearance tiers with deny paths and the audit chain (04).
- "The governance tests are the most important deliverable — and they only prove something if every deny case runs against a role that has an envelope attached. A deny test on an envelope-less role proves nothing: the default is auto-approve."
- "If beginners look confused": "Start with two roles and two rules. Write a test that proves rule 1 allows something. Write a test that proves rule 2 blocks something. Expand from there."
- "If experts look bored": "Stretch: combine with 6.6's multi-agent pipeline. Every agent in the pipeline should have its own envelope."
  **Transition**: "And now, the capstone. Lesson 6.8. Time to ship."

---

## Slide 81: Lesson 6.8: Capstone — Full Production Platform

**Time**: ~1 min
**Talking points**:

- Read the five learning objectives.
- "The capstone integrates everything. You are not building from scratch; you are connecting components you have already built in previous lessons. Emphasis is on deployment, monitoring, and production-readiness."
- Scaffolding: ~40%. "This is the integration exercise. The components exist. You assemble them."
  **Transition**: "The deployment layer is Nexus."

---

## Slide 82: Nexus: One Codebase, Three Interfaces

**Time**: ~3 min
**Talking points**:

- Walk through the Nexus code. "Nexus is the deployment layer. You write your service once, and Nexus exposes it as an API (for web apps), a CLI (for developers), and an MCP server (for AI agents). No code duplication."
- Walk the middleware precisely: "The CORS allow-list is a CONSTRUCTOR argument (cors_origins=[...]). JWT auth and rate limiting arrive as a PLUGIN — NexusAuthPlugin — plugins are how you extend Nexus. The signing secret comes from the environment, never from source code."
- The registration call: "app.handler_extract('serve_qa', serve_qa, description=…) turns ONE handler into a REST endpoint + CLI command + MCP tool. This is the exact shape of Exercise 6.8 (ex_8/03)."
- "If beginners look confused": "Think of it as one restaurant kitchen that serves dine-in, takeaway, and delivery from the same menu. One kitchen, three interfaces."
- "If experts look bored": "The interesting architectural choice is that Nexus treats API, CLI, and MCP as equivalent channels. Most frameworks treat one as primary and the others as afterthoughts."
  **Transition**: "Nothing ships without auth."

---

## Slide 83: Authentication & Authorisation

**Time**: ~3 min
**Talking points**:

- The split, stated crisply: "Nexus AUTHENTICATES (JWT) — it proves WHO is calling and rejects missing or forged tokens with 401 before your handler runs. PACT AUTHORISES (verify_action) — the role claim in the VERIFIED token picks a PACT tier, and verify_action on that tier's D/T/R address decides WHAT they may do."
- Walk the handler code: role from `request.state.user.roles` (the verified token — never from the request body), verify_action on the tier's address, return `{"blocked": true, "reason": …}` when denied.
- "Remember the PACT default: an address with no envelope is auto-approved, so every tier the handler can route to must have an envelope attached — the capstone's build_capstone_stack does this. An unknown role should be refused by YOUR handler before PACT is even asked."
- "Unauthenticated requests must be rejected with 401. No exceptions."
- "If beginners look confused": "RBAC is 'roles have permissions'. JWT is 'a token proves who you are'. You combine them: the token identifies the user, and the PACT tier decides what the user's role can do."
- "If experts look bored": "The interesting unification is mapping human RBAC to agent D/T/R. One governance model, two kinds of actors."
  **Transition**: "Now let us see the whole stack end to end."

---

## Slide 84: Full Platform Integration

**Time**: ~3 min
**Talking points**:

- Walk through the flow: TrainingPipeline to DataFlow to Kaizen Agent to PACT Govern to Nexus Deploy to DriftMonitor.
- Walk through the table. Six packages, one pipeline. Every module you have completed contributes a piece.
- "This is the entire MLFP stack. M6 is where it all comes together. Every exercise from M1 onwards was preparation for this integration."
- Debugging traces: every agent action, every governance decision, every API call is traceable. "When something fails in production, you can reconstruct the entire chain."
- "If beginners look confused": "This is the assembly line. Each station does one thing, and the final product is a deployed AI system."
- "If experts look bored": "The interesting operational question is where failures cascade. A DriftMonitor alert triggers retraining; a PACT denial triggers an audit review; a Nexus 500 triggers a rollback. These are the SRE patterns."
  **Transition**: "Deployed models degrade. You must monitor."

---

## Slide 85: Production Monitoring: DriftMonitor

**Time**: ~2 min
**Talking points**:

- Why monitor: "Models degrade over time as data distributions shift. An accurate model today may be wrong tomorrow."
- Three types of drift: data drift (input distribution changes), concept drift (input-output relationship changes), performance drift (accuracy degrades).
- Walk through the DriftMonitor code with the real API: "It persists reports through a ConnectionManager, is scoped by tenant_id, and both set_reference_data and check_drift are ASYNC. Reference data first, then check each production batch: report.overall_drift_detected, report.feature_results[0].psi."
- PSI reading: "Below 0.1, no drift. 0.1 to 0.2, investigate. Above 0.2, alert — trigger the retraining pipeline."
- "In Exercise 6.8 (ex_8/04) the monitored feature is question length on the capstone QA traffic — real SQuAD traffic versus a window that mixes in MMLU exam questions — and the deliberately shifted batch MUST trigger the alert."
- "DriftMonitor was introduced in earlier modules. Here it is deployed in production. The key: monitoring is not optional. A deployed model without monitoring is a ticking time bomb."
- "If beginners look confused": "Think of it as a regular health check for your AI system. You take its temperature to catch problems early."
- "If experts look bored": "The interesting question is concept drift detection without labels. DriftMonitor uses statistical distribution tests on inputs, but confirming concept drift requires labelled feedback."
  **Transition**: "Agents need different debugging than models."

---

## Slide 86: Debugging Agent Reasoning Chains

**Time**: ~2 min
**Talking points**:

- Walk through the trace code with the real API: "The trace comes from the observatory's capture_run — there is NO trace=True flag on the agent. tool_usage lists one row per tool call in order; detect_loops flags the same call repeated — a stuck agent."
- "Governance decisions live somewhere else: the supervisor's hash-chained audit trail. Read record_type and action from governed.audit.to_list()."
- Common debugging patterns table: loop forever (ambiguous goal, missing tool), wrong tool (descriptions too similar), governance blocked (missing permission), budget exhausted (too many retries), incoherent reasoning (context overflow).
- "Debugging agents is different from debugging code. You read the reasoning trace, not a stack trace. The most common issue: the agent loops because it does not have the right tool. Fix the tools, not the prompts."
- "If beginners look confused": "The trace is the debug log. You read it in English. If the agent is confused, the trace shows you exactly where the confusion started."
- "If experts look bored": "The interesting observation is that most agent failures are tool design failures, not LLM failures. Fix the tools, and the agent usually improves dramatically."
  **Transition**: "Agents also need automated tests."

---

## Slide 87: Testing Agentic Systems

**Time**: ~2 min
**Talking points**:

- What to test: tool correctness (known input, known output), reasoning quality (correct tool selected, read from the captured trace), governance (blocked stays blocked — on an engine whose roles have envelopes), bounds (the agent stops at its turn ceiling), end-to-end (correct final result).
- Walk through the test code: capture a run, assert the expected tool appears in tool_usage, assert detect_loops is empty, assert a denied action returns not allowed.
- "Test STRUCTURE, not wording. LLM text varies run to run; the right tool being called, typed fields being present, and governance verdicts matching do not."
- "Every test must be able to FAIL. Exercise 6.8 (ex_8/04) runs a five-test harness that includes two deny cases — a harness that cannot fail proves nothing."
- "If beginners look confused": "You do not test what the agent says. You test what it does. Did it call the right tool? Did it stay within its ceiling? Did governance block the bad actions?"
- "If experts look bored": "Property-based testing for agents is the frontier. Random prompts, check invariants (bounds respected, no prohibited tool calls). The challenge is defining good properties."
  **Transition**: "Quick note on production inference optimisation."

---

## Slide 88: Inference Optimisation: Production Serving

**Time**: ~2 min
**Talking points**:

- vLLM: PagedAttention for efficient KV-cache memory, continuous batching for GPU utilisation, tensor parallelism for multi-GPU, OpenAI-compatible API.
- Flash Attention: tiling-based attention, reduces memory from O(n^2) to O(n), 2-4x faster than standard attention, built into most modern frameworks.
- "You do not implement these. You configure them. The engineering skill is knowing which optimisation applies to your deployment constraint — latency, throughput, or memory."
- "If beginners look confused": "Skip the details. The point is that serving LLMs in production needs special frameworks. vLLM is the standard."
- "If experts look bored": "PagedAttention is the interesting contribution. Treating KV-cache memory like OS page tables was a significant insight."
  **Transition**: "One more brief awareness slide before the capstone exercise."

---

## Slide 89: Multimodal LLMs (Brief Awareness)

**Time**: ~2 min
**Talking points**:

- Vision-language models: text + image understanding, native multimodal frontier models, and open-source families such as LLaVA.
- Applications: document understanding (OCR + reasoning), chart interpretation, visual question answering, multimodal RAG.
- "Trajectory: LLMs are becoming multimodal by default. The text-only era is ending. Future agents will see, hear, and read simultaneously. The same Kaizen patterns — Delegate, BaseAgent, tools — apply to multimodal models."
- "For this module: awareness only. The engineering patterns you have learned — prompting, agents, governance — transfer directly. The tools change; the architecture does not."
- "If beginners look confused": "Do not worry about this slide. It is a pointer for future learning, not something you need today."
- "If experts look bored": "The interesting question is whether multimodal agents need new governance primitives. Image inputs introduce new prompt injection vectors."
  **Transition**: "Time for the capstone."

---

## Slide 90: Exercise 6.8: Capstone — Deploy a Governed AI Platform

**Time**: ~2 min
**Talking points**:

- Five files: load a trained adapter from the AdapterRegistry and score it on MMLU (ex_8/01); compile the PACT org, attach envelopes, build three governed tiers (02); serve ONE governed handler via Nexus with JWT, CORS and rate limit, and call the API channel including the 401s (03); DriftMonitor on shifted traffic, debug a call, and a five-test harness (04); and a compliance report generated from the live audit chain (05).
- Be precise about channels: "The handler is registered on API + CLI + MCP by the same call. The exercise exercises the API channel in-process with real tokens; driving the CLI and MCP channels is the stretch goal."
- Criteria recap: three channels registered, auth works (missing or forged tokens rejected with 401), drift monitoring alerts on the shifted batch only, governance enforced in production, complete audit trail.
- "If it deploys, authenticates, monitors, and governs correctly, you have passed. That is the entire bar for the capstone."
- "If beginners look confused": "Start with the Nexus deployment. Get it running without auth. Then add auth. Then add DriftMonitor. Incremental."
- "If experts look bored": "Stretch: run a real drift injection test. Deploy, send normal traffic, then send shifted traffic, verify DriftMonitor catches it and triggers retraining."
  **Transition**: "Let us step back and review."

**[PAUSE FOR QUESTIONS — 3 min]**

---

## Slide 91: Key Formula Recap: Attention Mechanism

**Time**: ~2 min
**Talking points**:

- Recap the attention formula from M5. "It is the foundation of everything in M6."
- Where it shows up: 6.1 (LLMs use multi-head attention), 6.2 (LoRA targets attention projection matrices), 6.4 (dense retrieval uses attention-based embeddings), 6.8 (Flash Attention optimises the computation).
- The sqrt(d_k) scaling: "Prevents dot products from growing too large, which would push softmax into saturated regions with near-zero gradients. This is why training is stable at scale."
- "If beginners look confused": "This formula is the engine inside every LLM. Everything we built in M6 runs on this. You do not need to derive it, you need to know it is there."
- "If experts look bored": "The interesting operational consequence of sqrt(d_k) scaling is that attention is still the dominant cost for long sequences. That is why Flash Attention matters."
  **Transition**: "All four formulas in one place."

---

## Slide 92: M6 Formula Summary

**Time**: ~2 min
**Talking points**:

- Walk through the four formulas: LoRA (W = W_0 + BA), DPO loss, GRPO advantage (std-normalised), Attention.
- "Four formulas, four concepts: LoRA decomposes weight updates, DPO aligns with preferences, GRPO normalises rewards within a group, Attention is the computation substrate for all of them."
- "You do not need to memorise these. You need to know what problem each one solves."
- "If beginners look confused": "Each formula solves one problem. LoRA makes fine-tuning cheap. DPO makes alignment simple. GRPO makes verifiable-task training stable. Attention is the core LLM operation."
- "If experts look bored": "Write them down without looking. You should be able to. If not, revisit the corresponding lesson tonight."
  **Transition**: "The complete Kailash stack, mapped."

---

## Slide 93: The Complete Kailash Stack

**Time**: ~2 min
**Talking points**:

- Walk through the flow diagram once: kailash-ml, kailash-dataflow, kaizen, kailash-align, pact, nexus.
- Walk through the API reference table. This is the bookmark slide for the capstone: BaseAgent/Signature/Delegate + ToolRegistry/GovernedSupervisor (kaizen), AlignmentPipeline/AlignmentConfig/AdapterRegistry (align), GovernanceEngine/load_org_yaml/verify_action/ConstraintEnvelopeConfig/RoleEnvelope (pact), Nexus/handler_extract/NexusAuthPlugin/cors_origins (nexus), DataExplorer/DriftMonitor (kailash-ml), MCPServer/@server.tool()/MCPClient (kailash-mcp).
- "Every Kailash API used in M6 is on this slide. Bookmark it for the capstone exercise."
- "If beginners look confused": "Screenshot this slide. When you are building the capstone and you cannot remember which class handles governance, this is your reference."
- "If experts look bored": "Note that kailash-mcp is the thinnest package of the six. It is a protocol implementation, not a business logic layer."
  **Transition**: "The journey, one last time."

---

## Slide 94: The M6 Journey

**Time**: ~2 min
**Talking points**:

- Walk through the timeline once: 6.1 to 6.8, one node each.
- The six-verb arc: Understand (6.1), Customise (6.2-6.3), Ground (6.4), Empower (6.5-6.6), Govern (6.7), Ship (6.8).
- "Every concept produces runnable code. There is no theory-only lesson in M6. If you completed the exercises, you have built a production-ready AI platform."
- "If beginners look confused": "The journey words — understand, customise, ground, empower, govern, ship — are your mental index. Each one points to 1-2 lessons."
- "If experts look bored": "Notice how the verbs compound. You cannot ship without governing. You cannot govern without empowering. Each step requires the previous."
  **Transition**: "What makes this module different from the average LLM course."

---

## Slide 95: What Makes This Module Different

**Time**: ~2 min
**Talking points**:

- Comparison grid: a typical LLM course is "call the API, parse the response, build a chatbot". MLFP M6 is "implement LoRA from scratch, derive DPO, build governed multi-agent systems, deploy with monitoring".
- "The difference: we teach you to build systems that are safe, tested, governed, and deployed. Not just systems that work in a notebook."
- "If the module felt hard today, hard means you learned something real."
- "If beginners look confused": "It is okay if the math was hard. The engineering patterns — agents, governance, deployment — will still serve you even if the derivations fade."
- "If experts look bored": "This slide is for the rest of the room. Let it sit."
  **Transition**: "Your decision framework for real work."

---

## Slide 96: Decision Framework: When to Use What

**Time**: ~2 min
**Talking points**:

- Walk through the table. Problem on the left, solution on the right, lesson number for reference.
- "This is your cheat sheet. When you face a real problem, find it in the left column and the solution is in the right."
- "This slide plus Slide 93 is the entire M6 reference card."
- "If beginners look confused": "You do not need to memorise this. Screenshot it. Refer back when you face a real problem."
- "If experts look bored": "The interesting decisions are the ambiguous ones. 'Model needs domain knowledge' could be RAG or fine-tuning. Use RAG first; fine-tune only when RAG is insufficient."
  **Transition**: "And the mistakes to avoid."

---

## Slide 97: Common Mistakes to Avoid

**Time**: ~2 min
**Talking points**:

- Walk through each mistake. The most common: fine-tuning when RAG would work (cheaper, faster, updatable). The most dangerous: no governance on agents (security liability).
- The newest entry, say it slowly: "Roles without envelopes. PACT auto-approves them — that is the fail-open default from 6.7. Attach an envelope to every role, and test the deny path."
- Other entries: agent without bounds (always set max_turns; on priced providers also budget_limit_usd at construction), testing LLM exact output (tests flake — test tool selection, governance, budget), deploying without monitoring (silent degradation), agents for simple tasks (unnecessary cost).
- "These are the mistakes that every team makes once. Learn from ours."
- "If beginners look confused": "If you only remember one line from this slide: every agent needs a turn ceiling and every role needs an envelope. No exceptions."
- "If experts look bored": "The 'testing LLM exact output' one catches more senior engineers than juniors. Juniors know the output is non-deterministic. Seniors think they can work around it."
  **Transition**: "Where M6 sits in the MLFP programme."

---

## Slide 98: M6 in the MLFP Curriculum

**Time**: ~2 min
**Talking points**:

- Walk through what M6 builds on (M1 Python and Polars, M2 statistics, M3 supervised ML, M4 NLP and embeddings, M5 DL and transformers).
- Walk through what M6 delivers: complete LLM engineering skillset, from-scratch implementations, production deployment with governance, full Kailash stack integrated end-to-end.
- "M6 is the culmination of the MLFP programme. Every module contributed a building block. M6 assembles them into a production AI platform."
- "If beginners look confused": "Every prerequisite you worried about in earlier modules has paid off today. M1 to M5 were not gatekeeping. They were preparation."
- "If experts look bored": "The interesting observation is that M6 consumes every package in the Kailash stack. No other module uses all six."
  **Transition**: "The spectrum view."

---

## Slide 99: The ML Spectrum: Where LLMs Fit

**Time**: ~1 min
**Talking points**:

- Walk through the module spectrum briefly. M1-M2 data and statistics, M3 supervised, M4 pattern discovery and language, M5 representation learning, M6 reasoning, agency, governance.
- Walk through the M6 lesson spectrum. Each lesson occupies a different position.
- "M6 sits at the top of the spectrum: from data understanding through to autonomous reasoning systems."
- "If beginners look confused": "This is the map. M6 is the far right, the most autonomous. Earlier modules are more constrained and more predictable."
- "If experts look bored": "The interesting trend is the governance column. It only appears in 6.7 but it is the thread that makes everything else deployable."
  **Transition**: "Check yourselves."

---

## Slide 100: Self-Assessment: Can You...

**Time**: ~2 min
**Talking points**:

- Ask students to mentally check off each item silently.
- Foundations (everyone): pre-training explanation, 5+ prompting techniques, Kaizen Delegate, RAG pipeline, ReAct agent, bounded agents (turn ceiling, cost budget), PACT access controls, Nexus deployment.
- Theory (stretch): implement LoRA from scratch, derive DPO, GRPO vs DPO, RAGAS metrics, MCP server, governance tests.
- Advanced (expert): model merging, QLoRA and quantisation, vLLM optimisations.
- "If you cannot check off all Foundations items, revisit the relevant exercises. If you can check off Theory items, you are ready for advanced work."
- "If beginners look confused": "If the Theory column feels out of reach, that is fine. Foundations is the bar for passing this module."
- "If experts look bored": "Everyone in the Advanced column should aim for all three items. Model merging is the one most commonly missed."
  **Transition**: "How you will be assessed."

---

## Slide 101: End-of-Module Assessment

**Time**: ~2 min
**Talking points**:

- Describe the assessment EXACTLY as shipped in assessment/README.md: four auto-graded coding tasks, 100 marks, 3 hours, open-book, no AI assistants, running on local Ollama — no API keys, no internet.
- The four tasks: Task 1 (20 marks) prompt engineering & structured output (Signature + BaseAgent); Task 2 (25) RAG pipeline with evaluation (Ollama embeddings + Delegate); Task 3 (25) a tool-using agent over a real dataset (Delegate + ToolRegistry); Task 4 (30) governance for a production agent fleet (PACT GovernanceEngine).
- Coverage note, stated honestly: "Fine-tuning (6.2–6.3), multi-agent/MCP (6.6) and deployment (6.8) are assessed through their EXERCISES, not the timed tasks."
- "Students complete each starter.py; the graders check the return contract in each problem.md. Open-book means you can look things up — what you cannot look up is the thinking."
- Logistics: "Make sure Ollama is running with both models pulled before the session starts."
- "If beginners look confused": "Open-book means the skill is engineering, not memorisation. Can you build it, govern it, and explain why your design decisions are correct? The code IS the answer."
- "If experts look bored": "Task 4 weights governance heavily. An agent fleet without envelopes fails, regardless of how impressive the model is."
  **Transition**: "Resources for deeper study."

---

## Slide 102: Resources & References

**Time**: ~1 min
**Talking points**:

- Papers column: Hu et al. (LoRA), Houlsby et al. (Adapters), Rafailov et al. (DPO), Shao et al. 2024 (DeepSeekMath — introduces GRPO; later used in DeepSeek-R1, 2025), Yao et al. (ReAct), Wei et al. (CoT), Kojima et al. (zero-shot CoT), Lewis et al. (RAG).
- Kailash documentation: kaizen, align, pact, nexus, mcp.
- Tools: lm-eval-harness, RAGAS, vLLM.
- "Reference slide. Papers are for depth. Kailash docs are your primary reference for the exercises."
- "If beginners look confused": "You do not need to read the papers. They are here if you want depth."
- "If experts look bored": "The DPO paper (Rafailov 2023) is the most elegant read on this list. Skim it tonight if you have energy."
  **Transition**: "The eight sentences we want you to remember."

---

## Slide 103: Key Takeaways

**Time**: ~2 min
**Talking points**:

- Read each takeaway aloud:
  1. Prompting is an engineering skill. Start simple, escalate by measured need.
  2. LoRA democratised fine-tuning. Hundreds of times fewer trainable parameters, often near full fine-tuning quality.
  3. DPO eliminated the reward model. Alignment is now accessible.
  4. RAG is the production default. Ground LLMs in facts, not memory.
  5. Agents need tools, not just prompts. ReAct equals reasoning plus action.
  6. Multi-agent systems need coordination protocols. MCP for tools, A2A for agents.
  7. Governance is engineering, not philosophy. Code it. Test it. Ship it.
  8. Deploy with monitoring. A model without DriftMonitor is a liability.
- And the coda on the slide: "Know your defaults: un-enveloped roles are auto-approved. Attach the envelope, then test the deny."
- "If you remember nothing else from today, these eight sentences will serve you well."
- "If beginners look confused": "Write them down. Each one is a complete thought. You will hit every single one of these in real projects."
- "If experts look bored": "The takeaway you are most likely to forget under pressure is number 7. It is the one that quietly distinguishes production-grade systems from demos."
  **Transition**: "Pacing notes for next time — this is the instructor slide."

---

## Slide 104: For the Instructor: Pacing Notes

**Time**: ~1 min (skip or briefly acknowledge for student audiences)
**Talking points**:

- Instructor-only slide. If students are in the room, acknowledge briefly and move on. If this is an instructor-training session, walk through the table.
- Per-lesson estimates: 6.1 (2.5h), 6.2 (3.5h), 6.3 (2.5h), 6.4 (3h), 6.5 (3h), 6.6 (2.5h), 6.7 (2.5h), 6.8 (4h). Total ~24 hours of instructional content (the figure the textbook uses), compressed to 180 minutes for this overview delivery.
- Biggest risks: 6.2 (from-scratch implementations need debugging) and 6.8 (integration issues). Allocate buffer time for both.
- "If running behind: the fine-tuning landscape survey (slide 37) can be assigned as reading."
- "If beginners look confused": "This slide is not for you. Skip it."
- "If experts look bored": "Calibrate your next delivery against these estimates. Adjust as needed."
  **Transition**: "Let us take the final discussion."

---

## Slide 105: Discussion Questions

**Time**: ~8-10 min
**Talking points**:

- Open discussion. Let students discuss in pairs first (3-4 min), then share with the class (5-6 min).
- Reflect questions:
  1. RAG vs fine-tuning — give a concrete example where each is the right choice.
  2. What happens if you deploy an agent without governance? Worst case?
  3. Why does DPO not need a reward model while RLHF does?
- Apply questions: 4. 10,000 internal documents — how would you build a policy Q&A system? 5. AI customer support assistant — which Kailash packages and why?
- "These questions test application, not recall. There are no single correct answers. The quality of reasoning matters."
- "If beginners look confused": "Focus on questions 1 and 4. They are the most grounded in concrete scenarios."
- "If experts look bored": "Question 2 is the most interesting red-team exercise. Walk through the attack surface — and ask what the fail-open default from 6.7 means for their answer."
  **Transition**: "And we close."

---

## Slide 106: Machine Learning with Language Models and Agentic Workflows

**Time**: ~2 min
**Talking points**:

- Read the closing slide title and subtitle aloud: "Module 6 Complete. You can now build, fine-tune, align, ground, govern, and deploy production AI systems."
- Read the provocation: "The goal is not to use AI. The goal is to deploy AI that is safe, tested, governed, and useful."
- Thank the class. This is the final module of MLFP.
- "You have completed the entire MLFP programme. Six modules. From zero Python and polars (M1) to a deployed governed AI platform (M6). Everything connects. Everything you built along the way is a component of what you just shipped today."
- Remind them of the assessment logistics: the four auto-graded tasks from slide 101, the 3-hour window, Ollama running with both models pulled.
- Close with the broader point: "You came in asking how ML works. You leave knowing how to ship it safely. That is the difference between ML curiosity and ML engineering."
- "If beginners look confused": "You do not need to feel expert on day one. You need to feel capable. If you can build the capstone, you are capable."
- "If experts look bored": "The hardest part of this journey for strong practitioners is not the math — it is the discipline of governance and testing. That is what separates published models from deployed systems."
  **Transition**: "Congratulations on completing MLFP. Now go build something that matters."

**[CLOSE — applause, photos, informal Q&A]**

---

## Instructor Notes

### Pacing Summary

The overview delivery targets 180 minutes across 106 slides. The 15-slide Observatory section (slides 6–20) runs about 35 minutes and is the backbone of the day — every later lesson reuses its lenses. Lesson title slides and closing slides are shorter (~1 minute). Key formula slides (LoRA, DPO, GRPO) and exercise introductions run longer (~3-5 minutes). Build in two 1-minute pauses for questions (after Slide 5 and after Slide 90) and one 3-minute break if the room needs it.

Full classroom delivery of every exercise — as outlined on Slide 104 — takes approximately 24 hours across the 8 lessons. This speaker-notes document supports the compressed 180-minute overview that introduces every slide at survey depth.

### Audience Calibration

The room will contain:

- **Novices** who completed M1-M5 and are building their first LLM application. Prioritise green FOUNDATIONS callouts and plain-language fallbacks ("If beginners look confused").
- **LLM practitioners** who have used hosted chat APIs but never implemented LoRA or DPO. Prioritise blue THEORY callouts and derivation deep-dives.
- **ML researchers** who know the papers but have not built governed production systems. Prioritise purple ADVANCED callouts and governance/testing emphasis.

Every slide includes a fallback for both ends. Use them as needed; you do not need to read every bullet.

### Critical Messages (Repeat Often)

1. Every agent needs a turn ceiling; on a priced provider, a dollar cap set at construction. No exceptions.
2. Governance without tests is governance theatre — and a deny test only counts against a role with an envelope attached (the installed PACT auto-approves envelope-less and unknown roles).
3. RAG first, fine-tune second.
4. The captured trace is the debug log.
5. Deploy with monitoring or do not deploy.

These five messages anchor the module. Reinforce them in every discussion.

### Final Note

This is the final module of MLFP. When you close on Slide 106, you are closing the entire programme, not just one module. Honour the arc. Congratulate the class. Leave them with the message that they are now capable of shipping real AI systems, and that the engineering discipline they learned here — not the specific models or APIs — is what will remain valuable as the field evolves.
