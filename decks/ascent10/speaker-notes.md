# Module 10: Alignment, RL & Governance -- Speaker Notes

Total time: ~180 minutes (3 hours)

---

## Slide 1: Title Slide

**Time**: ~1 min
**Talking points**:

- Read the provocation slowly: "The question isn't whether AI will transform your industry. The question is: will YOU lead that transformation?"
- Let it settle. This is the capstone module. Everything converges here.
- "Everything you have built across nine modules converges today. You learn to align models with human values, train agents through reinforcement learning, and govern AI systems in production. This is where engineers become leaders."
- If beginners look confused: "This is the final module. We bring everything together and add the control layer."
- If experts look bored: "We will derive DPO from first principles, cover full PPO clipping maths, and formalise PACT operating envelopes."
  **Transition**: "Let me remind you how far you have come..."

---

## Slide 2: Recap: The Journey So Far

**Time**: ~2 min
**Talking points**:

- Walk the table quickly. Do not re-teach. This is a reference.
- M1-M3: foundations. M4-M5: supervised ML and production. M6: unsupervised. M7-M8: deep learning and NLP. M9: LLMs, agents, RAG.
- "You have gone from loading your first CSV in polars to building multi-agent RAG systems. Today is the capstone -- alignment, RL, and governance bring everything together."
  **Transition**: "Let me show you the full engine map..."

---

## Slide 3: Module 9 Engines -- Cumulative Map

**Time**: ~1 min
**Talking points**:

- Point to all 13 ML engines on the visual.
- "All 13 ML engines are now in play. Today we add the alignment, RL, and governance layers that turn ML models into governed, production-grade AI systems."
- Highlight the new additions: Align (fine-tuning, preference alignment), RLTrainer, PACT GovernanceEngine.
  **Transition**: "Here is the roadmap for today..."

---

## Slide 4: Module 10 Roadmap

**Time**: ~1 min
**Talking points**:

- Walk through seven sections: A (Fine-Tuning), B (Preference Alignment), C (Reinforcement Learning), D (Model Merging & Export), E (AI Governance), F (Capstone), G (Grand Synthesis).
- "Seven sections, from fine-tuning to full-stack governance. By the end you will have covered every major concept in modern ML engineering."
- If beginners look confused: "First we learn to customise models, then to make them safe, then to control them in production."
  **Transition**: "But first -- why does any of this matter?"

---

## Slide 5: Why This Module Matters

**Time**: ~1 min
**Talking points**:

- Two columns: technical gap (unaligned models, RL for interaction, fine-tuning for adaptation, governance for deployment) and career impact (alignment engineers in demand, EU AI Act is law, RL powers next-gen agents).
- "The world has more ML models than it has people who can align, govern, and deploy them responsibly. That is the gap you are about to fill."
  **Transition**: "Let me make this concrete with a real case..."

---

## Slide 6: Case Study: EU AI Act -- First Enforcement Actions

**Time**: ~2 min
**Talking points**:

- Read the headline slowly. February 2025 -- first prohibited AI practices enforceable.
- Fines up to 7% of global annual revenue for prohibited practices, 3% for other violations.
- "This is not hypothetical. The EU AI Act is live. Companies without governance frameworks are spending millions on emergency compliance. Companies that planned ahead? They already have their documentation, risk assessments, and audit trails in place."
- If beginners look confused: "Governments now fine companies for uncontrolled AI. We learn how to avoid that today."
- If experts look bored: "GDPR fines exceeded 1.2B euros. AI Act fines will be larger. 7% of global turnover is existential for most companies."

**PAUSE** -- ask: "Has anyone here worked in an organisation that had to scramble for GDPR compliance? Same energy, higher stakes."

**Transition**: "Here is the timeline..."

---

## Slide 7: The Compliance Timeline

**Time**: ~1 min
**Talking points**:

- Walk the pipeline: Feb 2025 prohibited AI, Aug 2025 GPAI rules, Aug 2026 high-risk obligations, Aug 2027 full force.
- "Note the timeline. Prohibited practices are already enforceable. High-risk system obligations arrive in 2026. If you start governance work after the deadline, you are already in violation."
- Emphasise the callout: governance is architecture, not afterthought. PACT provides that architecture.
  **Transition**: "So what does this mean competitively?"

---

## Slide 8: Your Competitors Are Scrambling. You Will Be Ready.

**Time**: ~1 min
**Talking points**:

- Two columns: without governance (emergency auditing, retrofitting, legal exposure) vs with PACT governance (compiled into workflows, D/T/R at every decision, automatic audit trails, continuous compliance).
- "This is the competitive advantage of governance-by-design. You do not bolt it on after the fact. You build it into every workflow, every agent, every deployment. That is what PACT provides, and that is what you will learn today."
  **Transition**: "Let us begin with Section A -- fine-tuning."

---

## Slide 9: A. Fine-Tuning & Parameter-Efficient Methods

**Time**: ~2 min
**Talking points**:

- This is the decision framework: when to prompt, when to RAG, when to fine-tune.
- "Before you fine-tune, ask: can prompting or RAG solve this? Fine-tuning is the right choice when you need the model to learn a new behaviour or style that prompting alone cannot capture reliably."
- Walk the table: cost and latency trade-offs for each approach.
- If beginners look confused: "Think of prompting as giving instructions, RAG as giving a reference book, and fine-tuning as sending the model to school."
- If experts look bored: "The RAG + fine-tuning combination is what most frontier labs use internally for domain-specific deployments."
  **Transition**: "Let me give you a decision tree..."

---

## Slide 10: Decision Tree: Prompt vs RAG vs Fine-Tune

**Time**: ~2 min
**Talking points**:

- Walk through the five-step decision tree sequentially.
- "Most production use cases stop at step 1 or 2. Fine-tuning is powerful but expensive -- only invest when simpler approaches fall short."
- Step 4 introduces preference alignment (DPO/RLHF) -- foreshadow Section B.

**PAUSE** -- ask: "Think about a use case at your organisation. Where would it land on this tree?"

**Transition**: "When you do need fine-tuning, you have two approaches..."

---

## Slide 11: Full Fine-Tuning vs PEFT

**Time**: ~2 min
**Talking points**:

- Two columns: full fine-tuning (update all params, enormous compute, catastrophic forgetting risk) vs PEFT (freeze base, train small adapter, 0.1-1% params, single GPU).
- "PEFT is one of the most important innovations in modern ML. It makes fine-tuning accessible to everyone, not just organisations with massive GPU clusters."
- Kailash Align defaults to PEFT methods.
- If beginners look confused: "Imagine rewriting an entire textbook vs adding sticky notes. PEFT is the sticky notes -- same book, minor additions."
- If experts look bored: "PEFT methods now match or exceed full fine-tuning quality on most benchmarks, with 100x less compute."
  **Transition**: "The most important PEFT method is LoRA..."

---

## Slide 12: LoRA: Low-Rank Adaptation

**Time**: ~3 min
**Talking points**:

- This is a theory slide. Walk through the equation: W' = W_0 + BA where B and A are small matrices.
- "This is the key insight: we do not need to modify every single weight in a large model. We can express the weight update as a low-rank matrix, which requires far fewer parameters to store and train."
- r is the rank -- typically 4, 8, 16, or 32.
- If beginners look confused: "Instead of changing the whole engine, we add a small turbocharger. The original engine stays untouched."
- If experts look bored: "LoRA exploits the empirically observed low intrinsic dimensionality of fine-tuning weight updates. The rank r is the bottleneck dimension."
  **Transition**: "Why does a low-rank approximation work so well?"

---

## Slide 13: Why Low Rank Works: The SVD Connection

**Time**: ~2 min
**Talking points**:

- Connect to M6 PCA: "Remember PCA from Module 6? You learned that most of the information in high-dimensional data lives in a small subspace. The same principle applies to weight updates during fine-tuning."
- SVD of the weight change shows most singular values are near zero.
- LoRA exploits this by directly parameterising the low-rank structure.
- If beginners look confused: "Just as most variance in data sits in a few principal components, most of the fine-tuning signal sits in a few directions."
- If experts look bored: "Aghajanyan et al. 2021 showed the intrinsic dimensionality of fine-tuning is remarkably small -- a few hundred dimensions for billion-parameter models."
  **Transition**: "Let me show you the numbers..."

---

## Slide 14: LoRA: Parameter Savings

**Time**: ~2 min
**Talking points**:

- Full fine-tuning: d^2 parameters per layer. LoRA: 2 x d x r parameters.
- For d=4096, r=16: 16.8M vs 131K -- 128x fewer parameters.
- "The savings are dramatic. You can fine-tune a 7-billion parameter model with the same GPU memory that would barely handle forward passes during full fine-tuning."
- If beginners look confused: "From 16 million numbers to 131 thousand. That is like packing a shipping container into a suitcase."
- If experts look bored: "A 7B model goes from 7 billion trainable params to about 17 million -- small enough for a single consumer GPU."
  **Transition**: "Here is how training works step by step..."

---

## Slide 15: LoRA Training Process

**Time**: ~2 min
**Talking points**:

- Five steps: freeze W_0, initialise A random and B to zero, forward pass, backward pass through A and B only, update.
- "Key detail: B is initialised to zero so the model starts exactly where the pretrained model left off. No initial disruption. Training then gradually moves the weights in the learned low-rank direction."
- If beginners look confused: "Step 2 is the clever part -- starting at zero means the model begins unchanged and learns gradually."
  **Transition**: "Once trained, how do we deploy?"

---

## Slide 16: LoRA: Merging for Inference

**Time**: ~2 min
**Talking points**:

- During training: keep W_0, A, B separate. Swap adapters without reloading base.
- At deployment: merge W_final = W_0 + BA. Discard A and B.
- "This is the elegance of LoRA. At inference time, there is ZERO overhead. You merge the adapter into the base weights and deploy a standard model. No extra computation, no extra memory."
- If beginners look confused: "Like baking a cake -- during prep you have separate ingredients, but once baked, it is just a cake."
  **Transition**: "What hyperparameters control LoRA?"

---

## Slide 17: LoRA Hyperparameters

**Time**: ~2 min
**Talking points**:

- Walk the table: rank (4-64), alpha (16-64), target_modules (q_proj, v_proj), dropout (0.0-0.1).
- "The alpha/r ratio controls how much the adaptation influences the output. A ratio of 2 is a safe default. If the model is not learning your task, increase rank. If it is overfitting, decrease rank or add dropout."
- Rule of thumb: start r=16, alpha=32.
  **Transition**: "Which layers should get adapters?"

---

## Slide 18: LoRA: Which Modules to Target

**Time**: ~2 min
**Talking points**:

- Two columns: attention modules (q/k/v/o_proj) and MLP modules (gate/up/down_proj).
- "The original LoRA paper found that targeting query and value projections gives the best results per parameter. This is because attention matrices control what information flows between tokens -- the most impactful part of the transformer."
- Adding MLP modules doubles parameter count. Target all modules only when r is very small (4-8).
- If beginners look confused: "Attention is where the model decides what to focus on. That is where small changes have the biggest effect."
- If experts look bored: "Recent work (LoRA-FA) shows freezing A and only training B halves memory with minimal quality loss."
  **Transition**: "Let me show you how this looks in Kailash..."

---

## Slide 19: Kailash Align: LoRA in Practice

**Time**: ~2 min
**Talking points**:

- Walk through the code: LoRAConfig, AlignEngine, train, merge_and_save.
- "Five lines of configuration, three lines of execution. Kailash Align handles the complexity of adapter injection, gradient computation, and merging. You focus on data quality and hyperparameters."
- Point out how theory maps to code: rank, alpha, target_modules are exactly the hyperparameters from the previous slide.
  **Transition**: "What if your GPU cannot even hold the base model in FP16?"

---

## Slide 20: QLoRA: Quantized LoRA

**Time**: ~3 min
**Talking points**:

- Three innovations: NF4 quantization, double quantization, paged optimizers.
- "QLoRA made fine-tuning massive models accessible to researchers with a single GPU. The key insight is NF4: if model weights are normally distributed (which they empirically are), there is an optimal quantization scheme that minimises information loss."
- Result: 65B model on a single 48GB GPU.
- If beginners look confused: "QLoRA compresses the model so it fits on a smaller GPU, then fine-tunes on top of the compressed model."
- If experts look bored: "NF4 is information-theoretically optimal for N(0,1) -- it places quantization bins at the quantiles of the normal distribution."
  **Transition**: "Why is NF4 better than standard INT4?"

---

## Slide 21: NF4: Why It Works

**Time**: ~2 min
**Talking points**:

- Standard INT4: uniform spacing, wastes resolution on outliers.
- NF4: levels spaced by quantiles of N(0,1), more levels near zero where density is highest.
- "Think of it as adaptive resolution. Standard INT4 gives equal precision to all values, but neural network weights are concentrated around zero. NF4 puts more precision where it matters most."
- If beginners look confused: "Like a histogram with more bins near the peak -- you capture more detail where the data actually is."
  **Transition**: "How much memory does this actually save?"

---

## Slide 22: QLoRA Memory Comparison

**Time**: ~1 min
**Talking points**:

- Walk the table: Full FP16 vs LoRA vs QLoRA for 7B, 13B, 65B models.
- 7B: 120GB vs 28GB vs 6GB. 65B: 780GB vs 260GB vs 42GB.
- "These numbers are why QLoRA was a game-changer. Before QLoRA, fine-tuning a 65B model required a cluster. Now it fits on a single A100."
  **Transition**: "How does double quantization contribute?"

---

## Slide 23: Double Quantization: Details

**Time**: ~2 min
**Talking points**:

- Standard quantization: blocks of 64 with FP32 scaling constant = 0.5 bits/param overhead.
- Double quantization: quantize the constants FP32 -> FP8 = 0.127 bits/param overhead.
- "Double quantization is recursive quantization. You quantize the weights, then you quantize the numbers used to quantize the weights. It sounds absurd but it saves significant memory at negligible quality cost."
- For a 65B model, this saves about 3GB of memory.
  **Transition**: "Here is QLoRA in Kailash..."

---

## Slide 24: Kailash Align: QLoRA

**Time**: ~1 min
**Talking points**:

- Walk the code: QuantConfig with bits=4, quant_type="nf4", double_quant=True.
- "One extra config block. Kailash handles NF4 quantization, paged optimizers, and gradient checkpointing automatically."
  **Transition**: "LoRA is the most popular PEFT method, but it is not the only one..."

---

## Slide 25: Beyond LoRA: Advanced PEFT Methods

**Time**: ~2 min
**Talking points**:

- Walk the table: DoRA, LoRA+, QA-LoRA, Prefix Tuning, Prompt Tuning, (IA)3.
- "LoRA is the most popular PEFT method, but it is not the only one. DoRA often outperforms LoRA by decomposing weight updates into magnitude and direction, similar to weight normalisation."
- All methods available in Kailash Align via the method parameter.
- If beginners look confused: "LoRA is your default. Only switch to these if LoRA does not work well enough for your task."
- If experts look bored: "DoRA's magnitude-direction decomposition explains why LoRA sometimes underfits -- it conflates two orthogonal learning signals."
  **Transition**: "Let me go deeper on DoRA..."

---

## Slide 26: DoRA: Weight-Decomposed Low-Rank

**Time**: ~2 min
**Talking points**:

- Equation: W' = m \* (W_0 + BA) / ||W_0 + BA||\_c. Learnable magnitude vector m.
- "Full fine-tuning changes both magnitude and direction of weight vectors. Standard LoRA conflates these. DoRA separates them, allowing independent optimisation."
- "DoRA consistently outperforms LoRA across benchmarks with minimal additional parameters. The magnitude vector m adds only d parameters per layer."
- If beginners look confused: "DoRA is a smarter version of LoRA that separately controls 'how much' and 'which direction'."
  **Transition**: "There are also methods that predate LoRA..."

---

## Slide 27: Prefix Tuning and Prompt Tuning

**Time**: ~2 min
**Talking points**:

- Prefix tuning: learnable vectors at keys/values at every layer, about 0.1% params, good for NLG.
- Prompt tuning: soft tokens at input only, even fewer params, good for classification.
- (IA)3: just three small vectors that rescale activations. Fewer than 10K parameters total.
- "Prefix tuning and prompt tuning predate LoRA. They work by adding learnable context to the model's input or attention layers."
- If beginners look confused: "Think of prefix tuning as giving the model a permanent system prompt that it can learn to optimise."
  **Transition**: "Now let us talk about the data that drives fine-tuning..."

---

## Slide 28: Supervised Fine-Tuning (SFT)

**Time**: ~2 min
**Talking points**:

- Show the instruction-response pair format: instruction, input, output.
- "SFT is the foundation of fine-tuning. You are teaching the model what good outputs look like for your specific task."
- "Data quality >> quantity. 1,000 high-quality instruction-response pairs often outperform 100,000 noisy ones. Invest in curation, not collection."
- If beginners look confused: "You are showing the model examples of perfect answers so it learns to mimic that quality."
- If experts look bored: "The LIMA paper showed 1,000 curated examples matching 50K+ noisy examples."

**PAUSE** -- ask: "What kind of instruction-response pairs would you need for your domain?"

**Transition**: "How do you curate good SFT data?"

---

## Slide 29: SFT Data Best Practices

**Time**: ~2 min
**Talking points**:

- Five rules: diversity, consistency, correctness, deduplication, length balance.
- Highlight the warning: using LLM-generated data without human verification amplifies model biases.
- "The LIMA paper showed that just 1,000 carefully curated examples can match the performance of models fine-tuned on 50,000+ noisy examples. Quality beats quantity every time."
  **Transition**: "Here is SFT in Kailash..."

---

## Slide 30: Kailash Align: SFT Pipeline

**Time**: ~1 min
**Talking points**:

- Walk the code: AlignEngine with method="sft", SFTConfig with packing=True.
- "Packing is important -- it groups shorter examples to fill the full sequence length, reducing wasted computation. Kailash handles the padding, masking, and loss computation automatically."
  **Transition**: "Let me summarise the fine-tuning landscape..."

---

## Slide 31: Fine-Tuning Summary

**Time**: ~1 min
**Talking points**:

- Quick reference table: Full FT, LoRA, QLoRA, DoRA, Prefix Tuning, (IA)3.
- "This table is your decision guide. Start with LoRA or QLoRA. Only move to more exotic methods if you have a specific constraint that LoRA cannot meet."
  **Transition**: "What goes wrong in fine-tuning?"

---

## Slide 32: Common Fine-Tuning Failures

**Time**: ~2 min
**Talking points**:

- Walk the five failure modes: repetition, worse than base, ignores instructions, catastrophic forgetting, memory errors.
- "These are the five most common fine-tuning failures. The first thing to check is always the data format -- does it match the chat template the base model was trained with?"
- If beginners look confused: "This is your troubleshooting checklist. Bookmark this slide."
- If experts look bored: "Learning rate warmup with cosine decay is almost always the fix for training instability."
  **Transition**: "Speaking of format..."

---

## Slide 33: Data Formatting: Chat Templates

**Time**: ~2 min
**Talking points**:

- Show the Llama-3 chat template with special tokens.
- "Every model family has its own chat template. If your training data uses a different template, the model receives contradictory formatting signals and performance degrades."
- Template mismatch is the #1 cause of fine-tuning underperformance. Kailash Align auto-detects the correct template.
  **Transition**: "How do you know if fine-tuning worked?"

---

## Slide 34: Fine-Tuning Evaluation

**Time**: ~2 min
**Talking points**:

- Two columns: automated benchmarks (MT-Bench, AlpacaEval, MMLU, HumanEval) and manual evaluation (domain expert review, A/B testing, failure case analysis).
- "Always evaluate on both automated benchmarks AND manual review. Automated benchmarks catch regressions. Manual review catches subtle quality issues that metrics miss."
- If beginners look confused: "Run the tests AND have a human check the outputs. Both are necessary."
  **Transition**: "Let me put the whole workflow together..."

---

## Slide 35: Fine-Tuning: End-to-End Workflow

**Time**: ~2 min
**Talking points**:

- Seven steps: curate data, format, configure, train, evaluate, merge, quantize.
- "This is the complete workflow from data to deployment. Seven steps. Most of the effort goes into step 1 -- data curation is where you win or lose."
- If beginners look confused: "Think of it as a recipe: preparation (steps 1-3), cooking (step 4), tasting (step 5), plating (steps 6-7)."
  **Transition**: "Fine-tuning teaches the model WHAT to say. Now let us learn how to teach it WHAT NOT to say. Section B: Preference Alignment."

---

## Slide 36: B. Preference Alignment

**Time**: ~2 min
**Talking points**:

- "SFT teaches a model what to say. Alignment teaches it how to say it -- and what not to say."
- Two columns: without alignment (helpful but toxic, correct but harmful, capable but uncontrollable) vs with alignment (helpful AND harmless, correct AND appropriate, capable AND steerable).
- "A model that can write excellent code but also writes malware on request is not a good model. Alignment is the process of teaching models to be helpful, harmless, and honest."
- If beginners look confused: "Fine-tuning taught the model your domain. Alignment teaches it your values."
- If experts look bored: "The HHH framework from Anthropic formalises this as three competing objectives that alignment must balance."

**PAUSE** -- ask: "Can anyone think of a real-world example where a capable but unaligned model caused harm?"

**Transition**: "The original approach is RLHF..."

---

## Slide 37: The RLHF Pipeline

**Time**: ~3 min
**Talking points**:

- Walk the four-stage pipeline: pre-train, SFT, reward model, PPO.
- "This is the original RLHF pipeline from InstructGPT. Four stages, each building on the last. The magic is in steps 3 and 4: learning what humans prefer and optimising for it."
- If beginners look confused: "Stage 1: learn language. Stage 2: learn instructions. Stage 3: learn what humans like. Stage 4: get better at producing what humans like."
- If experts look bored: "The InstructGPT paper showed that a 1.3B RLHF model was preferred over a 175B base model. Alignment is a multiplier, not a tax."
  **Transition**: "The reward model is the heart of RLHF..."

---

## Slide 38: Reward Model: Bradley-Terry

**Time**: ~3 min
**Talking points**:

- The equation: P(y_1 beats y_2 | x) = sigma(r(x,y_1) - r(x,y_2)).
- "The Bradley-Terry model is from sports ranking -- the same maths used to rank chess players by Elo. Applied to LLMs, it tells us the probability that a human would prefer one response over another, based on a learned reward function."
- The probability depends only on the difference in rewards -- invariant to constant shifts.
- If beginners look confused: "It is like Elo ratings for chess. Higher-rated responses beat lower-rated ones more often."
- If experts look bored: "Bradley-Terry is a special case of Thurstone's Law of Comparative Judgment with the logistic link."
  **Transition**: "How do we train this reward model?"

---

## Slide 39: Reward Model Training

**Time**: ~2 min
**Talking points**:

- Loss function: negative log-likelihood of preferred response having higher reward.
- Collect thousands of (x, y_w, y_l) preference pairs.
- Train reward model (often same architecture as LLM, minus generation head).
- "The reward model is essentially a classifier trained on human preferences. For each prompt, it learns to assign higher reward scores to responses that humans prefer."
  **Transition**: "Now we use this reward model in RL..."

---

## Slide 40: RLHF Objective: Reward + KL Constraint

**Time**: ~3 min
**Talking points**:

- The objective: maximise reward minus beta \* KL(policy || reference).
- "This objective has a beautiful tension: maximise reward (be helpful) while not straying too far from the reference (stay coherent). The beta parameter controls this trade-off."
- "Why the KL term? Without it, the model would hack the reward by finding degenerate outputs that score high but are meaningless."
- If beginners look confused: "We want the model to improve, but not go crazy. The KL penalty is the leash."
- If experts look bored: "This is a constrained optimisation problem. The KL penalty is the Lagrange multiplier on the trust region, and beta controls the radius."

**PAUSE** -- ask: "What might happen if we set beta to zero? Think about what the model would learn to do."

**Transition**: "RLHF works, but it has significant problems..."

---

## Slide 41: Problems with RLHF

**Time**: ~2 min
**Talking points**:

- Complexity: three models in memory (policy, ref, reward). PPO is unstable. Reward hacking.
- Cost: human preferences are expensive. Training loop is slow. Multiple GPUs often required.
- "RLHF works, but it is complicated and expensive. Researchers asked: what if we could get the same result with a simpler loss function? That question led to Direct Preference Optimisation."
  **Transition**: "Enter DPO..."

---

## Slide 42: DPO: Direct Preference Optimisation

**Time**: ~2 min
**Talking points**:

- The big idea: skip the reward model and RL entirely. Derive a closed-form loss from the RLHF objective.
- Pipeline reduces from four stages to two: pre-train, SFT, DPO.
- "DPO is one of the most elegant results in modern ML. It reduces the four-stage RLHF pipeline to two stages by proving that the optimal policy has a closed-form relationship to the reward function."
- If beginners look confused: "Instead of training a separate model to judge quality, DPO bakes the judgment directly into the training loss."
- If experts look bored: "We are about to do the full derivation. Five steps from RLHF objective to DPO loss."
  **Transition**: "Let me walk you through the derivation, step by step..."

---

## Slide 43: DPO Derivation -- Step 1: The RLHF Objective

**Time**: ~2 min
**Talking points**:

- Start with the RLHF objective. Expand KL into log ratio.
- "We start with exactly the objective we saw earlier. The KL divergence expands into a log ratio between the policy and reference. This is the starting point for the derivation."
- If beginners look confused: "We are just rewriting the same equation in a more useful form."
  **Transition**: "Step 2: find the optimal policy..."

---

## Slide 44: DPO Derivation -- Step 2: Optimal Policy

**Time**: ~3 min
**Talking points**:

- Optimal policy: pi*(y|x) = (1/Z(x)) * pi_ref(y|x) \* exp(r(x,y)/beta).
- "The optimal policy is the reference model reweighted by the exponentiated reward. High-reward responses get more probability mass. Beta controls how aggressively."
- "This is a standard result from KL-regularised RL. The optimal policy has this Boltzmann-like form -- exactly what you see in physics with energy minimisation."
- If beginners look confused: "The best policy keeps the reference model but boosts good responses and suppresses bad ones."
- If experts look bored: "This is the Gibbs/Boltzmann distribution for the free energy functional. Standard variational inference."
  **Transition**: "Step 3: solve for the reward..."

---

## Slide 45: DPO Derivation -- Step 3: Solve for Reward

**Time**: ~2 min
**Talking points**:

- Rearrange to express reward as: r(x,y) = beta _ log(pi_/pi_ref) + beta \* log Z(x).
- "This is the critical rearrangement. We have expressed the reward function in terms of the ratio between the optimal policy and the reference model. The partition function Z(x) depends only on the prompt, not on the specific response."
  **Transition**: "Step 4: substitute into Bradley-Terry..."

---

## Slide 46: DPO Derivation -- Step 4: Into Bradley-Terry

**Time**: ~2 min
**Talking points**:

- Substitute reward into preference model: r(x,y_w) - r(x,y_l) cancels the Z(x) terms.
- "This is the magic moment. When we take the difference of rewards for the winning and losing response, the intractable partition function Z(x) cancels completely. We never need to compute it."
- If beginners look confused: "The hard part disappears because we only care about which response is BETTER, not how good each one is absolutely."
- If experts look bored: "The cancellation of Z(x) is what makes DPO practical. Without it, we would need to marginalise over all possible responses."
  **Transition**: "Step 5: the final loss..."

---

## Slide 47: DPO Derivation -- Step 5: The DPO Loss

**Time**: ~3 min
**Talking points**:

- The final loss: sigma of beta times (log ratio for winner minus log ratio for loser).
- "This is the DPO loss. Increase the log probability of the preferred response relative to the reference, and decrease the log probability of the rejected response relative to the reference."
- No reward model. No RL loop. Just supervised training with two forward passes per example.
- If beginners look confused: "The final result is simple: make the model more likely to produce good responses and less likely to produce bad ones."
- If experts look bored: "The loss is the negative log-likelihood of the Bradley-Terry model where the implicit reward is the log-probability ratio."

**PAUSE** -- this is the key theoretical result. Ask: "Questions on the derivation before we move on?"

**Transition**: "What role does beta play?"

---

## Slide 48: DPO: Understanding beta

**Time**: ~2 min
**Talking points**:

- Walk the table: small beta (conservative, high-stakes), medium beta (balanced, general-purpose), large beta (aggressive, creative tasks).
- "Beta is the single most important hyperparameter in DPO. Small beta means the model barely moves from the reference. Large beta means it aggressively chases preferred outputs, risking coherence."
- Default: beta=0.1 is a safe starting point.
  **Transition**: "Even without an explicit reward model, DPO has an implicit one..."

---

## Slide 49: DPO: Implicit Reward

**Time**: ~2 min
**Talking points**:

- Implicit reward: r_hat(x,y) = beta \* log(pi_theta / pi_ref). The log-probability ratio IS the reward.
- "This is a powerful insight for debugging. You can compute the implicit reward for any response and check whether it correlates with your notion of quality."
- If it diverges wildly, the model is overfitting to preference artifacts.
- If beginners look confused: "You can check how the model ranks different responses, even though there is no separate ranking model."
  **Transition**: "Here is DPO in Kailash..."

---

## Slide 50: Kailash Align: DPO

**Time**: ~1 min
**Talking points**:

- Walk the code: AlignEngine with method="dpo", DPOConfig, compute_implicit_rewards.
- "DPO training is simpler than RLHF. Kailash Align handles the reference model freezing and dual forward passes automatically."
  **Transition**: "Now let us look at a method that does not need preference pairs at all..."

---

## Slide 51: GRPO: Group Relative Policy Optimisation

**Time**: ~3 min
**Talking points**:

- The DeepSeek-R1 method. No reward model at all. Generate G responses per prompt, rank by verifiable reward, use relative ranking as signal.
- Advantage estimation: A_i = (r_i - mean(r)) / std(r). Group-normalised, no value network.
- "GRPO was used to train DeepSeek-R1, one of the strongest reasoning models. Instead of training a reward model, generate multiple responses and let their relative quality be the signal."
- If beginners look confused: "Instead of judging each response individually, compare them to each other. The best in the group gets reinforced."
- If experts look bored: "GRPO removes the critic network entirely, using group statistics for variance reduction instead."
  **Transition**: "What does the GRPO loss look like?"

---

## Slide 52: GRPO Loss Function

**Time**: ~2 min
**Talking points**:

- PPO-style clipping with group-relative advantages and per-example KL penalty.
- "GRPO is essentially PPO without a critic. Instead of learning a value function to estimate advantages, it uses the statistics of the group to normalise rewards."
- Connection to Section C (PPO) -- foreshadow that we will cover PPO in detail later.
  **Transition**: "When does GRPO work best?"

---

## Slide 53: GRPO: Verifiable Rewards

**Time**: ~2 min
**Talking points**:

- Two columns: verifiable tasks (maths, code, logic, retrieval) vs non-verifiable tasks (creative writing, summarisation, conversation).
- "The key insight is that GRPO needs verifiable rewards. For math, you can check the answer. For code, you can run tests. For tasks where quality is subjective, DPO or RLHF are better choices."
- If beginners look confused: "GRPO works when there is a right answer you can check automatically."
- If experts look bored: "GRPO with process reward models gives even better results -- reward each reasoning step, not just the final answer."
  **Transition**: "There are even simpler alignment methods..."

---

## Slide 54: ORPO: Odds Ratio Preference Optimisation

**Time**: ~2 min
**Talking points**:

- ORPO combines SFT and preference alignment in a single loss.
- "ORPO merges SFT and preference alignment into one loss. The first term teaches the model to generate good responses. The second term teaches it to prefer good responses over bad ones."
- Why ORPO? DPO requires a separately-trained SFT model as reference. ORPO does both at once.
  **Transition**: "SimPO goes even further..."

---

## Slide 55: SimPO: Simple Preference Optimisation

**Time**: ~2 min
**Talking points**:

- Average log probability as reward. Length-normalised. No reference model needed.
- "SimPO removes the reference model entirely. The average log probability of the model's own output serves as the implicit reward. Length normalisation prevents the model from learning that longer responses are always better."
- If beginners look confused: "SimPO is the simplest alignment method -- no reference model, no reward model, just the model learning from its own confidence."
  **Transition**: "What if you do not have paired preferences?"

---

## Slide 56: KTO: Kahneman-Tversky Optimisation

**Time**: ~2 min
**Talking points**:

- Works with unpaired good/bad labels -- no need for (y_w, y_l) pairs for the same prompt.
- Uses Kahneman-Tversky insight: losses loom larger than gains. Asymmetric loss function.
- "KTO is practical when you cannot pair responses to the same prompt. If you only have thumbs-up/thumbs-down labels from users, KTO is your method."
- If beginners look confused: "KTO needs simpler data -- just tag each response as 'good' or 'bad'. No need to compare two responses side by side."
- If experts look bored: "KTO's asymmetric loss connects to prospect theory -- the loss aversion coefficient lambda controls the asymmetry."
  **Transition**: "What if you have no human labels at all?"

---

## Slide 57: Constitutional AI & RLAIF

**Time**: ~2 min
**Talking points**:

- Three steps: define a constitution, model critiques and revises, use revised responses as preference pairs.
- RLAIF: use a stronger model to generate preference labels instead of humans.
- "Constitutional AI shows that you can bootstrap alignment without massive human annotation budgets. Define your values as principles, let the model self-critique, and use the improved outputs as training data."
  **Transition**: "There are even more frontier methods..."

---

## Slide 58: Online DPO & Process Reward Models

**Time**: ~2 min
**Talking points**:

- Online DPO: generate new responses during training, self-play, avoids overfitting to static data.
- Process Reward Models (PRM): score each step in reasoning, not just the final answer. Better for maths, code, multi-step tasks.
- "These are frontier methods. Online DPO prevents the model from memorising the preference dataset. Process reward models address credit assignment -- rewarding correct reasoning steps, not just correct final answers."
- If experts look bored: "PRM training data is expensive but reliability jumps significantly -- OpenAI reported 10+ percentage point gains on GSM8K with PRM."
  **Transition**: "Let me put all these methods side by side..."

---

## Slide 59: Preference Alignment: Method Comparison

**Time**: ~2 min
**Talking points**:

- Walk the comparison table: RLHF, DPO, GRPO, ORPO, SimPO, KTO, CAI/RLAIF.
- "This comparison table is your decision guide. DPO is the current default for most use cases."
- Kailash Align supports all methods via the method parameter.
- Start with DPO, move to GRPO for verifiable tasks, KTO for unpaired data.
  **Transition**: "Here is the full alignment pipeline in Kailash..."

---

## Slide 60: Kailash Align: Full Alignment Pipeline

**Time**: ~1 min
**Talking points**:

- Walk the code: SFT -> save -> DPO -> save -> evaluate.
- "Two stages, each a few lines. Kailash handles model loading, adapter management, and benchmark evaluation."
  **Transition**: "When should you choose DPO vs RLHF?"

---

## Slide 61: DPO vs RLHF: When to Use Which

**Time**: ~2 min
**Talking points**:

- DPO: paired preference data, limited compute, stability matters, simple pipeline.
- RLHF: reward model already trained, online learning needed, maximum performance, complex/dynamic reward.
- "In 2024 the field has largely shifted to DPO as the default. RLHF is still used when you need online learning or when the reward signal is complex and cannot be expressed through static preference pairs."
  **Transition**: "How do you collect preference data?"

---

## Slide 62: Preference Data Collection

**Time**: ~2 min
**Talking points**:

- Five steps: generate, annotate, validate (inter-annotator agreement > 70%), balance, format.
- "Preference data collection is the most expensive part of alignment. The key metric is inter-annotator agreement -- if annotators disagree on which response is better, the signal is noisy and training will be unstable."
- Scale: 5K-20K preference pairs is typical.

**PAUSE** -- ask: "How would you collect preference data for your domain? Who are the annotators?"

**Transition**: "Alignment has a cost..."

---

## Slide 63: Alignment Tax: The Performance Trade-off

**Time**: ~2 min
**Talking points**:

- What you gain: safety, instruction-following, appropriate refusals, consistent tone.
- What you may lose: raw capability on some benchmarks, willingness on edge cases, creative outputs.
- "Alignment always involves a trade-off. The model becomes safer but may lose some raw capability. The key is minimising this tax through high-quality preference data and appropriate beta values."
- "The alignment tax is real but shrinking. Modern methods achieve strong alignment with minimal capability loss."
- If beginners look confused: "Teaching the model to say no sometimes means it occasionally says no when you wish it had not."
- If experts look bored: "The tax is smallest with domain-specific preference data. Generic alignment data causes the most capability regression."
  **Transition**: "How do you measure alignment?"

---

## Slide 64: Alignment Evaluation

**Time**: ~2 min
**Talking points**:

- Walk the table: helpfulness (MT-Bench, AlpacaEval), harmlessness (TruthfulQA, BBQ), honesty (TruthfulQA), knowledge (MMLU, ARC), reasoning (GSM8K, HumanEval).
- "A model that scores well on helpfulness but poorly on honesty is not well-aligned. Evaluate across all three H's -- helpful, harmless, honest -- plus verify that raw capabilities have not degraded."
- No single benchmark captures alignment. Always evaluate on multiple dimensions.
  **Transition**: "Now we go deeper into the RL that underpins RLHF. Section C: Reinforcement Learning."

---

## Slide 65: C. Reinforcement Learning

**Time**: ~2 min
**Talking points**:

- A different paradigm. Two columns: supervised learning (labelled examples, fixed dataset, immediate feedback, single correct answer) vs RL (learn from interaction, data generated by agent, delayed rewards, explore vs exploit).
- "RL is fundamentally different from everything you have learned so far. There is no training set. The agent generates its own data by interacting with the environment."
- If beginners look confused: "In supervised learning, you have an answer key. In RL, you have to figure out the answers by trying things and seeing what works."
- If experts look bored: "RL is the theoretical framework that makes RLHF possible. We are now going to build that framework from scratch."
  **Transition**: "Let us formalise this with the MDP..."

---

## Slide 66: Markov Decision Process (MDP)

**Time**: ~3 min
**Talking points**:

- The tuple: (S, A, P, R, gamma). Walk each component.
- "The MDP is to reinforcement learning what the dataset is to supervised learning -- it defines the problem. Every RL algorithm operates on an MDP."
- If beginners look confused: "An MDP is a game: there are situations (states), choices (actions), rules (transitions), scores (rewards), and patience (discount)."
- If experts look bored: "We are assuming full observability here. POMDPs relax this but are intractable in general."

**PAUSE** -- ask: "Can you think of something you do daily that could be modelled as an MDP?"

**Transition**: "The key property that makes MDPs tractable..."

---

## Slide 67: The Markov Property

**Time**: ~2 min
**Talking points**:

- P(S_t+1 | S_t, A_t) = P(S_t+1 | S_t, A_t, history). The future depends only on the current state.
- "The Markov property is what makes the MDP work. The current state contains all relevant information for deciding the next action."
- Chess analogy: if you know the current board position, you do not need the move history.
- If beginners look confused: "It is like a GPS -- it only needs to know where you ARE, not how you got there."
- If experts look bored: "When the Markov property does not hold, you either engineer the state to include sufficient history or use recurrent architectures."
  **Transition**: "Given an MDP, how do we evaluate behaviour?"

---

## Slide 68: Policy, Value, and Q Functions

**Time**: ~3 min
**Talking points**:

- Policy pi(a|s): probability of taking action a in state s.
- V^pi(s): expected return starting from state s, following policy pi. "How good is it to BE in state s?"
- Q^pi(s,a): expected return starting from state s, taking action a. "How good is it to DO action a in state s?"
- If beginners look confused: "The policy is your strategy. V tells you how good your current position is. Q tells you how good a specific move is."
- If experts look bored: "Q is the fundamental quantity. V = E_a[Q(s,a)] and the optimal policy is pi*(s) = argmax_a Q*(s,a)."
  **Transition**: "How much should the agent care about the future?"

---

## Slide 69: The Discount Factor gamma

**Time**: ~2 min
**Talking points**:

- Walk the table: gamma=0 (greedy), 0.9 (moderate), 0.99 (long horizon), 1.0 (episodic only).
- "Gamma is the agent's time preference. A gamma of 0.99 means a reward 100 steps from now is worth about 37% of its face value today."
- Mathematical role: gamma < 1 ensures convergence of infinite sums.
- If beginners look confused: "It is like a financial discount rate. Money today is worth more than money next year."
  **Transition**: "Now the most important equation in RL..."

---

## Slide 70: Bellman Expectation Equation

**Time**: ~3 min
**Talking points**:

- Step-by-step derivation: start with V definition, split first step, recognise recursion.
- Final equation: V^pi(s) = sum_a pi(a|s) sum_s' P(s'|s,a) [R(s,a) + gamma V^pi(s')].
- "The Bellman equation is the foundation of all RL algorithms. It says that the value of any state equals the immediate reward plus the discounted value of the next state."
- If beginners look confused: "The value of where you are = what you get now + what you expect to get later."
- If experts look bored: "This is a system of |S| linear equations in |S| unknowns. Solvable in O(|S|^3) -- but only for small state spaces."
  **Transition**: "What about the optimal case?"

---

## Slide 71: Bellman Optimality Equation

**Time**: ~3 min
**Talking points**:

- V*(s) = max_a sum P [R + gamma V*(s')]. Q\* similarly.
- "The Bellman optimality equation replaces the expectation over actions with a maximum. If we can solve this equation, we have the optimal policy."
- Key difference: expectation equation uses sum_a pi(a|s), optimality uses max_a.
- If beginners look confused: "Instead of averaging across actions, we always pick the best one."
- If experts look bored: "The optimality equation is nonlinear due to the max, which is why we need iterative methods."
  **Transition**: "How do we solve these equations?"

---

## Slide 72: Policy Evaluation & Improvement

**Time**: ~3 min
**Talking points**:

- Policy evaluation: given pi, compute V^pi iteratively.
- Policy improvement: given V^pi, improve pi by being greedy.
- Policy iteration: alternate until pi stops changing. Guaranteed convergence.
- "Policy iteration has two steps: evaluate how good the current policy is, then improve the policy by being greedy. This alternation provably converges to the optimal policy."
- If beginners look confused: "Test your strategy, then improve it. Repeat until it cannot get any better."
  **Transition**: "Can we combine both steps?"

---

## Slide 73: Value Iteration

**Time**: ~2 min
**Talking points**:

- V_k+1(s) = max_a sum P [R + gamma V_k(s')]. Direct application of Bellman optimality as an update rule.
- "Value iteration is simpler than policy iteration. But both require the full transition model."
- Limitation: what if we do not know P(s'|s,a)?
- If beginners look confused: "So far we assumed we know all the rules of the game. What if we do not?"
  **Transition**: "This motivates model-free methods..."

---

## Slide 74: Monte Carlo Methods

**Time**: ~3 min
**Talking points**:

- Three steps: run episode to completion, compute return G_t, update V(s).
- First-visit vs every-visit MC.
- "Monte Carlo methods learn from experience -- no model needed. Run an episode, see what happened, update your estimates. The downside: you must wait until the episode ends."
- If beginners look confused: "Play the whole game, then look back and learn from what happened."
- If experts look bored: "MC provides unbiased estimates of V^pi but with high variance due to full episode returns."
  **Transition**: "What if we do not want to wait until the end?"

---

## Slide 75: TD(0): Temporal Difference Learning

**Time**: ~3 min
**Talking points**:

- Core update: V(s) <- V(s) + alpha [r + gamma V(s') - V(s)]. The TD error delta.
- "TD learning is one of the most important ideas in RL. Instead of waiting for the episode to end, TD updates immediately using the current estimate. This is called bootstrapping."
- MC waits for full return. TD uses current estimate V(s'). Learn at every step.
- TD error delta: the surprise -- how much better or worse the experience was than expected.
- If beginners look confused: "Instead of waiting for the final score, update your estimate after every single move."
- If experts look bored: "TD combines sampling (like MC) with bootstrapping (like DP), getting the best of both worlds."
  **Transition**: "How do MC and TD compare?"

---

## Slide 76: MC vs TD: The Bias-Variance Trade-off

**Time**: ~2 min
**Talking points**:

- Walk the comparison table: bias, variance, needs episodes to end, online learning, function approximation.
- "Monte Carlo gives truth with high variance. TD gives biased estimates with low variance and instant feedback. TD usually wins."
- In practice, TD methods dominate because most real-world tasks are continuous.
  **Transition**: "Let us apply TD to control..."

---

## Slide 77: SARSA: On-Policy TD Control

**Time**: ~3 min
**Talking points**:

- Q(s,a) <- Q(s,a) + alpha[r + gamma Q(s',a') - Q(s,a)]. Name comes from S,A,R,S',A'.
- "SARSA learns Q-values by following its own policy. Because it uses the actual next action a-prime, it learns a policy that is safe under exploration."
- On-policy: evaluates and improves the SAME policy used to generate experience. SARSA is "cautious."
- If beginners look confused: "SARSA learns from what it actually does, not what it could do."
  **Transition**: "What if we want to learn the optimal policy regardless of what we do?"

---

## Slide 78: Q-Learning: Off-Policy TD Control

**Time**: ~3 min
**Talking points**:

- Q(s,a) <- Q(s,a) + alpha[r + gamma max_a' Q(s',a') - Q(s,a)]. Uses max instead of actual action.
- "Q-learning is the most famous RL algorithm. The key difference from SARSA is the max: Q-learning always imagines taking the best possible next action."
- SARSA: learns value of what it does. Q-learning: learns value of the best it could do.
- If beginners look confused: "SARSA learns about your actual play. Q-learning learns about perfect play."
- If experts look bored: "Q-learning converges to Q\* under standard conditions (all state-action pairs visited infinitely often, decaying learning rate). SARSA converges to Q^pi for the exploration policy."

**PAUSE** -- ask: "Which would you prefer for a self-driving car? SARSA or Q-learning? Think about safety during learning."

**Transition**: "How do we balance exploring and exploiting?"

---

## Slide 79: epsilon-Greedy Exploration

**Time**: ~2 min
**Talking points**:

- With probability 1-epsilon take best known action, with probability epsilon take random action.
- Common schedule: start epsilon=1.0, decay to 0.01 over training.
- "Epsilon-greedy is the simplest solution to exploration: with probability epsilon, take a random action. With probability 1-epsilon, take the best known action."
- If beginners look confused: "Sometimes you try something new (explore), mostly you do what works (exploit)."
  **Transition**: "Tabular Q-learning does not scale. Enter deep RL..."

---

## Slide 80: Deep Q-Networks (DQN)

**Time**: ~3 min
**Talking points**:

- Neural network Q_theta(s,a) for large state spaces.
- Two key innovations: experience replay (store and sample random mini-batches) and target network (separate, slowly-updated Q).
- "DQN combined deep learning with RL. Experience replay breaks correlations in sequential data, and the target network prevents the moving-target problem."
- If beginners look confused: "DQN replaces the Q-table with a neural network, so it can handle images and continuous states."
- If experts look bored: "DQN was the first algorithm to achieve human-level performance across 49 Atari games from pixels. Published in Nature, 2015."
  **Transition**: "Why is experience replay so important?"

---

## Slide 81: Why Experience Replay?

**Time**: ~2 min
**Talking points**:

- Without replay: correlated data, recent experience dominates, catastrophic forgetting.
- With replay: random sampling breaks correlations, past experiences revisited, data-efficient.
- "Experience replay turns RL into something closer to supervised learning by breaking the temporal correlation in the training data."
  **Transition**: "DQN has been improved in many ways..."

---

## Slide 82: DQN Variants

**Time**: ~2 min
**Talking points**:

- Double DQN: fixes overestimation. Dueling DQN: separates V and advantage. PER: prioritises high-error experiences.
- Rainbow DQN combines all improvements. State-of-the-art for discrete action spaces.
- If beginners look confused: "Each variant fixes one problem with the original. Rainbow combines all the fixes."
- If experts look bored: "Rainbow DQN adds distributional RL (C51) and noisy networks on top of these three."
  **Transition**: "All DQN methods work with discrete actions. For continuous actions, we need a different approach..."

---

## Slide 83: Policy Gradient: REINFORCE

**Time**: ~3 min
**Talking points**:

- The policy gradient theorem: gradient of expected return w.r.t. policy parameters.
- "REINFORCE is the foundational policy gradient algorithm. If action a led to high return G_t, increase its log-probability. If it led to low return, decrease it."
- "You can compute the gradient of expected reward even though the expectation is over trajectories."
- If beginners look confused: "Reinforce actions that worked well. Weaken actions that did not."
- If experts look bored: "The log-derivative trick (REINFORCE trick) is what makes this gradient computable without differentiating through the environment."
  **Transition**: "REINFORCE has very high variance..."

---

## Slide 84: REINFORCE: High Variance and Baselines

**Time**: ~2 min
**Talking points**:

- Subtract baseline b(s) to reduce variance without adding bias. When b(s) = V(s), G_t - V(s) = A(s,a) -- the advantage function.
- "Without a baseline, every action in a good episode gets reinforced. The advantage function fixes this by asking: was this action better or worse than average?"
- If beginners look confused: "Instead of 'that was a good game so every move was good', we ask 'was that move better than your average move?'"
  **Transition**: "This motivates the actor-critic architecture..."

---

## Slide 85: Actor-Critic Architecture

**Time**: ~3 min
**Talking points**:

- Actor pi_theta(a|s) outputs the policy. Critic V_phi(s) estimates state value.
- Advantage = r + gamma V(s') - V(s) = TD error from critic.
- "Actor-critic combines the best of both worlds: the actor learns a policy directly, and the critic provides a low-variance baseline."
- If beginners look confused: "The actor decides what to do. The critic evaluates how well it did."
- If experts look bored: "This is the foundation for all modern policy gradient methods. The critic reduces variance by orders of magnitude compared to REINFORCE."
  **Transition**: "Scale this to parallel environments..."

---

## Slide 86: A2C and A3C

**Time**: ~2 min
**Talking points**:

- A2C: synchronous, GPU-friendly batched updates.
- A3C: asynchronous workers, natural exploration through diversity. Historically important but largely superseded.
- "A2C with PPO's clipped objective dominates in practice. A3C is historically important but less used now."
  **Transition**: "PPO -- the most important RL algorithm today..."

---

## Slide 87: PPO: Proximal Policy Optimisation

**Time**: ~3 min
**Talking points**:

- Probability ratio r_t(theta) = pi_new / pi_old. Clipped objective.
- "PPO is used everywhere -- ChatGPT, game-playing agents, robotics. The key idea: if the policy changes too much, clip the objective."
- Epsilon typically 0.1-0.2.
- If beginners look confused: "PPO says: improve, but not too fast. Take small, safe steps."
- If experts look bored: "PPO approximates the trust region constraint from TRPO with a simple clip, avoiding the expensive conjugate gradient computation."

**PAUSE** -- this is the most important RL algorithm. Ask: "This is what powers RLHF. The policy is the language model, the action is the next token. Does that connection make sense?"

**Transition**: "Why does clipping work?"

---

## Slide 88: PPO: Why Clipping Works

**Time**: ~2 min
**Talking points**:

- When advantage > 0 (good action): clip at 1+epsilon prevents over-committing.
- When advantage < 0 (bad action): clip at 1-epsilon prevents over-avoiding.
- "The min operation takes the more pessimistic estimate, ensuring the policy update is never dangerously large."
- If beginners look confused: "It prevents the model from swinging wildly in either direction."
  **Transition**: "How do we estimate the advantage accurately?"

---

## Slide 89: GAE: Generalised Advantage Estimation

**Time**: ~2 min
**Talking points**:

- GAE interpolates between TD(0) and MC advantage using lambda.
- Lambda=0: TD(0) advantage (high bias, low variance). Lambda=1: MC advantage (low bias, high variance). Lambda=0.95-0.99: practical sweet spot.
- "GAE interpolates between TD and Monte Carlo advantage estimation using lambda. In practice, lambda around 0.95 gives the best balance."
  **Transition**: "Let me put the PPO loop together..."

---

## Slide 90: PPO Training Loop

**Time**: ~2 min
**Talking points**:

- Five steps: collect trajectories, compute GAE advantages, multiple epochs of SGD on clipped objective, update old policy, repeat.
- "PPO is on-policy but squeezes maximum value from each batch by doing multiple optimisation epochs."
- Key detail: clipping ensures multiple passes do not move the policy too far.
  **Transition**: "PPO is the standard. But for continuous control, there is an alternative..."

---

## Slide 91: SAC: Soft Actor-Critic

**Time**: ~3 min
**Talking points**:

- Maximum entropy RL: maximise reward AND entropy. Temperature alpha controls the trade-off.
- Why entropy? Encourages exploration, prevents premature convergence, robust to perturbations.
- Architecture: off-policy, twin Q-networks, automatic temperature tuning, continuous action spaces.
- "SAC adds entropy to the reward, encouraging the policy to be as random as possible while still achieving high reward. SAC is the go-to for continuous control."
- If beginners look confused: "SAC says: do well, but also stay flexible. Do not commit too hard to one strategy."
- If experts look bored: "SAC's automatic temperature tuning via constrained optimisation is elegant -- it targets a specific entropy level rather than requiring manual alpha selection."
  **Transition**: "What if we treat RL as a sequence problem?"

---

## Slide 92: Decision Transformer

**Time**: ~2 min
**Talking points**:

- RL as sequence modelling. Trajectory: (return-to-go, state, action, ...).
- "Condition on the desired return. At test time, specify a high return-to-go and the transformer generates actions to achieve it."
- No Bellman equation, no TD error, no policy gradient.
- If beginners look confused: "Instead of learning what to do, learn what actions lead to specific scores. Then ask for a high score."
- If experts look bored: "Decision Transformer shows that a well-trained sequence model can match or exceed TD-based methods on several benchmarks, raising questions about whether Bellman backup is necessary."
  **Transition**: "What if we cannot interact with the environment at all?"

---

## Slide 93: Offline RL: Learning Without Interaction

**Time**: ~2 min
**Talking points**:

- Many domains: interaction is dangerous (healthcare, driving, finance).
- Key methods: CQL (conservative Q), IQL (implicit Q), Decision Transformer.
- Core challenge: distribution shift between data collection policy and learned policy.
- "Offline RL is critical for real-world applications where exploration is dangerous. These methods learn from fixed datasets while avoiding overconfidence in unseen actions."
  **Transition**: "Now the grand connection..."

---

## Slide 94: RL Meets Alignment: The Full Circle

**Time**: ~2 min
**Talking points**:

- RLHF = PPO where: policy = language model, state = prompt + tokens so far, action = next token, reward = human preference score, episode = one complete generation.
- "This is the grand connection. RLHF IS reinforcement learning. The language model IS the policy. Token generation IS action selection. PPO's clipping prevents the model from deviating too far."
- "Everything you learned in this section -- MDPs, policy gradients, PPO, clipping, GAE -- is exactly what runs inside the RLHF pipeline from Section B."
- If beginners look confused: "The RL we just studied IS the mechanism that aligns language models. They are the same thing."
- If experts look bored: "GRPO removed the critic (Section B) by using group statistics. DPO removed both critic and the RL loop entirely. The progression: RLHF -> GRPO -> DPO is a simplification arc."

**PAUSE** -- let the connection land. This is the conceptual climax of the RL section.

**Transition**: "Here is RL in Kailash..."

---

## Slide 95: Kailash RLTrainer

**Time**: ~2 min
**Talking points**:

- Walk the code: RLTrainer with algorithm="PPO", CartPole-v1, standard hyperparameters.
- "Kailash RLTrainer wraps Stable-Baselines3. The same engine handles PPO, SAC, and DQN."
- Point out how the hyperparameters (clip_range, gae_lambda, n_epochs) map to theory.
  **Transition**: "How do you choose the right algorithm?"

---

## Slide 96: RL Algorithm Decision Guide

**Time**: ~1 min
**Talking points**:

- Quick reference table: discrete actions -> DQN, continuous -> SAC, LLM alignment -> PPO, fixed dataset -> CQL/IQL, general purpose -> PPO.
- "PPO is the safe default. SAC for continuous actions. DQN for discrete. Offline methods when you cannot interact with the environment."
  **Transition**: "What about multiple agents?"

---

## Slide 97: Multi-Agent RL

**Time**: ~2 min
**Talking points**:

- Cooperative: shared team reward, CTDE (centralised training, decentralised execution).
- Competitive: self-play, minimax.
- "Multi-agent RL is where RL meets the multi-agent systems you built with Kaizen in Module 9."
- Kaizen connection: Delegate pattern with shared objectives.
- If beginners look confused: "What happens when multiple AI agents need to work together or compete?"
- If experts look bored: "CTDE is the dominant paradigm because it sidesteps the non-stationarity problem of independent learners."
  **Transition**: "Reward design is critical and dangerous..."

---

## Slide 98: Reward Shaping & Reward Hacking

**Time**: ~2 min
**Talking points**:

- Reward shaping: add intermediate rewards to guide learning. Must preserve optimal policy. Potential-based shaping.
- Reward hacking: agent exploits loopholes. Maximises reward without achieving intent.
- "Goodhart's Law in RL: when a measure becomes a target, it ceases to be a good measure."
- "Reward shaping accelerates learning by giving the agent hints. But badly designed rewards lead to reward hacking. This is a major challenge in RLHF as well."
- If beginners look confused: "The agent is very clever at finding shortcuts. If there is a loophole in your reward, it will find it."
- If experts look bored: "Potential-based shaping is the only form guaranteed to preserve the optimal policy under the original reward."
  **Transition**: "Beyond epsilon-greedy..."

---

## Slide 99: Exploration Strategies Beyond epsilon-Greedy

**Time**: ~2 min
**Talking points**:

- Walk the table: epsilon-greedy, Boltzmann, UCB, intrinsic motivation, noisy networks.
- "Epsilon-greedy is the simplest strategy but far from the best. UCB balances exploration and exploitation using uncertainty. Intrinsic motivation adds a curiosity reward that encourages the agent to seek novel states."
  **Transition**: "What goes wrong in practice?"

---

## Slide 100: RL in Practice: Common Pitfalls

**Time**: ~2 min
**Talking points**:

- Walk the table: sparse rewards, non-stationary environments, high-dimensional actions, sample inefficiency, sim-to-real gap.
- "These are the challenges you will face in real RL applications. The sim-to-real gap is particularly important for robotics."
- If beginners look confused: "RL in theory is elegant. RL in practice is hard. This table tells you what to watch out for."
  **Transition**: "What if we learn a model of the environment?"

---

## Slide 101: Model-Based RL

**Time**: ~2 min
**Talking points**:

- Model-free (what we covered): learn directly, no model, sample-inefficient but robust.
- Model-based: learn P(s'|s,a), plan using learned model, sample-efficient, but model errors compound.
- World models: Dreamer, MuZero, IRIS.
- "Model-based RL is the frontier for sample-efficient learning. Instead of learning purely from experience, the agent builds an internal model of the world and plans within it."
- If beginners look confused: "Model-free: learn by doing. Model-based: learn by imagining."
- If experts look bored: "MuZero achieves superhuman Go, chess, and Atari without even knowing the rules. It learns the dynamics model implicitly."
  **Transition**: "Let me give you the complete RL taxonomy..."

---

## Slide 102: RL Taxonomy: The Complete Map

**Time**: ~1 min
**Talking points**:

- Walk the table: dynamic programming, Monte Carlo, TD, deep Q-learning, policy gradient, proximal methods, maximum entropy, offline RL, model-based.
- "This is the complete RL landscape. You have covered every major category. The choice depends on your constraints: model availability, action space, sample budget, and safety requirements."
  **Transition**: "Section D: from trained models to deployed models."

---

## Slide 103: D. Model Merging & Export

**Time**: ~1 min
**Talking points**:

- Pipeline: fine-tune -> merge -> quantize -> export -> deploy.
- "Model merging and export bridge training and deployment. You might have multiple LoRA adapters and need to combine them into a single deployable model."
  **Transition**: "How do you merge models?"

---

## Slide 104: Linear Merge & SLERP

**Time**: ~2 min
**Talking points**:

- Linear: W_merged = alpha _ W_A + (1-alpha) _ W_B. Simple, fast.
- SLERP: spherical interpolation. Preserves weight norms. Better for diverse models.
- "Linear merge is just averaging. SLERP interpolates on the unit sphere, preserving magnitude. SLERP tends to produce better results when models have diverged significantly."
- If beginners look confused: "Linear merge is mixing two paints. SLERP is a smarter mix that preserves the intensity of each colour."
  **Transition**: "What about conflicting adapters?"

---

## Slide 105: TIES: Trim, Elect, Merge

**Time**: ~2 min
**Talking points**:

- Three steps: trim small-magnitude changes, elect sign by majority vote, merge agreeing parameters.
- "Naive averaging cancels out important task-specific changes when models disagree on direction. TIES resolves conflicts before merging."
- If beginners look confused: "TIES removes noise, resolves disagreements, then averages. It is a cleaned-up merge."
  **Transition**: "DARE takes a different approach..."

---

## Slide 106: DARE: Drop And REscale

**Time**: ~2 min
**Talking points**:

- Randomly drop about 90% of the fine-tuning delta and rescale the rest.
- "Most fine-tuned weight changes are redundant. Randomly dropping about 90% and rescaling the rest preserves performance while reducing interference during merging. Similar to dropout but for weight deltas."
- If beginners look confused: "Like pruning a plant -- removing most branches makes the remaining ones stronger."
  **Transition**: "Once merged, how do you shrink the model for deployment?"

---

## Slide 107: Quantization for Deployment

**Time**: ~2 min
**Talking points**:

- Walk the table: FP32, FP16, INT8, INT4, GGUF. Sizes for a 7B model.
- "Quantization is how you get a 7B model running on a laptop. INT4 is aggressive but often surprisingly good with calibration."
- GPTQ vs AWQ: AWQ identifies important weights and preserves them at higher precision. AWQ often has better quality at the same bit width.
  **Transition**: "You can also make models smaller structurally..."

---

## Slide 108: Distillation & Pruning

**Time**: ~2 min
**Talking points**:

- Knowledge distillation: train small student from teacher's soft probability targets. "Dark knowledge."
- Pruning: remove unimportant weights. Structured (entire neurons) or unstructured (individual weights).
- "Distillation creates a genuinely smaller model. Pruning keeps the architecture but makes it sparse. Both reduce inference cost."
- If beginners look confused: "Distillation: train a small student by watching a big teacher. Pruning: cut away the parts of the model that are not doing much."
  **Transition**: "Kailash AdapterRegistry manages all of this..."

---

## Slide 109: Kailash AdapterRegistry

**Time**: ~2 min
**Talking points**:

- Walk the code: register adapters, merge with method="ties", quantize with AWQ, export as GGUF.
- "AdapterRegistry manages the full lifecycle: register, version, merge, quantize, export."
  **Transition**: "Where does each format go?"

---

## Slide 110: Model Export Summary

**Time**: ~1 min
**Talking points**:

- Walk the table: cloud API -> Safetensors, edge -> GGUF, production Python -> ONNX, Kailash -> ModelRegistry.
- "Each step is one Kailash engine. The deployment pipeline is fully integrated."
- Kailash integration: Align -> AdapterRegistry -> ModelRegistry -> InferenceServer -> Nexus.
  **Transition**: "There is a simpler merging trick..."

---

## Slide 111: Model Soups: Checkpoint Averaging

**Time**: ~2 min
**Talking points**:

- Average weights from K checkpoints during training or K hyperparameter configs.
- "Model soups are surprisingly effective. Train the same model with 5 different learning rates, average the final checkpoints, and you often get better performance than any individual run."
- Why it works: checkpoints lie in a connected low-loss region. Averaging produces a model closer to the center.
- If beginners look confused: "Train several times, average the results. It is like asking multiple experts and taking the consensus."
  **Transition**: "For classical ML models, ONNX is the standard..."

---

## Slide 112: ONNX Export & Runtime

**Time**: ~2 min
**Talking points**:

- ONNX: framework-agnostic, optimised runtime, hardware-specific acceleration.
- Kailash code: ModelRegistry.load -> export_onnx.
- "ONNX is the bridge between training frameworks and production runtimes. Kailash ModelRegistry handles the conversion, and InferenceServer serves ONNX models with automatic batching."
- Connection to M7 OnnxBridge.
  **Transition**: "Edge or cloud?"

---

## Slide 113: Serving Architecture: Edge vs Cloud

**Time**: ~2 min
**Talking points**:

- Walk the comparison table: latency, model size, privacy, cost, scalability.
- "The choice between edge and cloud depends on your constraints. Privacy-sensitive applications favour edge. High-throughput applications favour cloud. Kailash Nexus can serve both."
  **Transition**: "Now the critical section. Section E: AI Governance & Regulation."

---

## Slide 114: E. AI Governance & Regulation

**Time**: ~2 min
**Talking points**:

- Four points: EU AI Act is law (not guidelines), fines up to 7%, Singapore AI Verify, every deployed model needs an audit trail.
- "Governance is no longer a nice-to-have. The EU AI Act is law. Your models need governance or they cannot be deployed in regulated markets."
- "'We did not know' is not a defence."
- If beginners look confused: "From here on, we learn the rules that every production AI system must follow."
- If experts look bored: "The governance section covers both regulatory compliance and the PACT framework for operational governance."

**PAUSE** -- ask: "How many of you have deployed a model without any governance process? Be honest. By the end of this section, you will never do that again."

**Transition**: "Let me show you the risk tiers..."

---

## Slide 115: EU AI Act: Risk Tiers

**Time**: ~2 min
**Talking points**:

- Four tiers: unacceptable (banned), high risk (conformity assessment), limited risk (transparency), minimal risk (no requirements).
- "The four tiers determine your obligations. Unacceptable risk is banned outright. High risk requires conformity assessments. Limited risk needs transparency labels. Minimal risk has no requirements."
- If beginners look confused: "Think of it like food safety ratings. Some things are banned, some need inspection, some just need a label."
  **Transition**: "Which articles matter most?"

---

## Slide 116: EU AI Act: Key Articles

**Time**: ~2 min
**Talking points**:

- Art. 6: high-risk classification. Art. 9: risk management system. Art. 13: transparency. Art. 52: disclosure.
- Fines: up to 35M EUR or 7% for prohibited practices, 15M or 3% for other violations.
- "Art. 9 is the most operationally demanding -- it requires an ongoing risk management system, not just a one-time assessment."
  **Transition**: "What about general-purpose AI?"

---

## Slide 117: GPAI: General-Purpose AI Obligations

**Time**: ~2 min
**Talking points**:

- Walk the table: all GPAI need documentation, copyright compliance, data summary. Systemic risk GPAI additionally need evaluation, adversarial testing, incident reporting.
- Systemic risk threshold: >10^25 FLOPs or designated by AI Office.
- "If you deploy or fine-tune general-purpose AI models, you need to know these obligations. Downstream deployers inherit obligations for high-risk applications."
  **Transition**: "Singapore takes a different approach..."

---

## Slide 118: Singapore: AI Verify & ISAGO 2.0

**Time**: ~2 min
**Talking points**:

- AI Verify: open-source testing framework, 11 governance principles, automated testing toolkit. Currently voluntary.
- ISAGO 2.0: GenAI governance companion, model evaluation methodology, incident management.
- "Singapore takes a pragmatic approach with concrete tools for testing AI systems. The framework is voluntary but increasingly expected for government procurement."
  **Transition**: "The global landscape..."

---

## Slide 119: Global AI Governance Landscape

**Time**: ~1 min
**Talking points**:

- Quick reference table: EU (risk-based law), Singapore (voluntary + procurement), US (executive orders + NIST), China (content + algorithm rules), ISO 42001 (management standard).
- "Every major jurisdiction is developing AI governance. The EU leads with binding law. ISO provides a universal management standard."
  **Transition**: "PACT is how Kailash operationalises governance..."

---

## Slide 120: PACT: Governance by Design

**Time**: ~2 min
**Talking points**:

- D/T/R accountability grammar: Decider (authority), Trusted (executes), Responsible (accountable for outcome).
- "PACT provides the grammar for governance. D/T/R ensures every decision has someone who authorised it, executed it, and is accountable for the outcome."
- No orphan decisions. If any of D, T, R is missing, the action is blocked.
- If beginners look confused: "For every AI decision, we know who approved it, who did it, and who is on the hook if it goes wrong."
- If experts look bored: "D/T/R maps to RACI (Responsible, Accountable, Consulted, Informed) but is specifically designed for AI agent systems where delegation creates accountability chains."

**PAUSE** -- ask: "Think about the last AI decision at your organisation. Could you name the D, T, and R?"

**Transition**: "Here is PACT in code..."

---

## Slide 121: PACT GovernanceEngine

**Time**: ~2 min
**Talking points**:

- Walk the code: GovernanceEngine, compile_org with roles and clearances, can_access, explain_access.
- "GovernanceEngine compiles your org structure into executable access control. can_access returns a boolean. explain_access returns a human-readable explanation."
- The explain_access output tells you exactly why access was denied and who to escalate to.
  **Transition**: "Envelopes constrain what agents can do..."

---

## Slide 122: Operating Envelopes

**Time**: ~2 min
**Talking points**:

- effective_envelope = role_envelope intersect task_envelope. Always the most restrictive combination.
- Monotonic tightening: envelopes can only get tighter. An agent cannot grant itself more permissions. Delegation always reduces the envelope.
- Fail-closed: if envelope check fails, block. Never default to permissive.
- "The effective envelope is always the intersection -- the most restrictive combination. An agent can never escalate its own permissions."
- If beginners look confused: "An agent gets the SMALLEST set of permissions from all the rules that apply to it."
- If experts look bored: "Monotonic tightening is a lattice structure. The meet operation on envelopes guarantees the security property."
  **Transition**: "Envelopes also control data access..."

---

## Slide 123: Knowledge Clearance Levels

**Time**: ~2 min
**Talking points**:

- Walk the five levels: public, internal, confidential, secret, critical.
- "Knowledge clearance prevents data leakage through AI systems. A customer-facing chatbot at Level 1 cannot access Level 3 customer data."
- If beginners look confused: "Like security clearance in government. Higher level = more sensitive data."
  **Transition**: "Wrapping agents with governance..."

---

## Slide 124: PactGovernedAgent

**Time**: ~2 min
**Talking points**:

- Walk the code: PactGovernedAgent wraps BaseAgent with clearance, allowed/blocked tools, budget, D/T/R addresses.
- "PactGovernedAgent wraps any Kaizen agent with governance controls. The agent has a clearance level, budget, allowed tools, and D/T/R addresses."
- Every action the agent takes is checked against these constraints.
  **Transition**: "What happens when budgets run out?"

---

## Slide 125: Budget Cascading & Failure Handling

**Time**: ~2 min
**Talking points**:

- Budget cascading: parent allocates to children, each gets a subset, exhaustion stops gracefully, total never exceeds parent.
- Failure handling: budget exceeded (stop + report), tool blocked (log + deny), clearance violation (escalate to D), dereliction (agent fails task).
- "Budget cascading ensures cost control in multi-agent systems. If a child exhausts its budget, it stops -- it does not consume the parent's remaining budget."
  **Transition**: "Governance also means responsible AI..."

---

## Slide 126: Responsible AI: Bias Auditing

**Time**: ~2 min
**Talking points**:

- Types of bias: data, algorithmic, deployment, feedback loops.
- Auditing methods: demographic parity, equalised odds, calibration per group, intersectional analysis.
- "Bias auditing is not a one-time activity. Your DriftMonitor should include fairness metrics alongside performance metrics."
- If beginners look confused: "Does the model treat all groups of people fairly? This is how you check."
- If experts look bored: "Equalised odds and demographic parity are often in tension. You need to choose which fairness criterion matters most for your application."

**PAUSE** -- ask: "What fairness metric would matter most for a hiring model? A lending model? A medical diagnostic model?"

**Transition**: "What about privacy?"

---

## Slide 127: Differential Privacy: DP-SGD

**Time**: ~2 min
**Talking points**:

- DP-SGD: clip individual gradients, add calibrated Gaussian noise.
- (epsilon, delta)-differential privacy: model outputs are statistically indistinguishable whether or not any individual was in the training set.
- "DP-SGD provides mathematical guarantees that the model cannot memorise individual training examples. Critical when training on personal data."
- If beginners look confused: "It adds noise during training so the model cannot remember any single person's data."
- If experts look bored: "The privacy-utility trade-off is controlled by epsilon. Typical production values are epsilon=1 to epsilon=10. Below 1 is very private but quality degrades."
  **Transition**: "Transparency requirements..."

---

## Slide 128: Transparency & Model Cards

**Time**: ~2 min
**Talking points**:

- Model card: intended use, training data, performance across demographics, known failures, environmental impact.
- System card: architecture, safety evaluations, risk mitigations, monitoring plan, human oversight.
- "Model cards are the minimum transparency standard. They tell users what the model can and cannot do."
- Kailash ModelRegistry supports model card metadata. EU AI Act Art. 13 requires this for high-risk systems.
  **Transition**: "Before deployment, you red team..."

---

## Slide 129: Red Teaming AI Systems

**Time**: ~2 min
**Talking points**:

- Five steps: define scope, diverse team, systematic attacks, document findings, iterate.
- Red teaming is mandatory for GPAI with systemic risk under the EU AI Act.
- "Red teaming finds failures before users do. This includes prompt injection, training data extraction, bias exploitation, and misuse scenarios."
- If beginners look confused: "Hire people to try to break your AI. Fix what they find."
  **Transition**: "Here is your pre-deployment checklist..."

---

## Slide 130: Governance Checklist: Production Deployment

**Time**: ~1 min
**Talking points**:

- Walk the eight-row checklist: accountability, access control, budget, bias, privacy, transparency, monitoring, incident response.
- "Every production AI system should pass every row before serving its first user."
- Each row maps to a specific Kailash tool.
  **Transition**: "For ISO certification..."

---

## Slide 131: ISO/IEC 42001: AI Management System

**Time**: ~2 min
**Talking points**:

- First international standard for AI management systems. Covers risk management, data governance, monitoring, stakeholder communication, third-party AI.
- Internationally recognised, certifiable, maps to EU AI Act, framework-agnostic.
- "ISO/IEC 42001 is the first international standard for AI management systems. PACT operationalises the operational layer. If your organisation is pursuing ISO certification, PACT's audit trails and governance structures map directly."
- If beginners look confused: "ISO 42001 is like ISO 9001 but for AI. It is the quality standard that auditors check."
  **Transition**: "Where does your organisation stand?"

---

## Slide 132: AI Governance Maturity Model

**Time**: ~2 min
**Talking points**:

- Five levels: ad hoc, reactive, defined, managed, optimised.
- "Most organisations are at Level 1 or 2. PACT leapfrogs you to Level 5 by making governance part of the code, not a separate manual process."
- PACT + Kailash = Level 5: governance compiled into every workflow, automated enforcement, continuous audit trails, proactive risk detection via DriftMonitor.
- If beginners look confused: "Level 1: no rules. Level 5: rules are built into the software automatically."
  **Transition**: "What gets recorded in the audit trail?"

---

## Slide 133: Audit Trail: What Gets Logged

**Time**: ~2 min
**Talking points**:

- Walk the JSON audit event: timestamp, agent, action, D/T/R addresses, clearance required/held, budget used/remaining, result, input/output hashes.
- "Every action is logged with full provenance: who authorised it, who executed it, who is accountable, what clearance was required, and what the cost was. This is what compliance auditors need."
- If beginners look confused: "Every single AI decision creates a receipt. You can always go back and check what happened and who was responsible."
  **Transition**: "What happens when agents fail?"

---

## Slide 134: Agent Dereliction & Escalation

**Time**: ~2 min
**Talking points**:

- Dereliction: agent fails to act, timeout, repeated errors. Triggers escalation to D address.
- Escalation protocol: log event, notify R address, escalate to D address, D reassigns or marks as blocked.
- "Dereliction handling is critical for production agent systems. If an agent fails, the system does not silently continue. It escalates to a human decision-maker."
- If beginners look confused: "If the AI breaks, a human gets notified and takes over. No silent failures."
- If experts look bored: "The dereliction -> escalation pattern implements the human-on-the-loop control model. The agent operates autonomously until failure, then a human re-enters the loop."
  **Transition**: "Let me summarise governance..."

---

## Slide 135: Governance Summary

**Time**: ~1 min
**Talking points**:

- Key takeaway: governance is not a layer you add on top. It is the foundation you build on.
- Five pillars: PACT (D/T/R accountability), operating envelopes (boundaries), knowledge clearance (data access), budget cascading (cost control), audit trails (compliance).
- "Governance is architecture, not afterthought. Build it into every workflow."
  **Transition**: "Section F: let us bring everything together in the capstone."

---

## Slide 136: F. Capstone: Full-Stack AI with Kailash

**Time**: ~1 min
**Talking points**:

- All 8 packages in one pipeline: Core SDK -> DataFlow -> ML -> Align -> Kaizen -> PACT -> Nexus.
- "This is where everything comes together. Every Kailash package in a single pipeline."
  **Transition**: "Here is the pattern..."

---

## Slide 137: Capstone Pattern

**Time**: ~2 min
**Talking points**:

- Seven steps: train (TrainingPipeline/AlignEngine), register (ModelRegistry), workflow (Core SDK), agent (Kaizen), govern (PACT), deploy (Nexus), monitor (DriftMonitor).
- "Seven steps from raw model to production-governed AI. Each step uses a dedicated Kailash engine."
- If beginners look confused: "This is the recipe for a complete, production-ready AI system."
  **Transition**: "Let me show you the code..."

---

## Slide 138: Capstone Code: The Full Pipeline

**Time**: ~2 min
**Talking points**:

- Walk through the imports: every import is a different Kailash package.
- Train -> register -> govern -> deploy. Each line maps to a pipeline step.
- "Every import is a different Kailash package. Every line maps to a pipeline step. This is production ML engineering."
- If beginners look confused: "Seven imports, seven packages, one integrated system."
  **Transition**: "Here is the complete platform map..."

---

## Slide 139: Kailash Platform Map

**Time**: ~1 min
**Talking points**:

- Walk the table: Core SDK (workflow orchestration), DataFlow (zero-config DB), ML (13 engines + RLTrainer), Align (AlignEngine, AdapterRegistry), Kaizen (agents), PACT (governance), Nexus (deployment).
- "Seven packages. Each solves a distinct part of the ML lifecycle. You have now used every one."
  **Transition**: "Is the system production-ready?"

---

## Slide 140: Production Readiness Checklist

**Time**: ~1 min
**Talking points**:

- Two columns: technical (versioned, latency SLA, drift alerts, fallback, load tested) and governance (D/T/R, clearance, budget, model card, bias audit, red team report).
- "Technical readiness ensures it works. Governance readiness ensures it is safe and compliant."
  **Transition**: "Here is the architecture diagram..."

---

## Slide 141: Architecture: The Complete System

**Time**: ~2 min
**Talking points**:

- Walk the architecture: User -> Nexus -> PactGovernedAgent -> Kaizen Agent -> Core SDK Workflow -> ML/Align Engines -> ModelRegistry -> DriftMonitor -> DataFlow.
- "The complete Kailash architecture. Requests enter through Nexus, pass through PACT governance, are handled by agents, which orchestrate ML and Align engines. Everything is versioned, monitored, and auditable."
- If beginners look confused: "This is the complete picture. Every box is a package you have learned."
  **Transition**: "Here is your capstone exercise..."

---

## Slide 142: Capstone Exercise Preview

**Time**: ~2 min
**Talking points**:

- Eight requirements: DataExplorer, FeatureEngineer, TrainingPipeline, AlignEngine (LoRA), Kaizen (ReActAgent), PACT (D/T/R + budget + clearance), Nexus (API + CLI), DriftMonitor.
- "This exercise touches every major concept from Modules 1-10. A complete, governed, production-ready AI system."
- If beginners look confused: "This is your graduation project. Use everything you have learned."

**PAUSE** -- ask: "Questions about the capstone requirements before we discuss grading?"

**Transition**: "Here is how it will be assessed..."

---

## Slide 143: Capstone: Assessment Criteria

**Time**: ~1 min
**Talking points**:

- Walk the criteria: technical correctness (30%), code quality (20%), governance (25%), architecture decisions (15%), documentation (10%).
- "Note that governance is 25% of the grade -- more than code quality. This reflects the real-world importance of governance in production AI systems."
  **Transition**: "Section G: the grand synthesis. Let us look back at the entire journey."

---

## Slide 144: G. Grand Synthesis: The Journey Complete

**Time**: ~2 min
**Talking points**:

- Walk the full M1-M10 table. Do not re-teach. Let students feel the progression.
- "Ten modules. From zero Python to building governed, aligned AI systems. Module 1 was loading a CSV. Module 10 is deriving DPO loss functions and deploying PACT-governed agents."
- If beginners look confused: "Look how far you have come."
- If experts look bored: "The progression from DataExplorer to GovernanceEngine mirrors the maturation of the field itself."
  **Transition**: "Let me show you the feature engineering thread..."

---

## Slide 145: The Feature Engineering Spectrum

**Time**: ~2 min
**Talking points**:

- Progression: manual FE (M3) -> USML (M6) -> deep learning (M7) -> transformers (M8) -> LLMs (M9-10).
- "Feature engineering is the thread that runs through the entire programme. Understanding this spectrum is what separates engineers from practitioners."
- Each stage automates more of the feature engineering.
- If beginners look confused: "M3: you handcraft features. M10: the model learns its own features. That is the progression."
- If experts look bored: "The n->1 (clustering), n->k (dim reduction), n->m (deep learning) framework from USML connects all the way to transformer self-attention learning n->n features."
  **Transition**: "Similarly, the learning paradigms form a spectrum..."

---

## Slide 146: The Learning Paradigm Spectrum

**Time**: ~2 min
**Talking points**:

- Five paradigms: supervised (M4-M5), unsupervised (M6), self-supervised (M8-M9), RL (M10), preference alignment (M10).
- "You now understand all five. Most ML engineers know only the first two. This breadth is rare and valuable."
- If beginners look confused: "You have learned five different ways that machines can learn. Most practitioners only know two."
  **Transition**: "Here is the complete platform..."

---

## Slide 147: Kailash Platform: Complete Map

**Time**: ~1 min
**Talking points**:

- Visual of all seven packages: Core SDK (140+ nodes), DataFlow, ML (13 engines), Align (12 methods), Kaizen (agents), PACT (D/T/R), Nexus (multi-channel).
- "You have used every package. From your first polars operation in M1 to PACT governance in M10."
  **Transition**: "And all 13 ML engines..."

---

## Slide 148: 13 ML Engines: Complete Reference

**Time**: ~1 min
**Talking points**:

- Quick scan of the 13-engine table with which module introduced each one.
- "Thirteen engines, each introduced at the right point. You now know when and how to use each one."
  **Transition**: "Let me tell you about certification..."

---

## Slide 149: Certification: Terrene Open Academy

**Time**: ~2 min
**Talking points**:

- Two certifications: Foundation Ascent (M1-M5) and Summit Ascent (M6-M10).
- Foundation + Summit = complete ASCENT certificate.
- "Two certifications earned independently. Foundation Ascent covers practical fundamentals. Summit Ascent adds mathematical depth and production governance. Together they form the complete ASCENT certification."
  **Transition**: "What makes this programme different?"

---

## Slide 150: What Makes This Programme Different

**Time**: ~1 min
**Talking points**:

- Two columns: traditional ML courses (theory in isolation, toy datasets, no production, no governance, no agents) vs ASCENT (theory through practice, real datasets, production deployment, governance built in, multi-agent orchestration).
- "You have not just learned ML theory. You have learned to build, deploy, and govern production AI systems using a modern open-source stack."
  **Transition**: "Here is your complete toolkit..."

---

## Slide 151: Your Toolkit

**Time**: ~1 min
**Talking points**:

- Nine capability layers: data, experiment, train, version, fine-tune, agents, govern, deploy, monitor.
- "Nine capability layers. Each one uses dedicated Kailash engines. This is a complete, professional toolkit."
  **Transition**: "Where do you go from here?"

---

## Slide 152: Where You Go From Here

**Time**: ~1 min
**Talking points**:

- Deepen: contribute to Kailash SDK, read original papers, build portfolio projects, explore the codebase.
- Apply: deploy governed AI, champion responsible AI, build production pipelines, lead transformation.
- "The programme ends here but your journey does not. You now have the foundation to go deep in any direction."
  **Transition**: "One last thing..."

---

## Slide 153: The Final Provocation

**Time**: ~2 min
**Talking points**:

- Bring the opening provocation full circle: "The question isn't whether AI will transform your industry. The question is: will YOU lead that transformation?"
- "You now have the skills to answer yes."
- Let the moment land. This is a graduation moment.
- If beginners look confused: they should not be. They should feel empowered.
- If experts look bored: they should not be. They should feel the weight of what they have accomplished.

**PAUSE** -- extended silence. Let it land. Then: "Congratulations."

**Transition**: "Let me close us out..."

---

## Slide 154: Module 10 Complete (Closing Slide)

**Time**: ~1 min
**Talking points**:

- "Governance is not a checkbox. It is the foundation that makes AI trustworthy, deployable, and sustainable."
- "Congratulations -- you have completed ASCENT: ML Engineering from Foundations to Mastery."
- Invite applause. Acknowledge the journey. Open the floor for final questions.
- Remind them of the capstone exercise deadline and certification process.

---

## Timing Summary

| Section                                 | Slides  | Time                 |
| --------------------------------------- | ------- | -------------------- |
| Opening & Recap (1-8)                   | 8       | ~10 min              |
| A. Fine-Tuning & PEFT (9-35)            | 27      | ~50 min              |
| B. Preference Alignment (36-64)         | 29      | ~58 min              |
| C. Reinforcement Learning (65-102)      | 38      | ~85 min (with break) |
| D. Model Merging & Export (103-113)     | 11      | ~20 min              |
| E. AI Governance & Regulation (114-135) | 22      | ~40 min              |
| F. Capstone Integration (136-143)       | 8       | ~12 min              |
| G. Grand Synthesis (144-154)            | 11      | ~15 min              |
| **Total**                               | **154** | **~180 min**         |

**Suggested break**: After slide 102 (end of RL section). This is the natural halfway point.

**Pacing notes**:

- Sections A and B are theory-heavy. Move briskly through code slides to save time for derivations.
- The DPO derivation (slides 43-47) is the mathematical highlight. Take your time.
- The RL section is the longest. If running behind, compress DQN variants (slide 82) and offline RL (slide 93).
- Governance (Section E) is where many students tune out. Lead with regulation consequences to maintain engagement.
- Grand Synthesis should feel celebratory, not rushed. End on a high note.
