# Module 6: Alignment, Governance & Organisational Transformation -- Speaker Notes

Total time: ~180 minutes (3 hours)

---

## Slide 1: Title Slide

**Time**: ~2 min
**Talking points**:

- Read the provocation: "Your competitors are scrambling. You'll be ready."
- Let it land. This is the final module. Everything comes together here.
- Module 6 answers three questions: How do we teach models our values? Who controls the AI? How does an agent learn from experience?
- If beginners look confused: "Today we learn how to control the AI systems we built in M5."
- If experts look bored: "We will derive the DPO loss from first principles, cover PPO clipping, and formalise operating envelopes."
  **Transition**: "Let me remind you where we have been..."

---

## Slide 2: Where We've Been

**Time**: ~2 min
**Talking points**:

- Quick table: M1 through M5. Do not re-teach. This is a reference.
- Emphasise: "You have the full ML lifecycle, agents, and deployment. What is missing is control."
  **Transition**: "What is this final frontier?"

---

## Slide 3: The Final Frontier

**Time**: ~3 min
**Talking points**:

- Three questions: How do we teach models our values (alignment)? Who controls the AI (governance)? How does an agent learn from experience (RL)?
- New engines: AlignmentPipeline, AdapterRegistry, GovernanceEngine, PactGovernedAgent, RLTrainer.
- If beginners look confused: "M5 built powerful agents. M6 teaches them right from wrong and puts them on a leash."
- If experts look bored: "We are bridging the gap between capability (M5) and safety (M6)."
  **Transition**: "Here is today's roadmap..."

---

## Slide 4: Today's Roadmap

**Time**: ~2 min
**Talking points**:

- Walk through the timeline: 6.1 SFT/LoRA, 6.2 DPO, 6.3 RL, 6.4 Merging, 6.5 PACT, 6.6 Governed Agents, 6.7 Scale, 6.8 Capstone.
- Two blocks: alignment first (teaching the model), then governance (controlling the organisation).
- If beginners look confused: "First half: how to teach a model your way of doing things. Second half: how to control who does what."
  **Transition**: "Let me start with a story about why this matters..."

---

## Slide 5: Opening Case -- The EU AI Act

**Time**: ~4 min
**Talking points**:

- Read the headline slowly: "First fines expected 2025. Up to 7% of global turnover."
- Walk through the timeline: entered into force Aug 2024, prohibited practices Feb 2025, full compliance Aug 2026.
- Mention Singapore AI Verify / ISAGO 2.0 -- voluntary but market-expected.
- The hook callout: "Governance is not compliance theatre -- it is competitive advantage."
- If beginners look confused: "Governments are now fining companies for uncontrolled AI. Today we learn how to avoid that."
- If experts look bored: "7% of global turnover is not hypothetical -- GDPR fines reached 1.2B euros. AI Act will exceed that."
  **Transition**: "Let me make this concrete..."

---

## Slide 6: Why This Module Matters

**Time**: ~3 min
**Talking points**:

- Two columns: Without governance (burns $50k, leaks PII, no audit trail) vs With governance (budgets, privacy, role-based access, audit chain).
- If beginners look confused: "Without rules, your AI is a liability. With rules, it is an asset."
- If experts look bored: "Think about the operational risk of an ungoverned agent in a regulated industry."
  **Transition**: "These are not hypothetical scenarios..."

---

## Slide 7: Real Consequences

**Time**: ~3 min
**Talking points**:

- Walk through the table: Air Canada chatbot, Samsung code leak, Italy ChatGPT ban, NYC hiring tool.
- "Every incident above is preventable with the tools you will learn today."
- PAUSE for questions. Students often have their own examples to share.
- If beginners look confused: "These are real companies that got hurt because they did not control their AI."
- If experts look bored: "The Air Canada case is particularly instructive -- the chatbot fabricated a refund policy and the airline was legally bound by it."
  **Transition**: "Let us start with the first question: how do we teach a model our values?"

---

## Slide 8: 6.1 -- Supervised Fine-Tuning

**Time**: ~3 min
**Talking points**:

- "A pre-trained model already knows language. Fine-tuning is on-the-job training."
- Two columns: Pre-trained (knows grammar, facts, reasoning) vs Fine-tuned (knows YOUR tone, YOUR domain, YOUR format).
- If beginners look confused: "The model already speaks the language. Fine-tuning teaches it your company's way of speaking."
- If experts look bored: "SFT is the simplest alignment method -- direct imitation learning on demonstrations."
  **Transition**: "The question is: do we need to update ALL the parameters?"

---

## Slide 9: Full Fine-Tuning vs LoRA

**Time**: ~3 min
**Talking points**:

- Two columns: Full (update ALL parameters, 28 GB, multiple GPUs) vs LoRA (freeze original, add tiny matrices, 0.1-1%, single GPU).
- The analogy is brilliant: "Full fine-tuning rewrites the textbook. LoRA adds sticky notes to the pages that matter."
- If beginners look confused: Use the sticky notes analogy.
- If experts look bored: "The parameter efficiency ratio is the key insight -- why update 100% when 1% suffices?"
  **Transition**: "Let us see the math behind LoRA..."

---

## Slide 10: LoRA -- The Core Idea

**Time**: ~3 min (THEORY)
**Talking points**:

- The equation: W = W_0 + BA where B is d-by-r and A is r-by-d, with r much less than d.
- Only B and A are trained. W_0 stays frozen.
- If beginners look confused: "Instead of changing the whole weight matrix, we add a small correction."
- If experts look bored: "The rank r is the key hyperparameter. Too low: underfitting. Too high: overfitting and losing efficiency."

---

## Slide 11: LoRA -- Matrix Dimensions

**Time**: ~3 min (THEORY)
**Talking points**:

- Walk through the visual: W_0 (4096x4096 = 16.7M), B (4096x16 = 65K), A (16x4096 = 65K).
- "Trainable: 130K vs 16.7M -- 0.78% of original parameters."
- If beginners look confused: "The big box is frozen. The two small boxes are what we actually train."
- If experts look bored: "The rank choice (16 in this example) balances expressiveness and efficiency."

---

## Slide 12: Why Does Low Rank Work?

**Time**: ~3 min (THEORY)
**Talking points**:

- Aghajanyan et al. (2020): pre-trained models have low intrinsic dimensionality.
- Most singular values of the weight update are near zero.
- Connection to PCA from M4.3: "Just as PCA finds that most variance lives in a few components, LoRA exploits low-rank structure."
- If beginners look confused: "The changes needed for fine-tuning live in a small subspace. LoRA targets that subspace."
- If experts look bored: "The intrinsic dimensionality paper gives concrete numbers -- often only a few hundred dimensions matter."

---

## Slide 13: LoRA -- Forward Pass

**Time**: ~2 min (THEORY)
**Talking points**:

- h = W_0 x + BAx = (W_0 + BA)x. At deployment: merge to W_merged = W_0 + BA. Zero inference overhead.
- Key insight: A initialised with Kaiming normal, B to zero. So Delta W = 0 at start.
- If beginners look confused: "At serving time, you merge the sticky notes into the textbook. No extra cost."
- If experts look bored: "The scaling factor alpha/r controls the update magnitude."

---

## Slide 14: QLoRA -- 4-Bit Training

**Time**: ~3 min (THEORY)
**Talking points**:

- Three innovations: NF4 quantization, double quantization, paged optimizers.
- Base weights in 4-bit, LoRA adapters in 16-bit.
- "Fine-tune a 65B model on a single 48GB GPU."
- If beginners look confused: "QLoRA compresses the frozen weights so they take less memory, making it possible to run on smaller GPUs."
- If experts look bored: "NF4 optimal spacing for normally distributed weights is a clever statistical insight."

---

## Slide 15: QLoRA Memory Comparison

**Time**: ~2 min
**Talking points**:

- Walk through the table: Full (120 GB for 7B) vs LoRA (28 GB) vs QLoRA (6 GB).
- "QLoRA turns a data centre job into something you can run on a gaming laptop."
- If beginners look confused: "From needing a server room to needing just a laptop."
- This slide is a showstopper. Let the numbers sink in.

---

## Slide 16: Adapter Lifecycle

**Time**: ~2 min
**Talking points**:

- Timeline: Train, Evaluate, Merge, Quantize (GGUF), Deploy.
- Key points: adapters are small files (10-100 MB), stack multiple on same base, A/B test in production, roll back easily.
- If beginners look confused: "Think of adapters like interchangeable lenses on a camera."

---

## Slide 17: Kailash Bridge -- AlignmentPipeline

**Time**: ~3 min
**Talking points**:

- Show the code: AlignmentConfig with method="sft", lora_rank, quantization="nf4", then AdapterRegistry.
- SWITCH TO LIVE CODING if time allows.
- If beginners look confused: "All that LoRA/QLoRA theory? Six lines of code in Kailash."
  **Transition**: "SFT teaches the model to follow instructions. But is that enough?"

---

## Slide 18: 6.2 -- Preference Alignment

**Time**: ~3 min
**Talking points**:

- "A model that predicts well is not necessarily helpful, harmless, or honest."
- Alignment = making AI do what humans WANT, not what they literally SAID.
- The analogy: "A perfectly obedient employee who follows every instruction literally is dangerous."
- If beginners look confused: "SFT teaches format. Alignment teaches values."
- If experts look bored: "This is the core HHH (helpful, harmless, honest) objective from Anthropic."
  **Transition**: "The original approach was RLHF..."

---

## Slide 19: The RLHF Pipeline

**Time**: ~3 min
**Talking points**:

- Three steps: SFT, Reward Model, PPO.
- The problem callout: "Step 2 + 3 require two extra models, are unstable, and sensitive to hyperparameters."
- If beginners look confused: "The original method works but is complicated and expensive. DPO simplifies it."

---

## Slide 20: DPO -- Skip the Reward Model

**Time**: ~3 min
**Talking points**:

- Two columns: RLHF (3 steps, 3 models, unstable) vs DPO (1 step after SFT, 1 model + reference, stable).
- "All you need: pairs of (prompt, preferred response, rejected response)."
- If beginners look confused: "Instead of training a separate model to score responses, DPO learns directly from human preferences."
- If experts look bored: "The elegance is in eliminating the reward model -- the policy IS the reward model."
  **Transition**: "Let me derive why this works..."

---

## Slide 21: DPO Derivation Step 1 -- Bradley-Terry Model

**Time**: ~3 min (THEORY)
**Talking points**:

- Bradley-Terry: probability of preference depends on difference in rewards.
- "The probability that a human prefers y_1 over y_2 depends on the DIFFERENCE in rewards."
- If beginners look confused: "It is like a chess rating system -- the difference in ratings predicts who wins."
- If experts look bored: "The Bradley-Terry model has deep connections to logistic regression and Elo ratings."
  **Transition**: "Step 2: the RLHF objective..."

---

## Slide 22: DPO Derivation Step 2 -- RLHF Objective

**Time**: ~2 min (THEORY)
**Talking points**:

- KL-constrained reward maximisation. Beta controls drift from reference.
- "The KL penalty prevents the model from gaming the reward model."
- If beginners look confused: "We want the model to improve but not change too much from what it already knows."

---

## Slide 23: DPO Derivation Step 3 -- Optimal Policy

**Time**: ~3 min (THEORY)
**Talking points**:

- Closed-form optimal policy: pi* proportional to pi_ref * exp(reward/beta).
- "The KL-constrained objective has a closed-form solution."
- If beginners look confused: "There is an exact mathematical answer for the best possible policy."
- If experts look bored: "This is a standard result from KL-regularised optimisation."

---

## Slide 24: DPO Derivation Step 4 -- Solve for Reward

**Time**: ~2 min (THEORY)
**Talking points**:

- Rearrange to express reward in terms of policy.
- Key move: take log, solve for r(x,y).
- If beginners look confused: "We flip the equation around so we can plug it into the preference model."

---

## Slide 25: DPO Derivation Step 5 -- Substitute into Bradley-Terry

**Time**: ~3 min (THEORY)
**Talking points**:

- Substitute into Bradley-Terry. The partition function Z(x) cancels.
- "The intractable partition function drops out. Preferences are expressed entirely in terms of policy log-ratios."
- This is the key insight of DPO. Emphasise the cancellation.
- If beginners look confused: "The hard math simplifies beautifully -- the complicated parts cancel out."
- If experts look bored: "The cancellation of Z(x) is what makes DPO tractable -- it would be intractable otherwise."

---

## Slide 26: DPO Derivation Step 6 -- The DPO Loss

**Time**: ~3 min (THEORY)
**Talking points**:

- The final DPO loss in the result box. This is the payoff of the derivation.
- "A simple binary cross-entropy loss on log-probability ratios. No reward model. No RL loop."
- If beginners look confused: "All that derivation produced a simple formula that just says: make good responses more likely, bad responses less likely."
- If experts look bored: "The simplicity of the final loss is remarkable given the complexity of the derivation."
- PAUSE for questions. This is one of the most important derivations in the course.
  **Transition**: "What does this loss actually do?"

---

## Slide 27: DPO -- What It Means

**Time**: ~2 min
**Talking points**:

- Three bullets: increase preferred, decrease rejected, relative to reference.
- The plain English callout: "Make the model more likely to say things humans prefer, less likely to say things humans dislike."
- If beginners look confused: This slide IS the simplified explanation.

---

## Slide 28: GRPO -- Group Relative Policy Optimisation

**Time**: ~3 min (THEORY)
**Talking points**:

- DeepSeek-R1's approach: no reward model, no reference model.
- Generate G outputs, score with simple reward, normalise within group.
- If beginners look confused: "GRPO is like DPO but even simpler -- compare responses within a group, no reference model needed."
- If experts look bored: "The group normalisation trick is elegant -- it turns an absolute reward into a relative advantage."
- SKIPPABLE if running short.

---

## Slide 29: Beyond DPO

**Time**: ~2 min (ADVANCED)
**Talking points**:

- Table: ORPO, SimPO, KTO, IPO.
- Trend: "Moving away from paired preferences toward unpaired signals and self-play."
- SKIPPABLE if running short.

---

## Slide 30: LLM Evaluation

**Time**: ~3 min (THEORY)
**Talking points**:

- Table: Perplexity, BLEU, ROUGE, BERTScore, LLM-as-Judge. Each with pitfalls.
- "Always use multiple metrics. No single number captures quality."
- If beginners look confused: "There is no perfect way to measure AI quality. Use several measures."

---

## Slide 31: LLM-as-Judge Biases

**Time**: ~2 min (THEORY)
**Talking points**:

- Four biases: position, verbosity, self-enhancement, authority.
- Mitigations: randomise position, concise rubrics, diverse judges, validate against humans.
- If beginners look confused: "Even AI judges have biases. You need to account for them."
- SKIPPABLE if running short.

---

## Slide 32: Kailash Bridge -- DPO & Evaluation

**Time**: ~2 min
**Talking points**:

- Show the code: AlignmentPipeline with method="dpo" and evaluator.evaluate().
- SWITCH TO LIVE CODING if time allows.
  **Transition**: "Now let us shift to reinforcement learning..."

---

## Slide 33: 6.3 -- Reinforcement Learning

**Time**: ~3 min
**Talking points**:

- "An agent interacts with an environment. Actions, states, rewards. Trial and error."
- Dog training analogy: "You do not tell the dog HOW to sit. You reward it when it does."
- If beginners look confused: "RL is learning by doing, not learning by example."
- If experts look bored: "RL has a fundamentally different data-generation process than supervised learning."
- This is a good break point (~75 minutes in). PAUSE for a 10-minute break.

---

## Slide 34: RL Vocabulary

**Time**: ~3 min
**Talking points**:

- Walk through the table: State, Action, Reward, Policy, Value, Q-value, Discount.
- "Plain English" column is for beginners. Symbol column is for experts.
- If beginners look confused: "Think of a video game. State = what you see. Action = what you press. Reward = points."
  **Transition**: "The Bellman equation is the foundation of all value-based RL..."

---

## Slide 35: Bellman Expectation Equation -- Derivation

**Time**: ~4 min (THEORY)
**Talking points**:

- Three derivation steps: definition, split first reward from rest, recognise recursion (Markov property).
- If beginners look confused: "The value of being somewhere equals the immediate reward plus the value of where you end up."
- If experts look bored: "The Markov property is what makes this a tractable recursion."

---

## Slide 36: Bellman Expectation -- Result

**Time**: ~3 min (THEORY)
**Talking points**:

- The result box. Read it aloud: "For each action, for each next state, add reward plus discounted future value."
- "This is a system of linear equations -- one per state."
- If beginners look confused: "This equation says: how good a state is depends on what you can get now plus what you expect to get later."

---

## Slide 37: Bellman Optimality Equation -- Derivation

**Time**: ~3 min (THEORY)
**Talking points**:

- Replace sum over actions with max over actions. The optimal policy is deterministic.
- If beginners look confused: "The best strategy always picks the best action, not a random one."
- If experts look bored: "Deterministic optimal policy in finite MDPs is a classical result."

---

## Slide 38: Bellman for Q-Values

**Time**: ~2 min (THEORY)
**Talking points**:

- Two columns: expectation and optimality for Q-values.
- Relationship: V* = max_a Q*(s,a). Policy* = argmax_a Q*(s,a).
- If beginners look confused: "Q-values answer: how good is it to DO this specific action in this specific state?"
- SKIPPABLE if running short -- the V-function equations are sufficient.

---

## Slide 39: Policy Gradient

**Time**: ~3 min (THEORY)
**Talking points**:

- The policy gradient theorem. Advantage function A(s,a).
- Intuition: "Increase probability of better-than-average actions, decrease worse-than-average."
- If beginners look confused: "Instead of learning values, directly learn which actions to take."
- If experts look bored: "The REINFORCE estimator has high variance -- that is what PPO addresses."
  **Transition**: "PPO makes policy gradient stable..."

---

## Slide 40: PPO -- Clipped Objective Step 1

**Time**: ~3 min (THEORY)
**Talking points**:

- Probability ratio r_t(theta) between new and old policy.
- Problem: naive objective allows unbounded policy updates.
- "TRPO solved this with constrained optimisation (expensive). PPO finds a simpler way."
- If beginners look confused: "We need to prevent the model from changing too much in one step."

---

## Slide 41: PPO -- Clipped Objective Step 2

**Time**: ~4 min (THEORY)
**Talking points**:

- The clipped objective in the result box. This is the key PPO equation.
- How clipping works: good actions capped at 1+epsilon, bad actions floored at 1-epsilon.
- Typical epsilon = 0.2 -- policy changes by at most 20% per step.
- If beginners look confused: "The model can only change a little bit at a time. This keeps training stable."
- If experts look bored: "The min operation is elegant -- it chooses the more conservative of the two objectives."
- PAUSE for questions. PPO is widely used and important to understand.

---

## Slide 42: GAE -- Generalised Advantage Estimation

**Time**: ~2 min (THEORY)
**Talking points**:

- GAE balances bias and variance. Lambda=0: low variance, high bias. Lambda=1: high variance, low bias. Lambda=0.95: practical sweet spot.
- If beginners look confused: "GAE decides how much to rely on predictions vs actual outcomes."
- SKIPPABLE if running short.

---

## Slide 43: SAC -- Maximum Entropy RL

**Time**: ~2 min (THEORY)
**Talking points**:

- Entropy bonus encourages exploration. Temperature alpha controls exploration vs exploitation.
- Alpha can be learned automatically.
- If beginners look confused: "SAC encourages the agent to try different things instead of always doing the same thing."
- SKIPPABLE if running short.

---

## Slide 44: RLHF as RL Application

**Time**: ~2 min (ADVANCED)
**Talking points**:

- Table mapping RL concepts to RLHF: environment=prompt distribution, action=next token, reward=reward model score.
- "DPO bypasses this. But understanding the RL formulation explains WHY DPO works."
- If beginners look confused: "This connects the RL lesson back to the alignment lesson."
- SKIPPABLE if running short.

---

## Slide 45: Offline RL

**Time**: ~2 min (ADVANCED)
**Talking points**:

- CQL and Decision Transformer. Learning without live environment interaction.
- Use case: healthcare, finance -- anywhere you cannot experiment.
- SKIPPABLE if running short.

---

## Slide 46: Kailash Bridge -- RLTrainer

**Time**: ~3 min
**Talking points**:

- Show the code: RLTrainer with algorithm="ppo", env_name, clip_range, gae_lambda.
- SWITCH TO LIVE CODING if time allows.
  **Transition**: "Can we combine adapters without retraining?"

---

## Slide 47: 6.4 -- Advanced Alignment: Model Merging

**Time**: ~2 min (ADVANCED)
**Talking points**:

- Concept: train separate LoRA adapters for different skills, merge into one model.
- Timeline: Adapter A (code) + Adapter B (chat) merged.
- If beginners look confused: "Think of combining two skill sets into one model without starting over."

---

## Slide 48: Merging Methods

**Time**: ~2 min (ADVANCED)
**Talking points**:

- Table: Model Soups, TIES, DARE, SLERP.
- "Training one adapter per task is cheaper than training one model for all tasks."
- SKIPPABLE if running short.

---

## Slide 49: TIES Merging Step by Step

**Time**: ~2 min (ADVANCED)
**Talking points**:

- Three steps: Trim, Elect sign, Merge. Conflicting updates cancel, agreeing updates reinforce.
- SKIPPABLE if running short.

---

## Slide 50: Differential Privacy in Alignment

**Time**: ~2 min (ADVANCED)
**Talking points**:

- DP-SGD: clip gradients, add calibrated noise.
- LoRA + DP-SGD: fewer parameters, less noise needed, better privacy-utility tradeoff.
- SKIPPABLE if running short.

---

## Slide 51: Kailash Bridge -- Merging & Evaluation

**Time**: ~2 min
**Talking points**:

- Show the code: merge with method="ties", compare with evaluator.
- SKIPPABLE if running short (covered conceptually already).
  **Transition**: "Now the big shift: from teaching the model to controlling the organisation..."

---

## Slide 52: 6.5 -- AI Governance with PACT

**Time**: ~3 min
**Talking points**:

- "Governance = rules about who can do what, enforced automatically."
- Hotel key card analogy: "The card encodes your access rights, and the lock enforces them."
- "Without governance, every AI system is a skeleton key."
- If beginners look confused: "Governance is like security badges at an office. You can only go where you are allowed."
- If experts look bored: "The distinction between authentication, authorisation, and auditing is fundamental here."
  **Transition**: "Let us start with the regulatory landscape..."

---

## Slide 53: EU AI Act -- Risk Tiers

**Time**: ~3 min (THEORY)
**Talking points**:

- Walk through the four tiers: Unacceptable (banned), High (conformity assessment), Limited (transparency), Minimal (no requirements).
- Use the coloured badges -- they are visual anchors.
- If beginners look confused: "The EU sorts AI systems by danger level and regulates accordingly."
- If experts look bored: "The classification in Annex III is where the real implementation complexity lies."

---

## Slide 54: Key EU AI Act Articles

**Time**: ~2 min (THEORY)
**Talking points**:

- Articles 6, 9, 13, 52. Focus on Art. 9 (risk management) and Art. 13 (transparency).
- Fines: 7% of global annual turnover for prohibited practices.
- If beginners look confused: "These are the specific rules companies must follow."
- SKIPPABLE if running short -- merge with previous slide.

---

## Slide 55: Singapore -- AI Verify & ISAGO 2.0

**Time**: ~2 min (THEORY)
**Talking points**:

- Singapore context: voluntary but market-expected. MAS already requires AI governance for financial institutions.
- If the class includes Singaporean professionals: "This is directly relevant to your industry."
- SKIPPABLE if audience is not Singapore-focused.
  **Transition**: "Now let us see how PACT implements governance..."

---

## Slide 56: PACT -- The Governance Grammar

**Time**: ~4 min
**Talking points**:

- D/T/R: Department, Team, Role. Three letters that define everything.
- "Every entity in the system -- human or AI -- has a D/T/R address."
- If beginners look confused: "Think of it as an org chart for both people and AI agents."
- If experts look bored: "The formal grammar ensures machine-verifiable governance, not just policy documents."
- PAUSE for questions. D/T/R is the conceptual foundation for everything that follows.

---

## Slide 57: D/T/R Addressing

**Time**: ~3 min (THEORY)
**Talking points**:

- Walk through the org tree: departments, teams, roles (human and AI).
- Address format: org/engineering/ml-platform/senior-ml-engineer.
- "Notice that ml-agent-alpha has the same addressing as a human role."
- If beginners look confused: "Every person and every AI gets an address, like an email address."

---

## Slide 58: Monotonic Tightening

**Time**: ~3 min (THEORY)
**Talking points**:

- Formal: envelope(R) is a subset of envelope(T) is a subset of envelope(D).
- "Once restricted, cannot be un-restricted by a child."
- The fail-closed callout: "If any level denies access, access is denied. No override. No exceptions."
- If beginners look confused: "Permissions can only get stricter as you go deeper in the org chart. A team member can never have more access than the team."
- If experts look bored: "Monotonic tightening is what makes the system formally verifiable."

---

## Slide 59: Operating Envelopes

**Time**: ~3 min (THEORY)
**Talking points**:

- Effective envelope = role envelope intersect task envelope.
- Show the two-column example: role allows $5 and DataExplorer+polars. Task requires $2 and DataExplorer.
- Effective: max cost = min(5, 2) = $2. Tools = intersection.
- If beginners look confused: "You CAN do X and you are ASKED to do Y. What actually happens is the overlap."
- If experts look bored: "The intersection semantics make policy composition straightforward."

---

## Slide 60: Knowledge Clearance

**Time**: ~3 min (THEORY)
**Talking points**:

- Walk through the 5 levels: Public, Internal, Confidential, Restricted, Critical.
- "An agent with Level 3 clearance cannot access Level 4 data, even if its role technically allows it."
- If beginners look confused: "This is like security clearance in government. Different levels of secrecy."
- If experts look bored: "Knowledge clearance is orthogonal to authority -- you can have high authority but low clearance."

---

## Slide 61: Enforcement -- TrustPosture & ConfidentialityLevel

**Time**: ~2 min (THEORY)
**Talking points**:

- TrustPosture: Untrusted, Cautious, Standard, Elevated.
- ConfidentialityLevel: maps to clearance 1-5.
- "TrustPosture determines what you can do. ConfidentialityLevel determines what you can see."
- SKIPPABLE if running short.

---

## Slide 62: Kailash Bridge -- GovernanceEngine

**Time**: ~3 min
**Talking points**:

- Show the code: GovernanceEngine, compile_org, Address, can_access, explain_access.
- SWITCH TO LIVE CODING if time allows.
- If beginners look confused: "All that governance theory? Kailash implements it in code."
  **Transition**: "Now let us wrap a real agent with governance..."

---

## Slide 63: 6.6 -- Governed Agents

**Time**: ~3 min
**Talking points**:

- Two columns: Ungoverned (can call any API, no limit, all data, no audit) vs PACT-Governed (restricted tools, enforced budget, filtered data, logged).
- "An agent without governance is a liability. A governed agent is an asset."
- If beginners look confused: "This is the difference between a free agent and a controlled employee."

---

## Slide 64: GovernanceContext -- Frozen by Design

**Time**: ~3 min (THEORY)
**Talking points**:

- Computed at task start, immutable, checked on every tool call, fail-closed.
- "Why frozen? If an agent could modify its own context, prompt injection could escalate privileges."
- If beginners look confused: "The agent gets a permission slip at the start. It cannot change its own permission slip."
- If experts look bored: "This is the key defense against privilege escalation via prompt injection."

---

## Slide 65: RoleEnvelope & TaskEnvelope

**Time**: ~2 min (THEORY)
**Talking points**:

- Show both JSON examples side by side.
- "Effective: max_cost = min(50, 10) = $10. Tools = intersection."
- If beginners look confused: "Two sets of rules. The stricter one wins."

---

## Slide 66: Audit Chains

**Time**: ~3 min (THEORY)
**Talking points**:

- Show the audit record JSON: timestamp, agent, action, decision, context, cost, budget remaining.
- "When the regulator asks 'why did it decide this?', you hand them the audit chain."
- If beginners look confused: "Every action the agent takes is recorded with full details."
- If experts look bored: "The audit chain provides the evidence base for EU AI Act Art. 13 transparency requirements."
- PAUSE for questions. Audit chains are often the most eye-opening concept for business-focused students.

---

## Slide 67: Kailash Bridge -- PactGovernedAgent

**Time**: ~3 min
**Talking points**:

- Show the code: RoleEnvelope, TaskEnvelope, GovernanceContext, PactGovernedAgent.
- "The governance context is frozen at creation."
- SWITCH TO LIVE CODING if time allows.
  **Transition**: "One governed agent is easy. What about fifty?"

---

## Slide 68: 6.7 -- Agent Governance at Scale

**Time**: ~2 min
**Talking points**:

- Three challenges: budget cascading, clearance propagation, failure handling.
- If beginners look confused: "When you have many AI agents, you need to coordinate their budgets and permissions."

---

## Slide 69: Budget Cascading

**Time**: ~3 min (THEORY)
**Talking points**:

- Sum of individual budgets at most equals team budget, which is bounded by department budget.
- Pre-committed allocation. Over-budget triggers hard stop.
- If beginners look confused: "The team has a total budget. Each agent gets a piece. When the piece runs out, the agent stops."

---

## Slide 70: Clearance Propagation

**Time**: ~3 min (THEORY)
**Talking points**:

- When A delegates to B: B's effective clearance = min(B's role clearance, A's task clearance).
- Anti-pattern: Agent A (clearance 2) delegates to Agent B (clearance 4) to access restricted data. PACT prevents this.
- If beginners look confused: "You cannot get around your security clearance by asking someone with higher clearance to do it for you."
- If experts look bored: "The monotonic tightening on delegation is the key security property."

---

## Slide 71: Dereliction -- When Agents Fail

**Time**: ~3 min (THEORY)
**Talking points**:

- Table: budget exceeded (hard stop), clearance violation (immediate denial), timeout (graceful shutdown), tool misuse (call blocked), cascade failure (supervisor decides).
- If beginners look confused: "Every type of failure has a specific, predictable response."
- If experts look bored: "The cascade failure handling -- retry, reassign, or escalate -- mirrors real incident management."

---

## Slide 72: Algorithmic Impact Assessment

**Time**: ~2 min (ADVANCED)
**Talking points**:

- Five-step AIA: Scope, Stakeholders, Risk analysis, Mitigation, Monitoring.
- EU AI Act Art. 9 requires this for high-risk systems.
- SKIPPABLE if running short.
  **Transition**: "Let us see the Kailash implementation..."

---

## Slide 73: Kailash Bridge -- Governance at Scale

**Time**: ~2 min
**Talking points**:

- Show the code: BudgetCascade and GovernanceTestHarness.
- "The test harness validates clearance propagation, budget enforcement, and dereliction handling."
  **Transition**: "The capstone puts everything together..."

---

## Slide 74: 6.8 -- Capstone: Full Platform

**Time**: ~3 min
**Talking points**:

- Timeline: Core SDK, DataFlow, ML, Kaizen, PACT, Nexus, Align -- all 8 packages.
- "Build a governed ML pipeline where agents explore, train, evaluate, and deploy -- all within PACT envelopes."
- Scaffolding: ~40%.
- If beginners look confused: "This is everything you learned in the entire programme, in one exercise."

---

## Slide 75: Kailash Engine Map -- Module 6

**Time**: ~2 min
**Talking points**:

- Table of all engines with packages and purposes.
  **Transition**: "Here is the theory-to-engine mapping..."

---

## Slide 76: Theory-to-Engine Mapping

**Time**: ~2 min
**Talking points**:

- Walk through the table: LoRA maps to AlignmentConfig, DPO to AlignmentConfig(method="dpo"), PPO to RLTrainer, D/T/R to Address.
- This is reference material for the labs.

---

## Slide 77: Architecture -- The Trust Stack

**Time**: ~3 min (THEORY)
**Talking points**:

- Walk through the stack from top to bottom: Application (Nexus), Agent (Kaizen + PACT), ML (Training + Alignment + RL), Data (DataFlow + FeatureStore), Trust Plane (PACT GovernanceEngine).
- "The trust plane is UNDERNEATH everything. Every layer checks it before acting."
- If beginners look confused: "Think of it as a building with the security system in the foundation."
- If experts look bored: "This architecture makes governance a cross-cutting concern rather than an afterthought."

---

## Slide 78: Lab Setup

**Time**: ~2 min
**Talking points**:

- Walk through the 6 exercises: SFT, DPO, Governance, Governed Agents, RL, Capstone.
- "Open modules/ascent06/local/ex_1.py."

---

## Slide 79: Environment Setup

**Time**: ~2 min
**Talking points**:

- Show the setup code: dotenv, ASCENTDataLoader, pip install commands.
- GPU note: SFT/DPO work with QLoRA on single GPU or Colab T4. RL on CPU. Governance needs no GPU.

---

## Slide 80: Data Loading

**Time**: ~1 min
**Talking points**:

- Show loader patterns: preference pairs, RL env config, governance org definition.
- "Never hardcode paths. Never use pandas."
  **Transition**: "Before the lab, let us discuss some scenarios..."

---

## Slide 81: Discussion -- Alignment Choices

**Time**: ~5 min
**Talking points**:

- Three scenarios: SFT vs DPO data prep, "too formal" fine-tuned model, failed merge.
- PAUSE for discussion. Let students debate.
- If beginners look confused: Start with scenario 1 (most concrete). "1,000 transcripts rated 1-5 stars. For SFT, you need instruction-response pairs. For DPO, you need preferred-rejected pairs."
- If experts look bored: Focus on scenario 3 (merge debugging). "What does TIES do differently from Model Soups when parameters conflict?"

---

## Slide 82: Discussion -- Governance Scenarios

**Time**: ~5 min
**Talking points**:

- Three scenarios: 2am retrain decision, junior analyst's clearance question, board audit question.
- PAUSE for discussion. Scenario 3 (board question) is excellent for practising non-technical communication.
- If beginners look confused: Start with scenario 2 (most relatable). "Clearance is about data sensitivity, not intelligence."
- If experts look bored: Focus on scenario 1 (operational). "Walk through the exact PACT decision chain at 2am."

---

## Slide 83: Discussion -- RL Applications

**Time**: ~4 min
**Talking points**:

- Three scenarios: offline RL for portfolio, PPO reward spike, DPO vs PPO choice.
- SKIPPABLE if running short.
  **Transition**: "Let us wrap up with the key takeaways..."

---

## Slide 84: Key Takeaways -- Everyone

**Time**: ~2 min
**Talking points**:

- Five bullets: fine-tuning, alignment, governance, RL, Kailash engines.
- Reinforce: "Every concept maps to a production-ready Kailash engine."

---

## Slide 85: Key Takeaways -- If You Followed the Math

**Time**: ~2 min (THEORY)
**Talking points**:

- LoRA low-rank, DPO derivation, Bellman equations, PPO clipping, PACT envelopes.
- SKIPPABLE if running short.

---

## Slide 86: Key Takeaways -- Experts

**Time**: ~2 min (ADVANCED)
**Talking points**:

- ORPO/SimPO/KTO, TIES/DARE, offline RL, DP-SGD, algorithmic impact assessments.
- SKIPPABLE if running short.

---

## Slide 87: Cumulative Kailash Engine Map

**Time**: ~2 min
**Talking points**:

- Full M1-M6 map. M6 row highlighted.
- "You now have the COMPLETE Kailash toolkit."
- Let this sink in. This is a milestone moment for the programme.

---

## Slide 88: What Comes Next

**Time**: ~2 min
**Talking points**:

- Left column: Capstone (all 8 packages, ~40% scaffolding).
- Right column: After ASCENT (terrene.dev docs, open-source contributions, Advanced Certificate).
- "Advanced Certificate awarded (M5+M6)."

---

## Slide 89: Assessment Preview

**Time**: ~3 min
**Talking points**:

- Quiz topics: LoRA rank, DPO beta, D/T/R addressing, PPO clipping, EU AI Act tiers.
- Capstone assessment: deploy a governed agent pipeline. Assessed on correct integration, envelope enforcement, audit completeness.
- If beginners look confused: "The capstone is about connecting everything together, not building from scratch."

---

## Slide 90: Closing Slide

**Time**: ~2 min
**Talking points**:

- Read the provocation: "Build AI that does what you want, controlled by who you trust."
- Thank the class. Acknowledge the journey from zero Python (M1) to governed AI agents (M6).
- "Open your laptops. Start with Exercise 6.1."

---

## Time Budget Summary

| Section                          | Slides | Time         |
| -------------------------------- | ------ | ------------ |
| Title + Recap                    | 1-4    | ~9 min       |
| Opening Case (EU AI Act)         | 5-7    | ~10 min      |
| 6.1 SFT / LoRA                   | 8-17   | ~28 min      |
| 6.2 DPO / Preference Alignment   | 18-32  | ~32 min      |
| Break                            | --     | ~10 min      |
| 6.3 Reinforcement Learning       | 33-46  | ~35 min      |
| 6.4 Advanced Alignment (Merging) | 47-51  | ~10 min      |
| 6.5 PACT Governance              | 52-62  | ~30 min      |
| 6.6 Governed Agents              | 63-67  | ~14 min      |
| 6.7 Governance at Scale          | 68-73  | ~15 min      |
| 6.8 Capstone + Engine Maps       | 74-77  | ~10 min      |
| Lab Setup                        | 78-80  | ~5 min       |
| Discussion                       | 81-83  | ~14 min      |
| Synthesis + Assessment + Closing | 84-90  | ~15 min      |
| **Total**                        |        | **~237 min** |

**Note**: This module is the densest in the programme. To fit 180 minutes:

- Skip ALL Advanced slides (28, 29, 31, 44, 45, 47, 48, 49, 50, 51, 72, 83, 85, 86): saves ~28 min
- Compress the DPO derivation: show Steps 1, 5, 6 only (Bradley-Terry, cancellation, final loss), narrate the middle steps: saves ~8 min
- Compress RL: show Bellman result + PPO result only, skip derivation steps and GAE/SAC: saves ~10 min
- Shorten discussion to one alignment + one governance scenario: saves ~5 min
- Total savings: ~51 min, bringing it to ~186 min. Close enough for a 3-hour session.

**Mark as skippable**: Slides 28 (GRPO), 29 (Beyond DPO), 31 (LLM-as-Judge Biases), 38 (Bellman Q-values), 42 (GAE), 43 (SAC), 44 (RLHF as RL), 45 (Offline RL), 47-51 (all of 6.4 Merging), 54 (EU Articles detail), 55 (Singapore AI Verify), 61 (TrustPosture), 72 (AIA), 83 (RL Discussion), 85 (Math Summary), 86 (Expert Summary).

**Best break point**: After 6.2 DPO derivation (~79 minutes in). This is the natural boundary between alignment and RL/governance.

**Live coding opportunities**: Slide 17 (AlignmentPipeline), Slide 32 (DPO training), Slide 46 (RLTrainer), Slide 62 (GovernanceEngine), Slide 67 (PactGovernedAgent).
