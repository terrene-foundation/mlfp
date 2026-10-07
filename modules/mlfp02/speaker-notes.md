# Module 2: Statistical Mastery for ML and AI Success — Speaker Notes

Total time: ~195 minutes core (slides 1-69 and 96-100) + ~55 minutes optional supplementary material (slides 70-95: formula references and deeper content per lesson).

These notes follow the current deck (`deck.html`, 100 slides) one section per slide, in deck order. The core teaching arc is slides 1-69 (the eight lessons) plus the closing block (96-100). Slides 70-76 are formula-reference cards (photograph/bookmark material, ~1 min each). Slides 77-95 are optional depth per lesson — use them when the class is ahead of schedule or a topic needs a second pass.

---

## Slide 1: Statistical Mastery for Machine Learning and AI Success

**Time**: ~2 min
**Talking points**:

- Welcome the class back. Bridge from M1: "In Module 1 you explored data and computed summary statistics. Now we ask the harder question: how confident are we in those numbers? Statistics gives us the language of uncertainty."
- Read the provocation aloud: "How confident are you in that number?"
- Ask the class: "Has anyone made a business decision based on an average, only to discover the average hid something important?"
- **If beginners confused**: "This module gives you the tools to know when a number is trustworthy."
- **If experts bored**: "We go deep into MLE, bootstrapping, CUPED, and causal inference. Formula-heavy module."

**Transition**: "Here is what you will be able to do by the end."

---

## Slide 2: What You Will Learn

**Time**: ~2 min
**Talking points**:

- Walk the left column: probability and Bayes, MLE and bootstrap estimation, A/B test design and analysis, linear and logistic regression, ANOVA, CUPED, feature engineering and the feature store.
- Walk the three depth layers. Reassure foundations-level students that every formula is explained in plain language first. "M2 is formula-heavy by design. The green slides always explain the intuition before the blue slides show the math."
- Tell advanced students: "This module has 15+ key formulas. The purple slides go beyond what is testable."
- **If beginners confused**: "Green slides are yours. Blue is stretch. Purple is bonus."
- **If experts bored**: "Purple slides cover conjugate priors, BCa intervals, and the CUPED derivation."

**Transition**: "Here is the eight-lesson arc."

---

## Slide 3: Your Journey: 8 Lessons

**Time**: ~1 min
**Talking points**:

- Quick overview of the arc: 2.1-2.3 build the statistical foundations, 2.4 applies them to experiments, 2.5-2.6 introduce predictive models, 2.7 is advanced experiment analysis, and 2.8 integrates everything in a capstone.
- Point out the "You will be able to..." column — each lesson ends with a concrete capability.
- **If beginners confused**: "Think of it as learning the rules of evidence before building your first predictive model."
- **If experts bored**: "CUPED in 2.7 and causal inference are industry-grade techniques."

**Transition**: "And each lesson pairs with a Kailash engine."

---

## Slide 4: Kailash Engines You Will Meet

**Time**: ~2 min
**Talking points**:

- ExperimentTracker: your lab notebook — logs each run's parameters, metrics and decision. You compute; it records.
- FeatureEngineer: builds interaction, polynomial, binned and temporal candidates, then selects the best. FeatureStore: stores typed features and returns them as of a point in time, with a lineage hash.
- TrainingPipeline (M3) and ModelVisualizer (plots). Emphasise the pattern: in M2 the statistics are computed by hand (numpy, scipy, polars); the engines record runs, build and store features, and plot results. None of them runs a t-test or CUPED for you.
- **If beginners confused**: "The engines are your power tools. We teach you the hand tools first."
- **If experts bored**: "FeatureStore implements point-in-time retrieval, which prevents a very common data leakage bug."

**Transition**: "Lesson 2.1 — probability."

---

## Slide 5: Lesson 2.1: Probability and Bayesian Thinking

**Time**: ~1 min
**Talking points**:

- Read the objectives aloud: truth tables, Bayes' theorem on real scenarios, sampling bias, choosing distributions.
- Bridge from M1: "You know how to load and explore data. Every number you computed in M1 came from a sample. How much should you trust it?"
- **If beginners confused**: "Probability is just measuring how likely something is. We use it every day without realising."
- **If experts bored**: "We cover conjugate priors and the friendship paradox."

**Transition**: "Let me start with a question that fools most people."

---

## Slide 6: COVID ART Test: Can You Trust the Result?

**Time**: ~3 min
**Talking points**:

- Set up the case: positive ART test, 95% sensitivity, 97% specificity, 1% prevalence in Singapore. Key question: "What is the probability you actually have COVID?"
- Let the class guess before revealing. Most say ~95%. The math says ~24%.
- Walk the frequency version: in 10,000 people, 100 infected, 9,900 not. 95 of the 100 test positive. 297 of the 9,900 also test positive (3% false-positive rate). So 95 / (95 + 297) = 24.2%.
- Land the ML connection: "Every ML model is a test. False positive rates and base rates determine whether you can trust its output."
- **If beginners confused**: "When the disease is rare, even a very accurate test produces many false alarms."
- **If experts bored**: "This is the base rate fallacy. It applies to every ML classifier — it is why accuracy misleads on imbalanced data."

**Transition**: "Let's build the probability machinery that explains this."

---

## Slide 7: Probability Fundamentals

**Time**: ~3 min
**Talking points**:

- Core rules: P(A) between 0 and 1; P(A) + P(A') = 1. Independent events (coin flips) vs dependent events (cards without replacement). Decision steps: does order matter? Does one event affect the other?
- Walk the rain-and-umbrella truth table: joint probabilities fill the interior cells and sum to 1; marginals are the edges.
- Ask: "What is P(Umbrella | Rain)?" Answer: 0.27 / 0.30 = 0.90. And P(Rain | Umbrella) = 0.27 / 0.40 = 0.675 — condition on a different margin and the answer changes.
- **If beginners confused**: "A truth table is just a grid that shows how often two things happen together."
- **If experts bored**: "This is a contingency table. The chi-squared test in Lesson 2.4 tests whether the cells differ from expected under independence."

**Transition**: "The vertical bar is the key notation — conditional probability."

---

## Slide 8: Conditional Probability

**Time**: ~3 min
**Talking points**:

- P(A|B): "Given that B happened, how likely is A?" The vertical bar shrinks the universe to the world where B happened.
- Product rule: P(A, B) = P(A) × P(B|A).
- Supermarket example: 60% buy vegetables; of those, 30% buy olive oil. P(Veg and Oil) = 0.60 × 0.30 = 0.18.
- **If beginners confused**: "Imagine standing at the vegetable aisle. You only count the people who pass you. Of those, how many also buy olive oil?"
- **If experts bored**: "This generalises to the chain rule: P(A,B,C) = P(A) P(B|A) P(C|A,B)."

**Transition**: "Rearrange the product rule and you get the most important formula of the lesson."

---

## Slide 9: Bayes' Theorem

**Time**: ~4 min
**Talking points**:

- Write it on the board: P(A|B) = P(B|A) P(A) / P(B). Name the four pieces: posterior, likelihood, prior, evidence.
- Denominator by total probability: P(B) = P(B|A)P(A) + P(B|A')P(A') — sum over all the ways B could happen. It is a normalising constant.
- Apply to the COVID case: P(COVID|Pos) = (0.95 × 0.01) / (0.95 × 0.01 + 0.03 × 0.99) = 0.0095 / 0.0392 = 0.242.
- **If beginners confused**: "Bayes' theorem is a formula for updating your belief when you get new information."
- **If experts bored**: "The Bayesian interpretation treats probability as degree of belief, not frequency. We build on this in the conjugate priors slide."

**Transition**: "The frequency table makes the same calculation visible."

---

## Slide 10: Bayes in Action: COVID ART Test

**Time**: ~3 min
**Talking points**:

- Walk the per-10,000 table row by row: 100 infected → 95 test positive; 9,900 not infected → 297 false positives. Of 392 positive tests, only 95 are true.
- Key question: "If prevalence rises to 10%, what happens to the posterior?" Answer: it jumps to about 78%. The prior matters enormously.
- Connect to ML: "297 false positives swamp 95 true positives because the disease is rare. This is why accuracy is misleading on imbalanced datasets."
- **If beginners confused**: "Focus on the table. Out of 392 positive tests, only 95 actually have COVID."
- **If experts bored**: "Sensitivity and specificity are properties of the test. Positive predictive value depends on prevalence. This distinction trips up many practitioners."

**Transition**: "Two numbers summarise any distribution — expected value first."

---

## Slide 11: Expected Value and Key Distributions

**Time**: ~3 min
**Talking points**:

- E[X] = Σ pᵢ xᵢ — a probability-weighted average. Lottery example: $2 ticket, win $100 with probability 0.01. E[X] = $1.00 — less than the cost. Bad bet.
- Tour the theme park table: Normal (continuous, symmetric — HDB prices), Poisson (rare counts — MRT breakdowns), Exponential (time between events), Beta (probabilities — conversion rates), Uniform (equal likelihood).
- Do not go deep into each distribution; they recur throughout the module.
- **If beginners confused**: "Expected value is what you would get on average if you repeated the experiment thousands of times. It is the fair price of a bet."
- **If experts bored**: "Beta is the conjugate prior for binomial. We use it in the next slide."

**Transition**: "What happens to Bayes when the prior and the likelihood belong to the same family?"

---

## Slide 12: Conjugate Priors: Elegant Bayesian Updates

**Time**: ~3 min
**Talking points**:

- A prior is conjugate to a likelihood if the posterior has the same family as the prior. Beta-Binomial: prior Beta(α, β), observe k successes in n trials, posterior Beta(α + k, β + n − k). Normal-Normal: posterior stays Normal.
- HDB example: prior mean price μ ~ N($500K, $25K²), observe 50 transactions; the posterior shifts toward the observed mean, weighted by the strength of the data vs the strength of the prior.
- Conjugate priors give closed-form posteriors — no simulation. Non-conjugate priors need MCMC.
- **If beginners confused**: "Conjugate just means the math stays clean. The posterior is the same type of distribution as the prior, just with updated numbers."
- **If experts bored**: "Discuss when conjugate priors break down and you need variational inference or HMC."

**Transition**: "One more foundational idea — a bias hiding in how data is collected."

---

## Slide 13: Sampling Bias: The Friendship Paradox

**Time**: ~2 min
**Talking points**:

- On average, your friends have more friends than you do — not pessimism, mathematics. People with many friends appear in more friend lists, so they are oversampled.
- Same bias everywhere: popular restaurants get more reviews, viral posts get more impressions.
- ML implication: "If you sample users by their activity, you oversample power users. Your model learns from the most active 5%, not the typical user."
- Fix: always ask "How was this data collected? Who is overrepresented? Who is missing?"
- **If beginners confused**: "It is like surveying people at a party. The person who knows everyone shows up in every survey. They are counted many times."
- **If experts bored**: "This is size-biased sampling. It connects to importance sampling in reinforcement learning."

**Transition**: "Here is how the engine records this kind of Bayesian work."

---

## Slide 14: Kailash Bridge: Probability in Practice

**Time**: ~2 min
**Talking points**:

- Division of labour: you choose the prior, compute the Normal-Normal posterior and its credible interval in numpy. ExperimentTracker logs the prior, posterior and interval so the run is reproducible. It records; it does not compute posteriors.
- Walk the code: every tracker call is async — `await ExperimentTracker.create(...)`, open a run with `async with tracker.track(...)`, `await` each log call, then `await tracker.close()`.
- Emphasise: the engine does not replace understanding. You must choose the prior and interpret the posterior.
- Exercise 1 preview: Bayes' theorem on test results, Normal-Normal and Beta-Binomial updates on HDB prices with prior sensitivity, credible vs confidence intervals.
- **If beginners confused**: "The engine is your lab notebook. It records what you tried and what happened."
- **If experts bored**: "Because each run stores its parameters, you can compare prior choices side by side: a cheap sensitivity analysis."

**Transition**: "Lesson 2.2 — from beliefs to estimates."

---

## Slide 15: Lesson 2.2: Parameter Estimation and Inference

**Time**: ~1 min
**Talking points**:

- Read the objectives: population vs sample, confidence intervals, MLE, and when MLE fails and MAP helps.
- Bridge: "In 2.1 you learned about probability and distributions. Now: given a bag of data, how do we estimate the parameters of the distribution that generated it?"
- **If beginners confused**: "We go from 'what is the shape of the data' to 'what are the best-fit numbers for that shape'."
- **If experts bored**: "MLE, MAP, and Bessel's correction are the core techniques."

**Transition**: "A concrete estimation problem."

---

## Slide 16: How Much Is a 4-Room HDB Flat Worth?

**Time**: ~3 min
**Talking points**:

- The estimation problem: ~350,000 4-room flats (the population — you cannot appraise each); 200 recent transactions (your sample). From the sample mean and spread you estimate the population mean and spread.
- The notation table: Greek letters (μ, σ, N) for the population — fixed unknowns. Latin letters (x̄, s, n) for the sample — computed from data.
- The gap between statistic and parameter is what statistics quantifies.
- **If beginners confused**: "Imagine trying to guess the average height of everyone in Singapore by measuring 200 people at random."
- **If experts bored**: "The distinction between parameter and statistic is trivial but the consequences are not. Bessel's correction exists because of it."

**Transition**: "And that correction is the first formula."

---

## Slide 17: Population vs Sample Variance

**Time**: ~4 min
**Talking points**:

- Population variance divides by N. Sample variance divides by n−1 — Bessel's correction.
- Why: once you know the sample mean, only n−1 values are free to vary (the last is determined). Dividing by n systematically underestimates the true variance; n−1 is unbiased.
- Analogy: "If 5 people split a $100 bill evenly and you know 4 shares, you can calculate the 5th. Only 4 are free. The sample mean constrains one value."
- Bessel's correction is one of the most-asked data-science interview questions — walk it carefully.
- **If beginners confused**: "We divide by n−1 because using the sample mean instead of the true mean makes our spread estimate too small. n−1 corrects for that."
- **If experts bored**: "The proof: E[Σ(xᵢ − x̄)²] = (n−1)σ²."

**Transition**: "What if you took many samples?"

---

## Slide 18: Sampling Distributions and the Central Limit Theorem

**Time**: ~3 min
**Talking points**:

- Take 1,000 random samples of 200 HDB transactions each, compute each mean, plot them — that is the sampling distribution. Its spread tells you how much your estimate could vary.
- CLT: no matter the shape of the original data, the distribution of sample means approaches Normal as n grows. Mean of means = μ; standard error = σ/√n. Larger n → narrower distribution → more precise estimate.
- Demonstrate visually if possible: a skewed distribution, then the bell-shaped distribution of its means.
- **If beginners confused**: "If you repeat your study many times, the averages form a bell curve — even if the original data is not bell-shaped."
- **If experts bored**: "CLT requires finite variance. Heavy-tailed distributions (Cauchy) violate it. This matters for financial data."

**Transition**: "The CLT is what makes the confidence interval possible."

---

## Slide 19: Confidence Intervals: What They Really Mean

**Time**: ~3 min
**Talking points**:

- Hammer the correct interpretation: "A 95% CI means: if we repeated this experiment 100 times, about 95 of those intervals would contain the true parameter." NOT "95% chance the parameter is in this interval" — the parameter is fixed; the interval is random.
- Analogy: "The fish is in a fixed spot in the lake. Your net lands in a random spot. 95% of the time your net catches the fish. Once you have cast the net, the fish is either in it or not."
- Computing: x̄ ± z(α/2) · s/√n. Critical values: 1.645 for 90%, 1.96 for 95%, 2.576 for 99%. Wider interval = more confident, less precise.
- **If beginners confused**: "We are measuring our confidence in the method, not in any single result."
- **If experts bored**: "Bayesian credible intervals DO have the 'probability the parameter is in the interval' interpretation, because they treat the parameter as random."

**Transition**: "Where do the best-fit parameters themselves come from? Maximum likelihood."

---

## Slide 20: Maximum Likelihood Estimation (MLE)

**Time**: ~4 min
**Talking points**:

- The big idea: assume a model (e.g. Normal), then find the parameters θ that make the observed data most probable: argmax Π P(xᵢ|θ).
- Log-likelihood: products become sums — numerically stable, same maximum (log is monotonic).
- Analogy: "Imagine twisting the knobs of a Normal distribution — mean and std dev — until the curve fits your data as well as possible. MLE finds the knob positions."
- **If beginners confused**: "MLE answers: given this data, what are the best-fitting parameters?"
- **If experts bored**: "For the Normal: mean MLE = sample mean; variance MLE uses the 1/n denominator (biased — hence Bessel). Fisher information connects to the curvature."

**Transition**: "When does this break, and what is the fix?"

---

## Slide 21: When MLE Fails: Small Samples and MAP

**Time**: ~3 min
**Talking points**:

- Three failure modes: small n (MLE overfits the sample — with 5 points the estimate is unreliable); multimodal likelihood (optimiser finds the wrong peak); misspecified model (the "best wrong answer").
- MAP = MLE + log-prior: maximise likelihood plus prior. The prior pulls extreme estimates toward reasonable values.
- Connection: MAP with a Gaussian prior IS L2 regularisation (Ridge). This bridges Bayesian statistics and M3's regularised models.
- **If beginners confused**: "MAP is MLE with a safety net. When data is scarce, your prior beliefs prevent wild estimates."
- **If experts bored**: "Bernstein–von Mises: as n grows, the posterior concentrates around the MLE regardless of prior. MAP and MLE converge. L1 = Laplace prior."

**Transition**: "Let's see the fit with the engine."

---

## Slide 22: Kailash Bridge: Parameter Estimation

**Time**: ~2 min
**Talking points**:

- Walk the code: the negative log-likelihood and the BFGS optimiser are plain scipy. σ is reparameterised as exp(log σ) to keep it positive. On the 101 quarterly GDP growth values, MLE gives μ = 3.90 (the sample mean) and σ = 4.07 (the divide-by-n standard deviation) — exactly as the derivation predicts.
- ModelVisualizer's job: plot the data so you can judge whether a Normal is a sensible model at all.
- Exercise 2 preview: CLT and Bessel's correction, MLE with Fisher-information and profile-likelihood CIs (the profile re-fits σ at each μ value — invariant to reparameterisation, unlike the Wald CI), MAP shrinkage, failure modes, AIC/BIC across distribution families. Correct guidance: AIC selects for prediction; BIC for recovering the true model.
- **If beginners confused**: "The code finds the bell curve that fits best; the chart lets you check it actually looks like the data."
- **If experts bored**: "The log-sigma reparameterisation removes the positivity constraint; Exercise 2 shows the profile-likelihood CI is invariant to it while the Wald CI is not."

**Transition**: "Lesson 2.3 — what if no formula exists for your statistic?"

---

## Slide 23: Lesson 2.3: Bootstrapping and Hypothesis Testing

**Time**: ~1 min
**Talking points**:

- Read the objectives: bootstrap resampling and CIs, formulating hypotheses, p-values, multiple testing corrections.
- Bridge: "In 2.2 you computed CIs with formulas. But what if the formula does not exist for your statistic? Bootstrapping estimates anything without closed-form solutions."
- **If beginners confused**: "We learn a technique that works for any statistic, even when the math is too hard to derive."
- **If experts bored**: "BCa intervals and permutation tests are the advanced content."

**Transition**: "A real limitation to motivate it."

---

## Slide 24: When Theory Is Not Enough

**Time**: ~3 min
**Talking points**:

- The problem: you want a CI for the MEDIAN HDB resale price. The formula CI from 2.2 works for the mean; the median has no simple standard-error formula. You have one sample of 200 transactions; more data costs time and money.
- Bootstrap idea: if you cannot get more samples from the population, create new samples from the sample you have — resample with replacement thousands of times.
- Key insight: the sample stands in for the population; variation across resamples estimates the variation you would see across repeated studies.
- Ask: "How would you estimate the uncertainty of a median if you cannot derive a formula?" Let students brainstorm first.
- **If beginners confused**: "Imagine photocopying your data and shuffling the copies. Each shuffled copy gives a slightly different answer."
- **If experts bored**: "Efron's 1979 paper. Bootstrap is asymptotically consistent for most statistics."

**Transition**: "The algorithm, step by step."

---

## Slide 25: Bootstrap: The Resampling Algorithm

**Time**: ~4 min
**Talking points**:

- Steps: sample of size n → draw n observations WITH replacement → compute the statistic → repeat B = 10,000 times → the distribution of replicates is the bootstrap distribution.
- Emphasise "with replacement": some observations appear multiple times, some not at all. That is the source of variation.
- Percentile CI: sort the B replicates, take the α/2 and 1−α/2 percentiles. For 95%: the 250th and 9,750th of 10,000 sorted medians.
- No distributional assumption — works for median, IQR, correlation, ratio.
- **If beginners confused**: "You are creating fake datasets from your real dataset. Each fake dataset gives a slightly different answer. The spread of those answers IS your uncertainty."
- **If experts bored**: "Percentile is the simplest. BCa is better for skewed statistics and small samples — next slide."

**Transition**: "The percentile method has two failure modes. BCa fixes both."

---

## Slide 26: BCa Intervals: Fixing the Percentile Method

**Time**: ~3 min
**Talking points**:

- Two corrections: bias ẑ₀ (is the bootstrap distribution centred on the estimate?) and acceleration â (does the standard error change with θ? jackknife-estimated). Read the CI at adjusted percentiles.
- Walk the scipy code: `stats.bootstrap((prices,), np.median, method="BCa", ...)` — one call replaces the manual loop.
- On a 200-sale HDB sample the median is about $851K; percentile and BCa differ only slightly at the lower end (≈$799K vs ≈$797K) because a median of 200 prices is not very skewed. The gap grows with skew and smaller n. With ẑ₀ = â = 0, BCa reduces to percentile.
- Exercise 3.1 computes percentile, normal and BCa intervals side by side.
- **If beginners confused**: "BCa is the percentile interval with two automatic corrections; scipy does the arithmetic."
- **If experts bored**: "The acceleration comes from the jackknife skewness of the influence values (Efron, 1987)."

**Transition**: "From intervals to decisions — hypothesis testing."

---

## Slide 27: Hypothesis Testing: Making Decisions

**Time**: ~4 min
**Talking points**:

- Framework: H₀ "no effect", H₁ "there is an effect". p-value: probability of data at least as extreme as observed, ASSUMING H₀ is true.
- The misconception to drill: p is NOT P(H₀ is true). It is P(data this extreme | H₀) — probability of the data, not of the hypothesis.
- Ask: "If p = 0.03, what does that mean?" Correct: "If there were truly no effect, we would see data this extreme only 3% of the time."
- Decision table: α = 0.10 exploratory, 0.05 standard, 0.01 medical/regulatory, 0.0005 very strict. Language: "reject" or "fail to reject" — never "accept H₀". Absence of evidence is not evidence of absence.
- **If beginners confused**: "Think of it as a surprise score. A small p-value means the data would be very surprising if the null were true."
- **If experts bored**: "The ASA statement on p-values (2016) is required reading. P-values do not measure effect size or practical significance."

**Transition**: "The workhorse statistic, and an assumption-free alternative."

---

## Slide 28: Test Statistics and Permutation Tests

**Time**: ~4 min
**Talking points**:

- T-statistic: T = (x̄ − μ₀) / (s/√n) — a signal-to-noise ratio. How many standard errors is the sample mean from the hypothesised value? Large |T| → small p → reject.
- One-tailed vs two-tailed.
- Permutation test: combine both groups, shuffle labels, recompute the statistic thousands of times; p = fraction of shuffled statistics at least as extreme as observed.
- Advantage: no distributional assumptions, works for any statistic — the gold standard for A/B test analysis.
- **If beginners confused**: "Is the difference between my sample and the hypothesis big compared to the noise in the data?"
- **If experts bored**: "Permutation tests are exact for exchangeable data and avoid the Gaussian assumption. Fisher's original 1935 formulation."

**Transition**: "One test is fine. Twenty tests are a trap."

---

## Slide 29: Multiple Testing: The Bonferroni Trap

**Time**: ~3 min
**Talking points**:

- The problem: 20 metrics at α = 0.05 → even with NO real effect, expect 20 × 0.05 = 1 false positive. More tests → more "significant" results by chance.
- Bonferroni: α_adjusted = α/m. Conservative — controls the family-wise error rate.
- Benjamini-Hochberg (BH-FDR): less conservative; controls the expected proportion of false positives among rejections. Sort p-values, compare each to (i/m)α.
- Rule of thumb: Bonferroni for confirmatory tests (few, pre-registered); BH-FDR for exploratory screening (many metrics).
- **If beginners confused**: "Flip a coin 20 times and one heads is not surprising. Similarly, one significant p-value out of 20 tests is not surprising."
- **If experts bored**: "Holm-Bonferroni is a strictly better stepwise improvement over Bonferroni."

**Transition**: "The design question — how much data do you need?"

---

## Slide 30: Power Analysis: How Much Data Do You Need?

**Time**: ~3 min
**Talking points**:

- Four components: effect size δ (minimum difference worth detecting), sample size n, significance α (0.05), power 1−β (0.80). Fix any three, solve for the fourth.
- Minimum sample size: n = (z(α/2) + z(β))² · 2σ² / δ² per group.
- Common mistake: running an experiment without power analysis, finding "no significant effect" — you may simply be underpowered.
- **If beginners confused**: "It is like planning how long to study for an exam. If you do not study enough, you cannot pass even if you know the material."
- **If experts bored**: "MDE is the industry term. In practice: α = 0.05, power = 0.80, σ from historical data, solve for n."

**Transition**: "The engine view of bootstrap work."

---

## Slide 31: Kailash Bridge: Bootstrap and Testing

**Time**: ~2 min
**Talking points**:

- Context for the course experiment: four arms designed 40/35/15/10; we compare control with treatment_a, the pair that passes its own SRM check against the designed 40:35 ratio (p ≈ 0.62).
- Walk the code: the bootstrap loop is yours — resample each arm, difference the means, 1,000 replicates. On this data the revenue lift is about $3.16 per user with a 95% percentile CI of roughly [$2.84, $3.50].
- ModelVisualizer turns the replicates into a histogram a stakeholder can read.
- Exercise 3 preview: percentile and BCa bootstrap CIs, power and MDE, z / Welch / Mann-Whitney tests with effect sizes, Bonferroni vs BH-FDR, a permutation test.
- **If beginners confused**: "Each bar is how often a resampled experiment gave that lift. The middle 95% is the CI."
- **If experts bored**: "Exercise 3 adds BCa, which corrects the percentile interval for bias and skew."

**Transition**: "Lesson 2.4 — designing the experiment itself."

---

## Slide 32: Lesson 2.4: A/B Testing and Experiment Design

**Time**: ~1 min
**Talking points**:

- Read the objectives: design with randomisation and power analysis, data collection plan, SRM detection, pitfalls.
- Bridge: "In 2.3 you learned the statistical tools. Now we design real experiments. The difference between a good and bad A/B test is in the design, not the statistics."
- **If beginners confused**: "This lesson is about planning experiments so your statistical tests give trustworthy answers."
- **If experts bored**: "SRM detection and the data collection framework are industry-standard topics."

**Transition**: "A hawker-centre question with a hidden flaw."

---

## Slide 33: Singapore Hawker Centre: Should We Raise Prices?

**Time**: ~3 min
**Talking points**:

- The case: a stall tests a $0.50 price increase on chicken rice. Hypothesis: "the price increase does not significantly reduce daily revenue." They run the new price on weekdays, old price on weekends.
- Key question: "What is wrong with this design?" Weekday vs weekend customers are different populations — a confounded comparison, NOT a valid A/B test.
- Ask: "What other factors differ between weekdays and weekends?" (Tourist traffic, office workers, family dining.)
- Proper design: randomise by receipt number 50/50; power analysis for a 5% revenue drop; run 4 weeks across paydays and weather; SRM check that the split really is 50/50.
- **If beginners confused**: "A proper experiment randomly assigns customers to groups so the only difference is the price."
- **If experts bored**: "Discuss cluster randomisation when individual randomisation is impossible (entire tables, not individual diners)."

**Transition**: "Before collecting anything, answer four questions."

---

## Slide 34: Data Collection Framework: Why, What, Where, How

**Time**: ~3 min
**Talking points**:

- Why: hypotheses, value of the answer, performance measures. What: ideal information wish list, budget/time constraints, minimum viable dataset.
- Where: internal (CRM, POS, app analytics, finance) vs external (data.gov.sg, Kaggle, public APIs, purchased data).
- How + frequency: at start — review existing data, discover gaps, validate quality; continuous — automated real-time or batch; duration sufficient for statistical power.
- Ask: "For our hawker experiment, what data do we need?" (Transaction amounts, timestamps, price group, customer count, weather.)
- **If beginners confused**: "Four questions: Why? What specifically? Where from? How and how often?"
- **If experts bored**: "The data collection plan is a capstone deliverable. Common problems: silos, red tape, aggregated vs transaction-level data."

**Transition**: "The first check after collection — before any outcome analysis."

---

## Slide 35: Sample Ratio Mismatch (SRM)

**Time**: ~3 min
**Talking points**:

- What: you expected 50/50 but observed 52/48. Random variation or systematic breakage? SRM invalidates the entire experiment — the groups are no longer comparable.
- Common causes: bot filtering applied unevenly, redirect bugs (one variant loads faster), population filtering (login-required vs guest).
- Chi-squared test: χ² = Σ (Oᵢ − Eᵢ)²/Eᵢ. Walk the example: expected 5,000/5,000, observed 5,200/4,800 → χ² = 40,000/5,000 + 40,000/5,000 = 16, df = 1, p ≪ 0.001. SRM detected.
- SRM is the FIRST check in any analysis. If the split is off, stop — results are unreliable.
- **If beginners confused**: "You ordered 50 red balls and 50 blue balls but received 52 red and 48 blue. Factory fault or random? The chi-squared test answers this."
- **If experts bored**: "Mature experimentation platforms run an SRM check on every experiment. Kohavi et al. (2020), 'Trustworthy Online Controlled Experiments', is the reference."

**Transition**: "The full gallery of ways experiments go wrong."

---

## Slide 36: Common Experiment Pitfalls

**Time**: ~3 min
**Talking points**:

- Walk the pitfall table with a concrete example each: no randomisation (groups differ systematically), no power analysis (too small to detect real effects), peeking (early stopping inflates false positives), overlapping treatments (effects interact), temporal bias (treatment coincides with holidays or news).
- Peeking detail: "If you check your experiment every day and stop when it looks significant, your false positive rate is not 5%: with 20 daily looks it is roughly 25%. Exercise 7.3 simulates exactly this."
- Clean DataOps architecture: schema defined upfront, automated transfer pipelines, version-controlled processing, standard connectors. Data silos are the #1 barrier.
- **If beginners confused**: "These are the ways experiments can give you wrong answers even when the math is correct."
- **If experts bored**: "Sequential testing (always-valid p-values) solves the peeking problem. Overlapping experiments need interaction checks or mutually exclusive traffic layers."

**Transition**: "The engine pattern for an auditable design."

---

## Slide 37: Kailash Bridge: Experiment Design

**Time**: ~2 min
**Talking points**:

- The course experiment was designed 40/35/15/10 across four arms — SRM must be tested against THAT design, never 50/50. Walk the code: across all four arms SRM fires (variant_c is over-allocated); the control vs treatment_a pair matches its designed 40:35 ratio (p ≈ 0.62), so that pair is safe to analyse.
- Rule: a failed SRM check means "do not ship, investigate", whatever the effect estimate says.
- The tracker does not run these checks — it stores your plan, your check and your decision as one auditable run.
- Exercise 4 preview: pre-registered design and power curve, SRM (overall and per segment), Welch's t-test with CIs, validity checks, a report logged to ExperimentTracker.
- **If beginners confused**: "The engine is your experiment logbook. You do the checks; it keeps the receipt."
- **If experts bored**: "Log the pre-registration parameters before looking at outcomes, so the record proves they were not chosen after the fact."

**Transition**: "Lesson 2.5 — from testing differences to modelling relationships."

---

## Slide 38: Lesson 2.5: Linear Regression

**Time**: ~1 min
**Talking points**:

- Read the objectives: multivariate regression, t-statistics on coefficients, R² and F, dummy encoding, non-linear terms.
- Bridge: "In 2.3 you tested whether a sample statistic differs from zero. Regression coefficients ARE sample statistics — the same t-test, applied to each coefficient."
- **If beginners confused**: "Regression is the most widely used predictive model. We are building our first model."
- **If experts bored**: "We cover OLS, the normal equations, and the matrix form."

**Transition**: "The running example for the lesson."

---

## Slide 39: How Much Is Your HDB Flat Worth?

**Time**: ~3 min
**Talking points**:

- The prediction problem: predict resale price from floor area, remaining lease, storey, distance to MRT, town. Linear regression fits a hyperplane; each coefficient reads "holding everything else constant, one more unit of this feature changes the price by this much."
- The equation: price = β₀ + β₁·area + β₂·lease + β₃·storey + ε. β₀ = base price, β₁ = price per sqm, ε = what the model cannot explain.
- Frame regression as inference first: "Does floor area significantly affect price?" is a hypothesis test on β₁.
- **If beginners confused**: "We are drawing the best-fit line through a cloud of points. The slope tells you the relationship."
- **If experts bored**: "The OLS estimator is BLUE under the Gauss-Markov conditions."

**Transition**: "How OLS actually finds the line."

---

## Slide 40: Ordinary Least Squares (OLS)

**Time**: ~4 min
**Talking points**:

- Two presentations: geometric (minimise the vertical distances, squared) and algebraic (normal equations: β̂ = (XᵀX)⁻¹Xᵀy — project y onto the column space of X).
- Why squared: penalises large errors more AND gives a unique minimum (convex loss surface).
- Each piece: y actual, ŷ prediction, y − ŷ residual, X the feature matrix, β the coefficient vector.
- Note: normal equations are elegant but not always numerically stable; large datasets use gradient descent.
- **If beginners confused**: "We draw a line. The line makes errors. We make the errors as small as possible by adjusting the slope and intercept."
- **If experts bored**: "The matrix form assumes XᵀX is invertible. When it is not (multicollinearity), regularisation (Ridge/Lasso in M3) is needed."

**Transition**: "Is each coefficient real, or noise?"

---

## Slide 41: Testing Coefficients: The T-Statistic

**Time**: ~3 min
**Talking points**:

- t = β̂ / SE(β̂) — signal over noise. H₀: β = 0 (feature has no effect); H₁: β ≠ 0. Large |t| → small p → reject.
- Connect back to 2.3: the same t-statistic, now per coefficient.
- Quick reference cutoffs: |t| > 1.645 (90%, *), 1.960 (95%, **), 2.576 (99%, ***).
- HDB interpretation: "β̂_area = $5,200 with t = 12.3, p < 0.001 means each additional square metre is associated with a $5,200 increase, highly significant." (Illustrative table values.)
- **If beginners confused**: "Is this coefficient probably zero (feature does not matter) or genuinely different from zero (feature matters)?"
- **If experts bored**: "Exact cutoffs depend on degrees of freedom (n − k − 1); for large n they converge to z-values."

**Transition**: "Two numbers for the whole model — R² and F."

---

## Slide 42: Model Evaluation: R-Squared and F-Statistic

**Time**: ~4 min
**Talking points**:

- R² = 1 − SS_res/SS_tot: fraction of variance explained. 0 = nothing, 1 = perfect. Adjusted R² penalises useless predictors.
- F = (SS_reg/k) / (SS_res/(n−k−1)): tests H₀ "all coefficients are zero". T tests one coefficient; F tests the whole model.
- On the course data (sentinel prices removed, ≈49,900 sales), floor area and building age together explain about 83% of price variance (R² = 0.83); F ≈ 120,000, p < 0.001. With n this large, almost any F is significant — which is why R² and effect sizes matter more than the p-value.
- **If beginners confused**: "R-squared is the percentage of the data's variation your model captures. F tells you if the model beats just guessing the average."
- **If experts bored**: "Adjusted R² = 1 − (1−R²)(n−1)/(n−k−1) can DECREASE when you add a useless predictor. Always report adjusted R² for multivariate models."

**Transition**: "Regression needs numbers — what about towns?"

---

## Slide 43: Categorical Variables: Dummy Encoding

**Time**: ~3 min
**Talking points**:

- Town is categorical — regression needs numbers. Solution: binary 0/1 columns per category, dropping one as the base/benchmark.
- Dummy variable trap: include ALL categories and XᵀX is singular (perfect multicollinearity). Always drop one.
- Salary example: salary = β₀ + β₁·age + β₂·age² + β₃·female + β₄·trans. Base category: male. β₃ = the average salary difference for female vs male employees, holding age constant. age² captures the peak-and-decline curve.
- **If beginners confused**: "We convert categories into yes/no flags. The model learns the average difference for each category compared to the base."
- **If experts bored**: "Effect coding (sum-to-zero) is an alternative where coefficients are deviations from the grand mean."

**Transition**: "Straight lines are not the limit."

---

## Slide 44: Beyond Straight Lines: Non-Linear Terms

**Time**: ~3 min
**Talking points**:

- Key insight: "linear regression" is linear in the COEFFICIENTS, not the features. You can model curves and interactions.
- Polynomial: age² for the salary curve; area² if price per sqm changes for very large flats.
- Log-linear: ln(price) = β₀ + β₁·area → β₁ × 100% change per unit.
- Interaction: price = β₀ + β₁·area + β₂·central + β₃·(area × central) — β₃ is how much MORE each sqm is worth in central towns. "The effect of area depends on the town."
- Cross-validation preview: more terms fit training better but risk overfitting; train/test and k-fold (M3.2) detect it.
- **If beginners confused**: "An interaction term says: the effect of one feature depends on the value of another. Living in the centre makes each square metre worth more."
- **If experts bored**: "FeatureEngineer.generate creates these polynomial and interaction terms from a schema (Lesson 2.8)."

**Transition**: "The phrase that makes multivariate regression powerful."

---

## Slide 45: Multivariate Regression: Ceteris Paribus

**Time**: ~3 min
**Talking points**:

- In simple regression β₁ is the raw correlation slope; in multivariate it is the PARTIAL effect — x₁'s effect after stripping out the other predictors. This is what makes regression powerful for causal reasoning (with caveats).
- Simpson's paradox warning — present the slide's example as hypothetical: if larger flats sat in cheaper towns, simple regression could blame low prices on large area, and adding town would flip the sign. In the actual course data the raw area-price correlation is positive (≈0.47), so do not present the reversal as an HDB fact.
- Multivariate regression controls for the confounders you INCLUDE. Omitted variable bias is always a risk.
- **If beginners confused**: "Multivariate regression lets you ask: what is the effect of size ALONE, separate from location?"
- **If experts bored**: "Omitted variable bias formally: E[b] = β + (XᵀX)⁻¹XᵀZγ for omitted Z."

**Transition**: "The engine pattern for regression modelling."

---

## Slide 46: Kailash Bridge: Linear Regression

**Time**: ~2 min
**Talking points**:

- Walk the code: FeatureEngineer.generate takes a FeatureSchema and returns candidate columns (area², age², area × age); the regression itself is the normal-equations solve you derived — coefficients, SEs and t-stats are yours to compute. ModelVisualizer.residuals draws predicted-vs-actual and the residual histogram.
- The filter drops sentinel prices in the raw file (a $10 sale, a $9M sale); without it the fit is meaningless. On this data the model explains about 83% of price variance.
- Exercise 5 preview: OLS from scratch (t, R², F), diagnostics (VIF, Breusch-Pagan), weighted least squares, then polynomial, interaction and dummy terms judged on a held-out test set.
- **If beginners confused**: "The engine builds the extra columns. You fit the line and interpret it."
- **If experts bored**: "FeatureEngineer.select ranks candidates against the target — a first taste of the feature selection done properly in M3."

**Transition**: "Lesson 2.6 — from how much to yes-or-no."

---

## Slide 47: Lesson 2.6: Logistic Regression and Classification

**Time**: ~1 min
**Talking points**:

- Read the objectives: logistic regression, sigmoid and log-odds, odds ratios, one-way ANOVA and post-hoc tests, ANOVA vs t-test vs regression.
- Bridge: "Linear regression predicts a continuous number. What if the outcome is yes/no? Will the employee leave? Will the customer convert?"
- **If beginners confused**: "We are moving from predicting how much to predicting yes or no."
- **If experts bored**: "Logistic regression is the foundation for neural network activation functions. The sigmoid returns in M4."

**Transition**: "A case where linear regression breaks."

---

## Slide 48: Will This Employee Resign?

**Time**: ~3 min
**Talking points**:

- Outcome: resigned (1) or stayed (0). Linear regression can predict below 0 or above 1 — a probability of −0.3 or 1.5 makes no sense.
- Logistic regression wraps the linear equation in a function that squashes output into [0, 1] — the sigmoid.
- Common mistake: linear regression for binary outcomes — uninterpretable predictions and violated error assumptions.
- Ask: "Has anyone seen a model predict 120% probability? That is what happens with linear regression on binary outcomes."
- **If beginners confused**: "A probability must be between 0% and 100%. We need a model that respects this."
- **If experts bored**: "The linear probability model is sometimes used for interpretability, but it violates homoscedasticity and predicts outside [0,1]."

**Transition**: "The squashing function itself."

---

## Slide 49: The Sigmoid Function

**Time**: ~4 min
**Talking points**:

- σ(z) = 1/(1 + e^(−z)). Draw the curve: σ(0) = 0.5 (decision boundary); → 1 as z → +∞; → 0 as z → −∞. Smooth, differentiable, σ′ = σ(1−σ).
- Where z comes from: the same linear combination as linear regression, z = β₀ + β₁x₁ + ... Logistic = linear + sigmoid wrapper. The linear part computes a score; the sigmoid converts it to a probability.
- Deep learning preview: the sigmoid is an activation function; M4 uses sigmoid, ReLU and others.
- **If beginners confused**: "Big positive numbers become close to 1. Big negative numbers become close to 0. Zero becomes 0.5."
- **If experts bored**: "Vanishing gradients led to ReLU's dominance in deep learning, but for logistic regression the sigmoid is canonical."

**Transition**: "The scale the coefficients actually live on."

---

## Slide 50: Log-Odds and Odds Ratios

**Time**: ~4 min
**Talking points**:

- Log-odds: log(P/(1−P)) = β₀ + β₁x₁ + ... — maps (0,1) to (−∞, +∞). Odds: P/(1−P); P = 0.80 → odds 4:1.
- Odds ratio = e^β₁: multiplicative change in odds per unit increase.
- Worked example: β_overtime = 1.39 → e^1.39 = 4.01. Employees who work overtime have 4× the ODDS of resigning.
- Direction check: β > 0 higher odds, β < 0 lower odds, β = 0 no effect (OR = 1).
- Key question: "Does an odds ratio of 4 mean the probability is 4× higher?" No — odds, not probability. Baseline P = 10% (odds 0.11), 4× odds = 0.44 → P = 31%.
- **If beginners confused**: "Odds are a different way to express probability. An odds ratio tells you how much the odds change when a feature increases by 1 unit."
- **If experts bored**: "Odds ratios are not risk ratios. For rare events (P ≪ 1) they approximate relative risk; for common events they diverge."

**Transition**: "How the coefficients are fit — MLE again, not OLS."

---

## Slide 51: Fitting Logistic Regression: MLE (Not OLS)

**Time**: ~3 min
**Talking points**:

- Why not OLS: squared error is the wrong loss for 0/1 outcomes. Logistic regression uses MLE — coefficients that maximise the probability of the observed data.
- Log-likelihood: Σ [yᵢ log p̂ᵢ + (1−yᵢ) log(1−p̂ᵢ)], p̂ᵢ = σ(xᵢᵀβ). Walk the intuition: y=1 and p̂ near 1 → log p̂ near 0 (good); y=1 and p̂ near 0 → very negative (bad).
- This is cross-entropy loss — the same loss neural networks use for classification (M4).
- Connect back to 2.2's MLE machinery.
- **If beginners confused**: "We find the coefficients that make the model's predicted probabilities as close as possible to the actual yes/no outcomes."
- **If experts bored**: "No closed form — IRLS or gradient descent."

**Transition**: "Beyond two classes, and how we will judge classifiers."

---

## Slide 52: Beyond Binary: Multiclass and Evaluation Preview

**Time**: ~3 min
**Talking points**:

- Multiclass: One-vs-Rest (K binary classifiers), One-vs-One (K choose 2), multinomial (softmax — the generalised sigmoid; outputs sum to 1).
- Evaluation preview: accuracy, confusion matrix, precision, recall. Detailed metrics in M3.5.
- Plant the seed: accuracy alone misleads on imbalanced data — recall the COVID base-rate problem from 2.1.
- Do not go deep; M3.5 covers metrics fully.
- **If beginners confused**: "When the outcome has more than 2 categories (3-room, 4-room, 5-room), we extend logistic regression."
- **If experts bored**: "Softmax is the multi-class generalisation of sigmoid."

**Transition**: "Comparing three or more groups — ANOVA."

---

## Slide 53: ANOVA: Comparing Three or More Groups

**Time**: ~3 min
**Talking points**:

- T-test compares 2 groups; ANOVA generalises to 3+. H₀: all group means equal; H₁: at least one differs.
- F = MS_between / MS_within — the same signal-vs-noise intuition as the regression F.
- HDB example: prices across 3-room, 4-room, 5-room, executive. On the course data F runs into the thousands, p < 0.001 — types differ. But WHICH types? ANOVA does not say.
- Post-hoc: Tukey's HSD (all pairs, family-wise control — built on the studentized range distribution, NOT Bonferroni-corrected pairwise t-tests), Bonferroni (more conservative), Scheffe (most conservative, complex contrasts).
- **If beginners confused**: "ANOVA asks: are these groups really different, or could the differences be chance? A t-test for more than 2 groups."
- **If experts bored**: "ANOVA is a special case of linear regression with only categorical predictors. Two-way and repeated measures are in the reference material."

**Transition**: "One table to rule the toolbox."

---

## Slide 54: ANOVA vs T-Test vs Regression: When to Use What

**Time**: ~2 min
**Talking points**:

- Walk the table: t-test (2 groups, continuous outcome), ANOVA (3+ groups, continuous), linear regression (any groups, continuous outcome, mixed predictors), logistic regression (binary outcome).
- Deep connection: one-factor ANOVA = linear regression with dummy variables; same F. Regression dominates in practice because it mixes categorical and continuous predictors.
- **If beginners confused**: "Use the table as a quick reference. In practice, regression handles most cases."
- **If experts bored**: "The GLM unifies all of these. Logistic regression is a GLM with binomial family and logit link."

**Transition**: "The engine view of classification."

---

## Slide 55: Kailash Bridge: Classification and ANOVA

**Time**: ~2 min
**Talking points**:

- The lesson's case is employee attrition; the exercise applies the same model to a target the course data supports: is an HDB resale (2020 onwards) priced above the median?
- Walk the code: Bernoulli negative log-likelihood, BFGS, standardised floor area — so exp(β) is the odds ratio per one standard deviation (≈27 sqm), not per square metre. The fit is your own MLE; ModelVisualizer only draws the ROC curve and confusion matrix.
- Exercise 6 preview: logistic regression from scratch on HDB above-median price, odds ratios and cost-based thresholds, ROC/PR curves, calibration, one-way ANOVA with Tukey's HSD (studentized range).
- **If beginners confused**: "You fit the model; the engine draws the pictures that show how well it separates the classes."
- **If experts bored**: "Exercise 6 re-optimises the threshold for each cost matrix — the cost-optimal threshold depends on what a false positive and a false negative cost."

**Transition**: "Lesson 2.7 — making experiments more powerful and finding causes without randomisation."

---

## Slide 56: Lesson 2.7: CUPED and Causal Inference

**Time**: ~1 min
**Talking points**:

- Read the objectives: CUPED for variance reduction, SRM detection, Difference-in-Differences when randomisation is impossible, testing parallel trends.
- Bridge: "In 2.4 you designed A/B tests. In 2.5 you learned regression. CUPED uses regression to make experiments more powerful; DiD uses regression logic to draw causal conclusions from observational data."
- **If beginners confused**: "This lesson is about making your experiments better and finding causes even when you cannot run an experiment."
- **If experts bored**: "CUPED is one of the highest-leverage A/B test techniques: free variance reduction whenever a strong pre-period covariate exists."

**Transition**: "The pain that motivates CUPED."

---

## Slide 57: The Noisy Experiment Problem

**Time**: ~3 min
**Talking points**:

- The case: 4-week A/B test of a new checkout flow at a Singapore e-commerce site. Treatment shows 2% higher revenue per user, but p = 0.12 — not significant. Running longer is expensive.
- The problem: revenue per user is highly variable — big spenders and window shoppers in the same sample. Noise drowns the signal.
- CUPED solution: adjust the post-experiment metric using each user's PRE-experiment revenue. Noise drops, signal emerges, p = 0.03.
- Ask: "What if you could make your experiment twice as powerful without collecting more data?"
- CUPED = Controlled-experiment Using Pre-Experiment Data (Deng et al., 2013).
- **If beginners confused**: "Your experiment has static noise. CUPED removes it by looking at how each person behaved BEFORE the experiment."
- **If experts bored**: "The reduction is ρ²: ρ = 0.7 removes 49% of the variance — roughly the same as doubling the sample size."

**Transition**: "The mathematics."

---

## Slide 58: CUPED: The Mathematics

**Time**: ~4 min
**Talking points**:

- Variance reduction: Var(Y_adj) = Var(Y)(1 − ρ²), ρ = correlation between pre- and post-experiment metrics. ρ = 0.7 → remaining variance 51% (a 49% reduction); ρ = 0.5 → 75% remaining (25% reduction). Get the direction right: it is the REMAINING variance that is 1 − ρ².
- Estimator: Y_adj = Y − θ(X_pre − E[X_pre]); θ = Cov(Y, X_pre)/Var(X_pre) — the OLS slope of Y on X_pre from Lesson 2.5.
- CUPED IS regression: regress the post metric on the pre metric and use the residual. What remains is the treatment effect plus unpredictable noise.
- **If beginners confused**: "CUPED looks at how much of your outcome is explained by past behaviour, removes that, and leaves only the experiment's effect."
- **If experts bored**: "CUPED generalises to multiple pre-experiment covariates — the multivariate version is just multiple regression."

**Transition**: "When is this worth doing?"

---

## Slide 59: When CUPED Helps Most

**Time**: ~2 min
**Talking points**:

- Walk the table: revenue month-over-month ρ ≈ 0.7–0.9 → 49–81% reduction; session counts ρ ≈ 0.6–0.8 → 36–64%; brand-new features ρ ≈ 0–0.2 → 0–4%.
- Rule of thumb: CUPED works when users behave consistently over time (ρ > 0.5).
- Does NOT work for: new users (no pre-data), completely new features (no historical analog), one-time events (no repeat behaviour).
- **If beginners confused**: "This user spent a lot last month, so they probably spend a lot this month too. We remove that predictable part."
- **If experts bored**: "For new users, CUPAC uses a model-predicted covariate instead of history."

**Transition**: "What if you cannot randomise at all?"

---

## Slide 60: Difference-in-Differences (DiD)

**Time**: ~4 min
**Talking points**:

- When randomisation is impossible: policy changes, regulation, natural experiments. You observe treatment and control groups BEFORE and AFTER.
- ATT = (Ȳ_treat,post − Ȳ_treat,pre) − (Ȳ_ctrl,post − Ȳ_ctrl,pre). First difference removes group levels; second difference removes the common time trend.
- Cooling-measures example — be explicit: a HYPOTHETICAL measure on a simulated panel (Exercise 7.4), Central-region flats treated vs Non-Central control. Not an evaluation of any real policy.
- Critical assumption: parallel trends — without treatment both groups would have followed the same trajectory. Test with pre-period data.
- **If beginners confused**: "If both groups were on the same path, and then one group got a policy change, the difference in their paths IS the policy effect."
- **If experts bored**: "Parallel trends can be probed with pre-period event-study plots. Violations require synthetic control methods."

**Transition**: "Make-or-break: testing that assumption."

---

## Slide 61: Testing Parallel Trends

**Time**: ~3 min
**Talking points**:

- Pre-period validation: plot both groups over the pre-treatment period. Parallel lines (same slope, different levels) → assumption holds. Diverging before treatment → DiD unreliable.
- Placebo test: pretend treatment happened at a fake pre-period date and rerun DiD. A "significant effect" at the fake date indicts the design.
- Beyond DiD: propensity score matching (reference material) when no natural control exists.
- The causal ladder: A/B test (gold) → DiD (natural experiments) → propensity matching (observational) → instrumental variables. Each step weakens causal claims.
- **If beginners confused**: "Before trusting DiD, check that the two groups were moving in the same direction before the policy change."
- **If experts bored**: "Synthetic control (Abadie et al., 2010) builds a weighted control matching the treated unit's pre-treatment trajectory — more robust when no single control group fits."

**Transition**: "The engine record for adjustment work."

---

## Slide 62: Kailash Bridge: CUPED and Causal Inference

**Time**: ~2 min
**Talking points**:

- Be honest about this dataset: pre_metric_value correlates only weakly with revenue (ρ ≈ 0.21), so CUPED removes only ≈4% of the variance. That is the formula working as stated — reduction = ρ². CUPED pays off only when the pre-period covariate predicts the outcome strongly.
- There is no CUPED or DiD button: θ, the adjusted metric and the DiD estimate are computed in your code; the tracker records them.
- Never add a covariate measured AFTER randomisation — it can absorb the treatment effect itself.
- Exercise 7 preview: CUPED with pre-experiment covariates only, Bayesian A/B (expected loss), sequential testing (mSPRT), DiD on the simulated cooling-measure panel with a pre-trend test.
- **If beginners confused**: "CUPED subtracts the part of each user's spend that their past behaviour already predicts."
- **If experts bored**: "With several pre-period covariates, θ becomes a regression coefficient vector — Exercise 7 implements it."

**Transition**: "Lesson 2.8 — the capstone."

---

## Slide 63: Lesson 2.8: Capstone — Statistical Analysis Project

**Time**: ~1 min
**Talking points**:

- Read the objectives: a complete analysis from data to recommendations, temporal and interaction features, point-in-time feature storage, presenting to non-technical audiences.
- This is the integration lesson. The project is where you practise the end-to-end workflow that the module assessment then tests task by task.
- **If beginners confused**: "This lesson ties everything together. You choose a project and apply all the tools from the module."
- **If experts bored**: "The capstone includes FeatureStore integration with point-in-time correctness — a production-grade concern."

**Transition**: "The full pipeline and the three project options."

---

## Slide 64: From Statistics to Decision: The Full Pipeline

**Time**: ~3 min
**Talking points**:

- Walk the 7-step workflow: Load → Describe → Hypothesise → Test → Model → Interpret → Report.
- Option A: HDB Resale Valuation — typed features, point-in-time retrieval, regression with lineage (hdb_resale.parquet; the guided path in Exercise 8).
- Option B: Singapore Economic Indicators — GDP growth from trade balance, unemployment and inflation (economic_indicators.csv).
- Option C: Experiment Design & Analysis — SRM against the designed split, CUPED and a ship decision (experiment_data.parquet). The most advanced.
- **If beginners confused**: "Option A has the most guidance: Exercise 8 walks you through it. Start there if you feel uncertain."
- **If experts bored**: "Option C asks you to defend a ship or no-ship decision from a four-arm experiment with a broken arm — closest to industry practice."

**Transition**: "The bridge to M3 — better inputs."

---

## Slide 65: Feature Engineering: Creating Better Inputs

**Time**: ~3 min
**Talking points**:

- Features are model inputs; feature engineering derives better ones. Good features beat complex models. Types: temporal (month, day of week, recency), interaction (area × town), polynomial (x²), aggregation (rolling means, ratios).
- Domain knowledge drives it: knowing HDB prices depend on lease, location and area beats any algorithm.
- Walk the code: FeatureEngineer has two calls — generate() builds candidates from the declared FeatureSchema (strategies: interactions, polynomial, binning, temporal), select() ranks them against the target and keeps top_k.
- Note the two schema classes: FeatureEngineer uses the top-level kailash_ml.FeatureSchema; the FeatureStore (next slide) uses its own schema class from kailash_ml.features. They are not interchangeable.
- This is a preview; M3.1 goes deep.
- **If beginners confused**: "Feature engineering is adding columns that help the model make better predictions — new information from existing data."
- **If experts bored**: "select() uses tree-based importance by default, so it can favour features a linear model cannot use. Sanity-check the ranking with domain knowledge."

**Transition**: "And the store that keeps those features honest."

---

## Slide 66: FeatureStore: Preventing Data Leakage

**Time**: ~3 min
**Talking points**:

- The leakage problem: training a model to predict January sales using features computed from all of 2024 — but in January you did not know February-December data. Point-in-time correctness: features must only use data available at prediction time. Leakage is the #1 reason models shine in training and fail in production.
- Lifecycle: define a typed FeatureSchema (from kailash_ml.features — the top-level kailash_ml.FeatureSchema is a different class the store rejects); materialise rows (idempotent upsert keyed by entity id and timestamp, with a lineage hash); retrieve with get_features(schema, timestamp=T) — only values stamped at or before T come back.
- The features frame needs an integer transaction_id, a Datetime transaction_date and the schema's fields; Exercise 8 builds it. DataFlow is constructed with an absolute sqlite path.
- **If beginners confused**: "When predicting the future, you cannot use future data. FeatureStore only gives you data that existed at the time you are predicting."
- **If experts bored**: "The timestamp filter only protects you if timestamps are honest: a feature computed later but stamped earlier still leaks."

**Transition**: "How the capstone is graded."

---

## Slide 67: Capstone Project Rubric

**Time**: ~2 min
**Talking points**:

- Walk the rubric: pipeline completeness 20%, feature engineering 15%, statistical rigour 20%, model interpretation 20%, communication 15%, FeatureStore usage 10%.
- This rubric is for the capstone PROJECT. The graded end-of-module assessment is separate: auto-graded coding tasks in the module's assessment folder (see the final block).
- Common deductions: CI interpreted as "probability" (2.2), p-value as P(H₀) (2.3), accuracy without a confusion matrix on imbalanced data (2.6).
- Emphasise: communication is 15%. A technically perfect analysis that cannot be explained to a non-technical audience is incomplete.
- **If beginners confused**: "Focus on the pipeline and interpretation. The FeatureStore portion is a stretch goal."
- **If experts bored**: "The weighting favours interpretation and communication, not just code."

**Transition**: "The whole workflow, end to end, in code."

---

## Slide 68: Capstone Workflow: Putting It All Together

**Time**: ~3 min
**Talking points**:

- Walk the complete Option A workflow in code: load HDB resales with sentinel prices dropped; profile; hypothesise H₀ "floor area has no effect"; OLS via the normal equations (2.5); interpret residuals; log the run so every number is auditable.
- On the cleaned data the slope is ≈$9,091 per square metre with t ≈ 490 — overwhelming evidence against H₀, which is why the report leads with the EFFECT SIZE, not the p-value. In Exercise 8's validated pipeline: 3,536 impossible rows removed, R² = 0.828.
- The feature engineering and FeatureStore steps from the previous two slides slot between steps 2 and 4; Exercise 8 does the full version with point-in-time features and lineage.
- **If beginners confused**: "This is the recipe. Follow it step by step with your chosen dataset."
- **If experts bored**: "The real challenge is interpretation and business recommendation, not the code. Anyone can fit a model; explaining it is the hard part."

**Transition**: "The complete engine map for the module."

---

## Slide 69: Kailash Bridge: The Complete M2 Engine Map

**Time**: ~2 min
**Talking points**:

- Recap the lesson-to-engine mapping: 2.1 ExperimentTracker (log the posterior), 2.2 ModelVisualizer (check the fit), 2.3 ModelVisualizer (bootstrap histogram), 2.4 ExperimentTracker (design, SRM, decision), 2.5 FeatureEngineer + ModelVisualizer, 2.6 ModelVisualizer (ROC, confusion matrix), 2.7 ExperimentTracker (log the adjustment), 2.8 FeatureEngineer + FeatureStore + ExperimentTracker.
- Be precise about the division of labour: in M2 you compute the statistics; the engines log runs, build and store features, and draw plots.
- M3 preview: preprocessing, model selection, hyperparameter tuning, evaluation. TrainingPipeline, HyperparameterSearch, ModelRegistry, PreprocessingPipeline take centre stage.
- **If beginners confused**: "This table is your cheat sheet. Each lesson taught you a concept; each engine is the tool you pair it with."
- **If experts bored**: "M3 introduces the full ML pipeline pattern: preprocessing, feature engineering, model selection, tuning, evaluation, deployment."

**Transition**: "Reference cards next — photograph them."

---

## Slide 70: Formula Reference: Probability (2.1)

**Time**: ~1 min
**Talking points**:

- Reference slide — students photograph or bookmark. Bayes' theorem, expected value, product rule, total probability in one place.
- Offer to revisit any formula whose derivation felt rushed.

**Transition**: "Estimation formulas."

---

## Slide 71: Formula Reference: Estimation (2.2)

**Time**: ~1 min
**Talking points**:

- Reference slide: population vs sample variance (n−1), log-likelihood, the z-based confidence interval.

**Transition**: "Bootstrap and testing formulas."

---

## Slide 72: Formula Reference: Bootstrap and Testing (2.3)

**Time**: ~1 min
**Talking points**:

- Reference slide: bootstrap percentile CI, the T-statistic, Bonferroni, the power-analysis sample size.

**Transition**: "The SRM formula."

---

## Slide 73: Formula Reference: SRM Detection (2.4)

**Time**: ~1 min
**Talking points**:

- Reference slide: chi-squared SRM test. Remind: run it BEFORE analysing any A/B test, against the DESIGNED split; if p < 0.01, stop and investigate — randomisation is broken.

**Transition**: "Regression formulas."

---

## Slide 74: Formula Reference: Regression (2.5)

**Time**: ~1 min
**Talking points**:

- Reference slide: normal equations, the coefficient t-statistic, R², the F-statistic.

**Transition**: "Logistic and ANOVA formulas."

---

## Slide 75: Formula Reference: Logistic Regression and ANOVA (2.6)

**Time**: ~1 min
**Talking points**:

- Reference slide: sigmoid, logit link, odds ratio = e^β, the ANOVA F-ratio.

**Transition**: "CUPED and DiD formulas."

---

## Slide 76: Formula Reference: CUPED and DiD (2.7)

**Time**: ~1 min
**Talking points**:

- Reference slide: CUPED variance reduction (1 − ρ²), optimal θ, the adjusted metric, the DiD ATT.

**Transition**: "Optional deeper content per lesson."

---

## Slide 77: PDF and CDF: Probability in Pictures

**Time**: ~2 min
**Talking points**:

- Optional depth (2.1). PDF: a histogram normalised to area 1 — height is density, not probability; P(a ≤ X ≤ b) is the area between a and b. CDF: F(x) = P(X ≤ x), monotonically increasing 0 → 1, the area under the PDF up to x.
- Practical use: the CDF reads percentiles directly — "90th percentile" means F(x) = 0.90.
- **If beginners confused**: "PDF is the shape of the data. CDF tells you what percentage of data falls below any value."
- **If experts bored**: "The quantile function (inverse CDF) is used in bootstrap percentile CIs and in generating random samples."

**Transition**: "Why averages converge at all."

---

## Slide 78: Law of Large Numbers: Why Averages Converge

**Time**: ~2 min
**Talking points**:

- Optional depth (2.2). As n grows, the sample mean converges to the population mean — 10 flips might give 70% heads; 10,000 give ≈50%. Guaranteed for any distribution with a finite mean.
- Casino analogy: the house does not need to win every hand; LLN guarantees the edge converges over millions of hands.
- Gambler's fallacy: a streak of heads does NOT make tails "due". LLN is about long-run averages, not individual outcomes.
- **If beginners confused**: "More data, more reliable average. That is why pollsters survey thousands, not 10."
- **If experts bored**: "Weak LLN (in probability) vs Strong LLN (almost sure). Heavy tails need more samples."

**Transition**: "The testing workflow as a checklist."

---

## Slide 79: Hypothesis Testing: Step-by-Step Workflow

**Time**: ~3 min
**Talking points**:

- Optional depth (2.3). Walk the 8 steps: state the business question; formulate H₀/H₁; choose α; collect data (power analysis first); compute the statistic; compare to critical value or p; decide; report EFFECT SIZE, not just significance.
- Type I / Type II table: reject a true H₀ → Type I (α); fail to reject a false H₀ → Type II (β); power = 1 − β.
- A significant 0.01% improvement is not practically meaningful.
- **If beginners confused**: "Follow these steps like a recipe. The output is a yes/no decision with a confidence level."
- **If experts bored**: "CIs convey significance and effect size simultaneously — often a better report than a bare p-value."

**Transition**: "The sizing calculation, worked."

---

## Slide 80: Power Analysis: How Many Customers?

**Time**: ~3 min
**Talking points**:

- Optional depth (2.4). Hawker A/B test: MDE 5% of $10 = $0.50, σ = $4 from history, α = 0.05, power 0.80 → z(β) = 0.84.
- Compute: n = (1.96 + 0.84)² × 2 × 16 / 0.25 = 7.84 × 32 / 0.25 = 1,003.5 → round UP: 1,004 customers per group, 2,008 total.
- At 100 customers/day ≈ 21 days. One week (700 total) is underpowered.
- Show the σ sensitivity: if σ were $8, you need 4× the sample.
- **If beginners confused**: "Before running the experiment, this formula tells you how many customers you need. Run it shorter and you might miss a real effect."
- **If experts bored**: "Simulation-based power when the test statistic is non-standard; analytical formulas assume normality."

**Transition**: "Reading real regression output."

---

## Slide 81: Reading a Regression Summary Table

**Time**: ~3 min
**Talking points**:

- Optional depth (2.5). The table is an invented teaching example (the course HDB file has no MRT-distance or mature-estate columns); students produce a real one in Exercise 5.1.
- Walk each row: each sqm adds $5,200; each km from MRT reduces price by $18,500; mature estates command a $42,000 premium; all p < 0.001; R² = 0.82, F = 450.
- Ask: "Which feature has the strongest effect?" (Mature estate at $42K total, but per-unit it is dist_to_mrt at −$18.5K/km.)
- Sign = direction, magnitude = size, p-value = confidence.
- **If beginners confused**: "Read it column by column: coefficient = how much, t-stat = how sure, p-value = how surprised."
- **If experts bored**: "Practical vs statistical significance: a $10 coefficient with p < 0.001 is significant but irrelevant."

**Transition**: "Is the model itself valid?"

---

## Slide 82: Residual Analysis: Is Your Model Valid?

**Time**: ~3 min
**Talking points**:

- Optional depth (2.5). OLS assumptions checked via residuals: linearity (residuals vs fitted shows no pattern), normality (Q-Q plot), homoscedasticity (no funnel), independence (no autocorrelation).
- Heteroscedasticity: funnel shape → standard errors wrong, t-statistics unreliable. Fix: robust SEs or transform the target (log y).
- ModelVisualizer's viz.residuals(y_true, y_pred) draws predicted-vs-actual and the residual histogram; the residuals-vs-fitted and Q-Q plots are built by hand in Exercise 5.2.
- **If beginners confused**: "After fitting, we check whether the model's errors look random. A pattern means the model is missing something."
- **If experts bored**: "Durbin-Watson tests autocorrelation; Breusch-Pagan tests heteroscedasticity; White's test is more general."

**Transition**: "Odds versus probability, side by side."

---

## Slide 83: Odds vs Probability: A Visual Comparison

**Time**: ~2 min
**Talking points**:

- Optional depth (2.6). Walk the table: P = 0.50 → 1:1; 0.75 → 3:1; 0.90 → 9:1; 0.10 → 1:9.
- Conversion: odds = P/(1−P); P = odds/(1+odds).
- Common confusion: "4× the odds" ≠ "4× the probability". Baseline P = 10% (odds 0.11), 4× odds = 0.44 → P = 31% — probability rose 3.1×, not 4×.
- Betting framing helps: "3:1 odds means for every 1 time it does not happen, it happens 3 times."
- **If experts bored**: "Log-odds is the natural scale for logistic regression — the logit link makes the predictor-odds relationship linear."

**Transition**: "The loss function behind the fit."

---

## Slide 84: Cross-Entropy: The Classification Loss Function

**Time**: ~3 min
**Talking points**:

- Optional depth (2.6). L = −(1/n) Σ [yᵢ log p̂ᵢ + (1−yᵢ) log(1−p̂ᵢ)] — negated log-likelihood.
- Walk both cases: y = 1 → loss −log p̂ (penalises low predicted probability); y = 0 → loss −log(1−p̂). Perfect prediction → 0.
- Why not squared error: for binary outcomes it has multiple local minima; cross-entropy is convex — a unique optimum. Information-theoretic reading: how different the predicted distribution is from the true one.
- Deep learning bridge: the exact loss for training neural classifiers (M4).
- **If beginners confused**: "The loss measures how wrong the model's predictions are. The model adjusts coefficients to make this number as small as possible."
- **If experts bored**: "Cross-entropy = KL divergence plus the entropy of the true distribution; minimising it = minimising KL."

**Transition**: "CUPED, step by step."

---

## Slide 85: CUPED: Step-by-Step Implementation

**Time**: ~3 min
**Talking points**:

- Optional depth (2.7). Steps: identify the pre-experiment covariate; compute θ = Cov(Y, X_pre)/Var(X_pre); adjust Y_adj = Y − θ(X_pre − X̄_pre); run the usual test on Y_adj; quantify the variance reduction.
- Walk the code. θ is just the OLS slope from 2.5 — CUPED is regression applied to experiments.
- On the course experiment (control vs treatment_a), pre_metric_value is only weakly correlated with revenue (ρ ≈ 0.21), so variance falls ≈4% — an honest demonstration that CUPED is only as good as its covariate.
- **If beginners confused**: "Compute one number (θ), use it to adjust the metric, then run the same test we already know."
- **If experts bored**: "θ is estimated on both arms pooled; because the covariate is pre-experiment, the adjustment cannot absorb the treatment effect."

**Transition**: "Writing testable hypotheses."

---

## Slide 86: Formulating Experiment Hypotheses

**Time**: ~2 min
**Talking points**:

- Optional depth (2.4). Good hypotheses are specific ("BOGO gives higher average spend than 20% discount"), measurable, testable, falsifiable.
- Bad ones: "the new design is better" (no metric), "users prefer our product" (not testable with A/B), "revenue will increase eventually" (not falsifiable).
- Template: "Changing [feature] from [A] to [B] will [increase/decrease] [metric] by [amount] within [timeframe]."
- Have students formulate the hawker-centre hypothesis with the template.
- **If experts bored**: "Pre-registration prevents p-hacking. Registration platforms: AsPredicted, OSF."

**Transition**: "Two flavours of bootstrap."

---

## Slide 87: Parametric vs Non-Parametric Bootstrap

**Time**: ~2 min
**Talking points**:

- Optional depth (2.3). Non-parametric: resample the data directly — no assumptions, any statistic, needs n ≥ 30. Default choice.
- Parametric: assume a distribution, fit it, generate new samples from it. More powerful IF the assumption is right; use when you trust the distribution and n is very small.
- **If beginners confused**: "Non-parametric = resample the data. Parametric = generate from a fitted distribution. When in doubt, non-parametric."
- **If experts bored**: "BCa is the recommended non-parametric method — corrects bias and skewness."

**Transition**: "Total probability with a hawker queue."

---

## Slide 88: Total Probability: The Hawker Centre Queue

**Time**: ~2 min
**Talking points**:

- Optional depth (2.1). Scenario: 3 stalls — A (40% of customers), B (35%), C (25%); P(wait > 10 min): A 20%, B 30%, C 10%.
- P(Wait) = 0.20×0.40 + 0.30×0.35 + 0.10×0.25 = 0.08 + 0.105 + 0.025 = 0.21.
- Total probability sums over all pathways, each weighted by its probability. It is the denominator of Bayes' theorem.
- Ask: "If you waited more than 10 minutes, which stall were you most likely at?" (B — largest contribution, 0.105.)
- **If experts bored**: "Total probability is the law of total expectation applied to indicators."

**Transition**: "Keeping models honest — cross-validation."

---

## Slide 89: Cross-Validation: Preventing Overfitting

**Time**: ~2 min
**Talking points**:

- Optional depth (2.5). Train/test split: fit on 80%, evaluate on 20%. Training R² 0.95 with test R² 0.60 = overfitting.
- K-fold: k equal folds; train on k−1, test on the remaining one; rotate; average. More reliable than a single split.
- Detailed in M3.2. The key message now: never evaluate on training data.
- **If beginners confused**: "Studying only the practice exam, then sitting the real exam. If the real one differs, your score drops. Cross-validation simulates the real exam."
- **If experts bored**: "Stratified k-fold preserves class proportions; time-series needs temporal splits — M3.2."

**Transition**: "Where SRM comes from and how to fix it."

---

## Slide 90: SRM: Common Causes and How to Fix Them

**Time**: ~2 min
**Talking points**:

- Optional depth (2.4/2.7). Walk the cause table: bot filtering (apply the filter BEFORE assignment), redirect bugs (fix and monitor), population filtering (same eligibility criteria for both groups), caching (cache-bust per-user assignment).
- If SRM is detected: stop the analysis, investigate, fix, re-run. Do NOT analyse SRM-contaminated data — the groups are not comparable.
- SRM is a data engineering problem, not a statistics problem.
- **If beginners confused**: "Something in the system is sending more users to one group than the other."
- **If experts bored**: "Well-run experimentation platforms halt or flag experiments automatically when SRM fires. Kohavi et al. catalogue 20+ causes."

**Transition**: "The Bayesian origin of regularisation."

---

## Slide 91: MAP Estimation and the Regularisation Connection

**Time**: ~3 min
**Talking points**:

- Optional depth (2.2). MAP = MLE + log-prior. Gaussian prior → log P(θ) = −λ‖θ‖² + C → L2 (Ridge). Laplace prior → L1 (Lasso) → sparse solutions.
- "A Gaussian prior says you believe coefficients are small — that is Ridge. A Laplace prior says most coefficients are zero — that is Lasso."
- Bridge to M3: Ridge and Lasso are regularised ML models; now you know their Bayesian origin.
- **If beginners confused**: "Regularisation prevents the model from fitting the noise. It connects back to the prior beliefs from 2.1."
- **If experts bored**: "Elastic net mixes the two priors. The regularisation path is computed efficiently with coordinate descent."

**Transition**: "Presenting to non-technical audiences."

---

## Slide 92: Presenting Statistical Findings

**Time**: ~2 min
**Talking points**:

- Optional depth (2.8). Do: lead with the business answer ("Yes, the increase reduced revenue by 5%"); quantify uncertainty ("95% confident the true reduction is 2-8%"); visualise; state limitations; recommend actions.
- Do NOT: lead with p-values; show raw output tables; claim causation without experimental design; report only "significant" results; use jargon.
- Ask: "Would you tell a CEO 'we reject the null at α = 0.05'?" No — "the data strongly suggests the treatment works."
- The capstone report is graded 15% on communication.
- **If experts bored**: "Tufte's 'The Visual Display of Quantitative Information' is the visualisation gold standard."

**Transition**: "The subtlest leakage of all."

---

## Slide 93: Point-in-Time Correctness: The Subtle Data Leakage

**Time**: ~3 min
**Talking points**:

- Optional depth (2.8). Scenario: in March 2025 you build a model predicting January 2025 sales. Feature "average_monthly_spend" from all of 2024 — correct, December 2024 data existed in January.
- But a "total_2024_spend" finalised only in February 2025 leaks: in January the system had 11 months, not 12. The model uses future information.
- FeatureStore enforces this: features "as of January 1, 2025" returns only values computed and available on that date.
- Typical consequence (illustrative): a leaky model shows R² = 0.95 in backtesting but R² = 0.40 in production — the gap is the leaked signal.
- **If beginners confused**: "When predicting the past, make sure features only use data actually available at that past time."
- **If experts bored**: "Point-in-time retrieval is standard in production feature stores; building training sets any other way invites leakage."

**Transition**: "One decision guide for the whole toolkit."

---

## Slide 94: Choosing Your Statistical Tool

**Time**: ~2 min
**Talking points**:

- Optional depth (2.6). Decision guide: continuous outcome — 2 groups → t-test, 3+ → ANOVA, with predictors → linear regression; binary outcome → logistic regression; comparing proportions → chi-squared; no distributional assumptions → permutation test or bootstrap.
- M2 toolkit table: Bayes (2.1), MLE/MAP (2.2), bootstrap (2.3), A/B design (2.4), linear regression (2.5), logistic regression (2.6), CUPED/DiD (2.7), feature engineering (2.8).
- Start with the outcome type — number or yes/no — and the tool chooses itself.
- **If experts bored**: "Regression subsumes most of these: t-test is regression with one binary predictor; ANOVA with one categorical predictor."

**Transition**: "The data plumbing behind the statistics."

---

## Slide 95: Clean DataOps: From Silos to Pipelines

**Time**: ~2 min
**Talking points**:

- Optional depth (2.4). The problem: clicks in a web-analytics tool, revenue in a CRM, usage in a product-analytics tool, costs in an ERP — nobody can join them without manual export/import.
- Clean architecture: agreed schema, automated ETL (not CSV email), version-controlled transformations, standard connectors.
- Kailash DataFlow provides connectors, schema validation and automated transfers; the data collection plan from 2.4 maps directly to DataFlow pipelines. FeatureStore persists its feature tables through DataFlow — the FeatureStore(DataFlow(...)) call from 2.8.
- Most experiment failures are data engineering failures, not statistics failures.
- **If beginners confused**: "Before you can do statistics, you need clean, unified data. This slide is about getting the data house in order."

**Transition**: "Module recap."

---

## Slide 96: Module 2: Key Takeaways

**Time**: ~2 min
**Talking points**:

- Statistical foundations: probability is the language of uncertainty — Bayes updates beliefs with evidence. Estimation connects data to parameters — MLE maximises likelihood; MAP adds priors. Hypothesis testing is decision-making under uncertainty — p is NOT P(H₀). Bootstrap works for any statistic without distributional assumptions.
- Modelling and experiments: linear regression quantifies relationships (R², t, F). Logistic regression predicts binary outcomes (odds ratios). CUPED reduces experiment variance with pre-experiment data. Feature engineering creates better inputs; FeatureStore prevents leakage.
- One bullet per lesson — quick recap.
- **If beginners confused**: "You have learned the statistical foundations every ML model is built on."
- **If experts bored**: "M3 builds on every technique here: regularisation (MAP), cross-validation, model selection, ensembles."

**Transition**: "The do-not-do-this checklist."

---

## Slide 97: M2 Common Mistakes Checklist

**Time**: ~2 min
**Talking points**:

- Walk the 8 mistakes: CI = "95% chance parameter is in range" (no — 95% of repeated intervals contain it); p-value = P(H₀ true) (no — P(data | H₀)); "accept H₀" ("fail to reject"); odds ratio = risk ratio (only for rare events); high R² = good model (check residuals, test set, adjusted R²); accuracy on imbalanced data (confusion matrix, precision, recall); SRM ignored (always check first); no power analysis (size before running).
- These are the common capstone deductions — have students check their capstone against the list before submitting.
- **If experts bored**: "Add: multiple comparisons without correction, and correlation read as causation."

**Transition**: "Where we go next."

---

## Slide 98: What Comes Next: Module 3 Preview

**Time**: ~2 min
**Talking points**:

- M3: Supervised ML — theory to production: 3.1 feature engineering/selection, 3.2 bias-variance/regularisation/CV, 3.3 the model zoo, 3.4 gradient boosting, 3.5 evaluation/imbalance/calibration, 3.6 interpretability/fairness, 3.7 orchestration/registry/hyperparameter search, 3.8 production — DataFlow, drift, deployment.
- How M2 prepares you: regression (2.5) becomes one option among many; logistic regression (2.6) the baseline classifier; CV (2.5 preview) gets full treatment in 3.2; feature engineering (2.8) is automated with FeatureEngineer; MLE (2.2) becomes the loss functions of all models; bootstrap (2.3) gives CIs for model metrics.
- Build anticipation.
- **If beginners confused**: "M2 gave you the statistical foundation. M3 uses it to build real ML models."
- **If experts bored**: "M3 introduces the full ML pipeline pattern. The Kailash engines automate preprocessing, training, tuning, deployment."

**Transition**: "What the assessment looks like."

---

## Slide 99: End of Module Assessment

**Time**: ~2 min
**Talking points**:

- Module assessment (individual): practical coding tasks on the course datasets — no multiple choice. Auto-graded on outcomes with exact answers and strict numeric tolerances. Expect: Bayes and SRM, bootstrap CIs and multiple-testing corrections, CUPED, OLS and logistic inference, a point-in-time feature table. Open book; AI assistants not allowed. Details in the module's assessment/README.md.
- The grader checks the numbers a correct analysis produces — choosing the right population, covariate and correction matters more than syntax.
- Capstone presentation (team): 5 minutes to non-technical stakeholders — business question, methodology summary, key findings, recommendations, limitations. Marked with the 2.8 rubric.
- **If beginners confused**: "The coding tasks check that you can do the statistics correctly. The capstone checks that you can apply and explain them."
- **If experts bored**: "The presentation is where you differentiate yourself. Technical people who communicate clearly are rare and valuable."

**Transition**: "Close the module."

---

## Slide 100: Statistical Mastery for Machine Learning and AI Success

**Time**: ~1 min
**Talking points**:

- Thank the class. Read the closing provocation: "The goal is not to compute a p-value. The goal is to make a better decision."
- Remind them of the capstone deadline and the assessment date. Point to the M3 preview.
- **If beginners confused**: "You just completed a formula-heavy statistics module. Well done."
- **If experts bored**: "The exercises have stretch goals. Push yourself on 2.7 (CUPED implementation) and 2.8 (FeatureStore integration)."
