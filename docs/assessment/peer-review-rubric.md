# Peer Review Rubric

**Weight**: 10% of final grade  
**Format**: Written review of another team's capstone  
**Criteria**: 4 dimensions, equally weighted at 25% each

---

## Overview

Each student submits an individual peer review of another team's capstone project. The review is graded on the quality of your feedback, not on the quality of the team you are reviewing.

A strong peer review demonstrates that you can read and evaluate production ML code, identify strengths and weaknesses in system design, and provide feedback that helps the other team improve. These are core skills for senior ML engineers who participate in code reviews, architecture reviews, and model governance boards.

### What You Review

You will receive access to another team's capstone repository and architecture document. Your review must cover:

1. **Code quality** -- Is the code readable, well-structured, and following Kailash patterns?
2. **Documentation** -- Is the documentation complete, accurate, and useful?
3. **Architecture** -- Are the framework choices appropriate and well-integrated?
4. **Constructive feedback** -- Are your suggestions specific, actionable, and respectful?

### Submission Format

Submit your peer review as a single document (PDF or Markdown, 3-5 pages) structured with the four sections above. Each section should include specific examples from the code or documentation you reviewed, not generic observations.

---

## Grading Scale

| Level             | Score Range | Meaning                                                     |
| ----------------- | ----------- | ----------------------------------------------------------- |
| Excellent         | 85-100%     | Thorough, insightful review with actionable feedback        |
| Good              | 70-84%      | Solid review covering all areas with useful feedback        |
| Satisfactory      | 50-69%      | Review covers required areas but lacks depth or specificity |
| Needs Improvement | 0-49%       | Superficial review or missing sections                      |

---

## Criterion 1: Code Quality Review (25%)

| Level                 | Score   | Indicators                                                                                                                                                                                                                                                                                                                                                                                            |
| --------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Excellent**         | 85-100% | Identifies specific code patterns (good and bad) with file paths and line references. Comments on naming conventions, function decomposition, and error handling. Recognises idiomatic Kailash engine usage versus workarounds. Points out potential bugs or edge cases with reasoning. Acknowledges well-written code, not just problems. Identifies testing gaps with suggestions for what to test. |
| **Good**              | 70-84%  | References specific files and code sections. Identifies major code quality issues and strengths. Comments on Kailash pattern usage. Provides reasoning for observations. Balanced between praise and critique.                                                                                                                                                                                        |
| **Satisfactory**      | 50-69%  | General observations about code quality without specific references ("the code is generally clean"). Identifies obvious issues but misses subtleties. Limited engagement with Kailash-specific patterns. Feedback is descriptive ("this function is long") rather than analytical ("this function mixes data loading with transformation, which makes it hard to test independently").                |
| **Needs Improvement** | 0-49%   | No specific code references. Comments are entirely generic ("code looks fine" or "code needs improvement"). No engagement with Kailash patterns. Review could apply to any codebase without modification.                                                                                                                                                                                             |

### What Strong Code Review Looks Like

- "In `src/pipeline/ingest.py` (lines 45-72), the error handling catches all exceptions with a bare `except` and logs them as warnings. This means schema validation failures are silently swallowed. Consider catching `ValidationError` specifically and raising on critical failures."
- "The team's use of `FeatureStore` in `src/features/store.py` is excellent -- features are versioned with timestamps and the naming convention is consistent. One suggestion: the feature descriptions are missing for the engineered features (lines 88-95), which would help future maintainers understand the business logic."

---

## Criterion 2: Documentation Review (25%)

| Level                 | Score   | Indicators                                                                                                                                                                                                                                                                                                                                                                                                       |
| --------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Excellent**         | 85-100% | Evaluates the README for setup completeness (did you actually try the instructions?). Reviews model cards for substantive content versus boilerplate. Identifies gaps between the architecture document and the implementation. Checks whether the governance/agent specification is actionable or generic. Comments on whether visualisations and explanations would make sense to a non-technical stakeholder. |
| **Good**              | 70-84%  | Reviews documentation for completeness and accuracy. Identifies major gaps. Comments on model card quality. Notes discrepancies between documentation and code. Provides suggestions for improvement.                                                                                                                                                                                                            |
| **Satisfactory**      | 50-69%  | Confirms documentation exists but does not evaluate its quality deeply. Notes obvious gaps (missing sections) but does not assess whether existing sections are substantive. Does not attempt to follow setup instructions.                                                                                                                                                                                      |
| **Needs Improvement** | 0-49%   | Does not review documentation or provides only surface-level comments ("documentation is present"). No engagement with model cards. No verification of documentation accuracy.                                                                                                                                                                                                                                   |

### What Strong Documentation Review Looks Like

- "The model card's Limitations section lists 'model may not generalise to unseen data' -- this is true of every model and is not useful. A better limitation would reference the specific data coverage gaps: the training data covers 2019-2024, so predictions for 2025+ are extrapolations. The team's EDA (notebook cell 14) actually shows a distributional shift in 2023 that the model card does not mention."
- "I followed the README setup instructions on a clean environment. Step 3 (`uv run python scripts/train.sh`) fails because `train.sh` is a shell script, not a Python file. The correct command should be `bash scripts/train.sh` or the script needs a shebang line."

---

## Criterion 3: Architecture Review (25%)

| Level                 | Score   | Indicators                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| --------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Excellent**         | 85-100% | Evaluates whether the chosen Kailash packages are appropriate for the problem (not just present). Identifies integration strengths and weaknesses between components. Comments on data flow through the system (is it clear, are there bottlenecks?). Assesses whether the governance/agent design adds genuine value or is bolted on for marks. Suggests alternative architectural approaches where appropriate, with reasoning. Identifies scalability or reliability concerns. |
| **Good**              | 70-84%  | Comments on package selection appropriateness. Identifies major integration issues. Evaluates data flow. Provides some architectural suggestions. Recognises where components are well-integrated versus loosely coupled.                                                                                                                                                                                                                                                         |
| **Satisfactory**      | 50-69%  | Confirms packages are used but does not evaluate whether they are the right choice. Limited analysis of integration quality. Comments on architecture at a surface level ("they used DataFlow for the pipeline"). No alternative approaches suggested.                                                                                                                                                                                                                            |
| **Needs Improvement** | 0-49%   | Does not evaluate architecture beyond confirming it exists. No analysis of package selection, integration, or data flow. No understanding demonstrated of how the components should work together.                                                                                                                                                                                                                                                                                |

### What Strong Architecture Review Looks Like

- "The team uses Nexus for serving but the API endpoint in `src/serving/api.py` bypasses the Nexus session manager and creates a raw Flask route. This means they lose Nexus's built-in rate limiting and input validation. Switching to the Nexus channel pattern would give them these features without custom code."
- "The PACT governance specification defines operating envelopes for the model, but the envelope is not enforced in code -- it exists only as a document. Connecting the envelope to the DriftMonitor alerts would make the governance actionable: when the model operates outside its envelope (e.g., input distributions exceed PSI thresholds), the system would flag the violation automatically."

---

## Criterion 4: Constructive Feedback (25%)

| Level                 | Score   | Indicators                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| --------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Excellent**         | 85-100% | Every criticism is paired with a specific, actionable suggestion. Feedback is prioritised (what matters most versus nice-to-haves). Tone is professional and respectful throughout. Strengths are acknowledged genuinely, not as a formality before criticism. Suggestions include concrete implementation guidance ("consider using `DriftMonitor.set_threshold(feature='price', psi=0.15)` based on your EDA showing the price distribution has a PSI of 0.08 quarter-to-quarter"). Overall review would genuinely help the team improve their system. |
| **Good**              | 70-84%  | Most criticisms include suggestions. Feedback is mostly prioritised. Tone is professional. Strengths acknowledged. Suggestions are actionable though not always concrete. Review would be useful to the team.                                                                                                                                                                                                                                                                                                                                            |
| **Satisfactory**      | 50-69%  | Criticisms identified but suggestions are vague ("improve error handling" without saying how or where). No clear prioritisation. Tone is acceptable but reads as a checklist rather than a thoughtful review. Strengths mentioned briefly. Review provides limited value to the team.                                                                                                                                                                                                                                                                    |
| **Needs Improvement** | 0-49%   | Criticism without suggestions. Harsh or dismissive tone. No acknowledgement of strengths. Feedback is not actionable ("this is bad" or "this is good" without reasoning). Review would not help the team. Alternatively, review is entirely positive with no substantive critique (unhelpfully uncritical).                                                                                                                                                                                                                                              |

### What Strong Constructive Feedback Looks Like

- "The drift monitoring thresholds are all set to the same value (PSI > 0.25). From the team's own EDA, the `floor_area_sqm` feature has very low variance (PSI between reference windows is ~0.02), while `town` encoding shifts more naturally (~0.12). I'd suggest per-feature thresholds: tighter for stable features (PSI > 0.1 for floor_area), looser for naturally variable ones (PSI > 0.3 for town). This would reduce false alerts while catching genuine drift earlier."
- "One strength I want to highlight: the team's feature engineering in `src/features/engineering.py` is exceptionally well-documented. Every engineered feature has a docstring explaining the business rationale, not just the transformation. This is a pattern I plan to adopt in my own work."

---

## Overall Grade Calculation

```
Peer Review Grade = (Code Quality x 0.25) + (Documentation x 0.25)
                  + (Architecture x 0.25) + (Constructive Feedback x 0.25)
```

### Grade Boundaries

| Grade | Score   | Description                                                         |
| ----- | ------- | ------------------------------------------------------------------- |
| A+    | 90-100% | Exceptional review demonstrating senior-level engineering judgement |
| A     | 85-89%  | Excellent, thorough review with actionable insights                 |
| A-    | 80-84%  | Strong review across all dimensions                                 |
| B+    | 75-79%  | Good review with solid analysis                                     |
| B     | 70-74%  | Meets all requirements with useful feedback                         |
| B-    | 65-69%  | Meets requirements with some gaps in depth                          |
| C+    | 60-64%  | Satisfactory with some strengths                                    |
| C     | 55-59%  | Meets minimum requirements                                          |
| D     | 50-54%  | Barely meets minimum requirements                                   |
| F     | 0-49%   | Does not meet minimum requirements                                  |

---

## Automatic Penalties

| Issue                                                      | Penalty                                          |
| ---------------------------------------------------------- | ------------------------------------------------ |
| Review is fewer than 2 pages                               | -10% from total                                  |
| No specific code or file references anywhere in the review | -15% from total                                  |
| Review contains no positive feedback (entirely negative)   | -10% from Constructive Feedback                  |
| Review contains no critical feedback (entirely positive)   | -10% from Constructive Feedback                  |
| Disrespectful or unprofessional tone                       | -15% from total + referral to module coordinator |
| Late submission                                            | -5% per calendar day                             |

---

## Review Assignment

- Reviews are assigned randomly (no team reviews its own capstone)
- Each team's capstone is reviewed by 2-3 individual reviewers from different teams
- Reviews are single-blind: reviewers see the team's work but the team does not see reviewer identities during grading
- After grades are finalised, anonymised reviews are shared with the reviewed team as developmental feedback

---

## Tips for Writing a Strong Review

1. **Read the code before the documentation.** Form your own understanding of what the system does, then check if the documentation matches.
2. **Try to run the system.** Even if you cannot run the full pipeline, attempt the setup. This reveals documentation gaps immediately.
3. **Be specific.** "The code could be improved" is not feedback. "The `train_model()` function in `src/training/train.py` (line 84) loads data inside the training loop, which means data is re-loaded on every hyperparameter trial" is feedback.
4. **Prioritise.** A review with 3 high-impact suggestions is more valuable than one with 20 nitpicks.
5. **Write the review you would want to receive.** Honest, specific, respectful, and actionable.
