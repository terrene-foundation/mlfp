# Team Capstone Rubric

**Weight**: 35% of final grade  
**Criteria**: 5 dimensions with differentiated weights

---

## Grading Scale

| Level             | Score Range | Meaning                                        |
| ----------------- | ----------- | ---------------------------------------------- |
| Excellent         | 85-100%     | Exceeds expectations; demonstrates mastery     |
| Good              | 70-84%      | Meets all requirements with minor gaps         |
| Satisfactory      | 50-69%      | Meets minimum requirements; notable weaknesses |
| Needs Improvement | 0-49%       | Missing requirements or fundamental errors     |

---

## Criterion 1: System Architecture (20%)

| Level                 | Score   | Indicators                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| --------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Excellent**         | 85-100% | Clean separation of concerns across components (pipeline, model, serving, governance/agents). Each Kailash package used idiomatically with proper integration points. Architecture diagram accurately reflects the implementation. Data flows through well-defined interfaces, not ad-hoc function calls. Configuration externalised (no hardcoded values). Error handling at system boundaries (what happens when DataFlow fails? When the model returns an unexpected output?). System can be understood from the architecture document without reading every line of code. |
| **Good**              | 70-84%  | Clear component separation with well-defined responsibilities. Kailash packages integrated correctly. Architecture document matches implementation with minor discrepancies. Most error handling in place. Configuration mostly externalised.                                                                                                                                                                                                                                                                                                                                 |
| **Satisfactory**      | 50-69%  | Components exist but boundaries are blurry (model training code mixed with serving logic). Kailash packages used but integration is fragile (manual steps between components). Architecture document exists but diverges from implementation. Limited error handling. Some hardcoded configuration.                                                                                                                                                                                                                                                                           |
| **Needs Improvement** | 0-49%   | Monolithic code with no clear component separation. Kailash packages used in isolation without integration. No architecture document or document bears no resemblance to the code. No error handling. Hardcoded paths, credentials, or configuration throughout.                                                                                                                                                                                                                                                                                                              |

### What We Look For

- Can you swap out the model without changing the serving layer?
- Does the data pipeline failure propagate gracefully or crash the entire system?
- Is the architecture document something you would hand to a new team member?
- Are the Kailash packages talking to each other through their intended interfaces?

---

## Criterion 2: Technical Depth (25%)

| Level                 | Score   | Indicators                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| --------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Excellent**         | 85-100% | ML technique selection justified with statistical reasoning (not just "XGBoost is popular"). Feature engineering demonstrates domain understanding, not just mechanical transformations. Hyperparameter tuning is systematic (grid/random/Bayesian via `HyperparameterSearch`, not manual). Model evaluation uses appropriate metrics for the problem type with confidence intervals. Results interpreted in domain context ("a 3% improvement in recall catches 150 more fraud cases per month"). Edge cases and failure modes analysed. |
| **Good**              | 70-84%  | Technique selection reasonable with justification. Feature engineering shows thought beyond defaults. Systematic hyperparameter tuning present. Appropriate evaluation metrics with some statistical rigour. Domain context present in interpretation.                                                                                                                                                                                                                                                                                    |
| **Satisfactory**      | 50-69%  | Standard techniques applied correctly but without justification. Basic feature engineering (one-hot encoding, scaling). Hyperparameter tuning attempted but limited (few parameters, small grid). Evaluation uses standard metrics without confidence intervals. Results reported as numbers without domain interpretation.                                                                                                                                                                                                               |
| **Needs Improvement** | 0-49%   | Inappropriate technique for the problem (regression for classification, clustering without evaluation). No feature engineering beyond raw columns. No hyperparameter tuning. Wrong evaluation metrics or metrics computed incorrectly. No interpretation of results.                                                                                                                                                                                                                                                                      |

### What We Look For

- Can the team explain why they chose their approach over alternatives?
- Does the feature engineering reflect understanding of the domain, not just applying every transformation available?
- Are the results meaningful in context, or just numbers on a screen?
- Has the team identified where their model struggles and why?

---

## Criterion 3: Production Quality (25%)

| Level                 | Score   | Indicators                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| --------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Excellent**         | 85-100% | `DriftMonitor` configured with justified thresholds (based on analysis of feature distributions, not arbitrary values). `ModelRegistry` contains versioned models with metadata (training data hash, hyperparameters, evaluation metrics, training timestamp). Nexus API validates inputs, returns structured errors, and handles edge cases. PACT governance spec is actionable (clear D/T/R assignments, realistic operating envelope) or Kaizen agents have meaningful safety constraints. One-command setup and deployment scripts work from clean environment. Monitoring alerts defined with clear escalation. Comprehensive test coverage for critical paths. |
| **Good**              | 70-84%  | Drift monitoring configured with reasonable thresholds. Model registry has versioned models with basic metadata. API validates inputs and returns structured errors. Governance/agent specs present and reasonable. Setup scripts work with minor manual steps. Tests cover main paths.                                                                                                                                                                                                                                                                                                                                                                              |
| **Satisfactory**      | 50-69%  | Basic drift monitoring present but thresholds arbitrary. Models saved but versioning is ad-hoc. API works for happy path but returns unhelpful errors for bad input. Governance/agent spec exists but is generic (could apply to any system). Setup requires significant manual steps. Minimal tests.                                                                                                                                                                                                                                                                                                                                                                |
| **Needs Improvement** | 0-49%   | No drift monitoring or monitoring is non-functional. Models not versioned. API crashes on unexpected input. No governance or agent specification. System cannot be set up from README instructions. No tests.                                                                                                                                                                                                                                                                                                                                                                                                                                                        |

### What We Look For

- Would an operations team accept this system for deployment?
- Do the drift monitoring thresholds make sense for this specific data?
- Does the governance specification actually constrain the system, or is it boilerplate?
- Can a new team member set up and run the system from the README alone?
- What happens when things go wrong? (Bad data, missing features, model timeout)

---

## Criterion 4: Presentation (15%)

| Level                 | Score   | Indicators                                                                                                                                                                                                                                                                                                                                                                                                                             |
| --------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Excellent**         | 85-100% | Live demo runs smoothly and tells a coherent story (problem to solution). Every component demonstrated with real data flowing through the system. Team members present confidently and handle Q&A with depth (explaining trade-offs, acknowledging limitations, proposing alternatives). Time management is precise (15 minutes, no rushing at the end). Architecture decisions explained in terms of impact, not just implementation. |
| **Good**              | 70-84%  | Live demo works with minor hiccups. All components shown. Team handles most Q&A well. Good time management. Explanations clear but occasionally implementation-focused rather than impact-focused.                                                                                                                                                                                                                                     |
| **Satisfactory**      | 50-69%  | Demo works but feels rehearsed/scripted with limited ability to deviate. Some components only briefly shown. Q&A answers are shallow or deflected. Time management issues (rushing or running over). Explanations focus on what was built rather than why.                                                                                                                                                                             |
| **Needs Improvement** | 0-49%   | Demo fails significantly or is pre-recorded without justification. Major components missing from demo. Q&A reveals team members do not understand components they did not personally build. Poor time management (under 10 minutes or over 20). One team member dominates while others are silent.                                                                                                                                     |

### What We Look For

- Does the demo show a working system, not a slideshow?
- Can every team member explain every component (not just their own)?
- Are Q&A answers thoughtful, or does the team panic at unexpected questions?
- Does the presentation respect the audience's time?

---

## Criterion 5: Teamwork (15%)

| Level                 | Score   | Indicators                                                                                                                                                                                                                                                                                                                                                                                                                              |
| --------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Excellent**         | 85-100% | Contribution log shows balanced workload across all members. Git history confirms contributions from all members throughout the project (not just at the end). Each member owns a distinct component and can speak to others' work. Code review evidence (PR comments, review approvals) shows collaborative development. Integration points between components show coordinated design (consistent interfaces, shared data contracts). |
| **Good**              | 70-84%  | Contribution log shows reasonable distribution with minor imbalance. Git history shows regular contributions from all members. Each member owns a component. Some code review evidence. Integration is functional with minor inconsistencies.                                                                                                                                                                                           |
| **Satisfactory**      | 50-69%  | Noticeable workload imbalance (one member with significantly fewer contributions). Git history shows bulk commits (suggesting last-minute work from some members). Components work but integration feels bolted-on rather than designed together. Limited code review evidence.                                                                                                                                                         |
| **Needs Improvement** | 0-49%   | One or more members contributed minimally (fewer than 10% of meaningful commits). Git history shows one member doing most of the work. Components do not integrate well (different coding styles, inconsistent interfaces, duplicated logic). No code review evidence. Contribution log contradicted by git history.                                                                                                                    |

### What We Look For

- Does the git history tell a story of collaborative development or last-minute panic?
- Can every team member explain the system end-to-end, not just their piece?
- Did the team design the integration points together, or did they bolt independent pieces together at the end?
- Is the contribution log honest and consistent with the git history?

### Individual Grade Adjustment

In cases of significant contribution imbalance, individual grades may be adjusted:

| Situation                                                          | Adjustment                                                        |
| ------------------------------------------------------------------ | ----------------------------------------------------------------- |
| Git history and contribution log confirm balanced work             | No adjustment                                                     |
| Minor imbalance (one member did less but contributed meaningfully) | Up to -5% for under-contributor                                   |
| Major imbalance (one member's contribution is minimal)             | Up to -15% for under-contributor, up to +5% for over-contributors |
| Peer feedback indicates a member did not participate               | Referred to module coordinator for individual assessment          |

---

## Overall Grade Calculation

```
Capstone Grade = (Architecture x 0.20) + (Technical Depth x 0.25)
               + (Production Quality x 0.25) + (Presentation x 0.15)
               + (Teamwork x 0.15)
```

### Grade Boundaries

| Grade | Score   | Description                            |
| ----- | ------- | -------------------------------------- |
| A+    | 90-100% | Exceptional across all criteria        |
| A     | 85-89%  | Excellent in most, good in all         |
| A-    | 80-84%  | Good to excellent across the board     |
| B+    | 75-79%  | Consistently good with some excellence |
| B     | 70-74%  | Meets all requirements well            |
| B-    | 65-69%  | Meets requirements with notable gaps   |
| C+    | 60-64%  | Satisfactory with some strengths       |
| C     | 55-59%  | Meets minimum requirements             |
| D     | 50-54%  | Barely meets minimum requirements      |
| F     | 0-49%   | Does not meet minimum requirements     |

---

## Automatic Penalties

| Issue                                                                        | Penalty                              |
| ---------------------------------------------------------------------------- | ------------------------------------ |
| System does not run from README instructions                                 | -10% from total                      |
| Fewer than 3 Kailash packages used                                           | -15% from Architecture               |
| Missing mandatory component (pipeline, model, deployment, governance/agents) | -10% per missing component           |
| Hardcoded API keys or credentials in repository                              | -10% from total                      |
| No model card                                                                | -10% from Production Quality         |
| No contribution log                                                          | -10% from Teamwork                   |
| No live demo attempted (pre-recorded only without justification)             | -10% from Presentation               |
| Late submission                                                              | -5% per calendar day                 |
| Academic integrity violation                                                 | Referred to academic integrity board |

---

## Moderation

All capstones are first-marked by the module tutor and moderated by a second marker. Presentation scores include input from at least two assessors present during the demo. Moderated grades are final.
