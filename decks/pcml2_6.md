---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 2.6: Sequential Testing and Causal Inference

### Module 2: Statistical Mastery

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Apply sequential testing to stop experiments early with valid results
- Implement Difference-in-Differences (DiD) for natural experiments
- Use propensity score matching for observational causal inference
- Distinguish when each causal inference method is appropriate

---

## Recap: Lesson 2.5

- CUPED removes predictable variation using pre-experiment data
- Stratified randomisation balances groups on known confounders
- Variance reduction can halve required sample sizes
- Always pre-register analysis plans before seeing results

---

## The Peeking Problem

```
Day 3:  p = 0.02  → "Significant! Ship it!"
Day 7:  p = 0.08  → (would have been non-significant)
Day 14: p = 0.04  → (significant again)
Day 21: p = 0.12  → (not significant at all)

Checking p-values repeatedly inflates Type I error.
At α=0.05, checking 5 times gives ~14% false positive rate.
```

Sequential testing solves this: **valid stopping rules** that control error.

---

## Sequential Testing: The Idea

Instead of fixed-sample testing, define **spending functions** that allocate your error budget over time.

```
Fixed test:       All α at the end
Sequential test:  Spend a little α at each look

          α budget remaining
  α │████████████████
    │███████████████     look 1: spend 0.005
    │██████████████      look 2: spend 0.008
    │████████████        look 3: spend 0.012
    │█████████           look 4: spend 0.025
    └────────────────→ Time
```

---

## O'Brien-Fleming Boundaries

Conservative early, generous late -- matching typical experiment needs.

```python
from scipy.stats import norm
import numpy as np

def obrien_fleming_boundary(n_looks, alpha=0.05):
    """Compute O'Brien-Fleming boundaries for n interim looks."""
    boundaries = []
    for k in range(1, n_looks + 1):
        info_fraction = k / n_looks
        z_boundary = norm.ppf(1 - alpha / 2) / np.sqrt(info_fraction)
        boundaries.append(z_boundary)
    return boundaries

boundaries = obrien_fleming_boundary(4)
# [4.05, 2.86, 2.34, 2.02]
# Very hard to stop early, easy to stop at the end
```

---

## Sequential Test in Practice

```python
import numpy as np
from scipy import stats

def sequential_test(control, treatment, boundaries, look_index):
    """Test at interim look with O'Brien-Fleming boundary."""
    n_c, n_t = len(control), len(treatment)
    diff = treatment.mean() - control.mean()
    se = np.sqrt(control.var()/n_c + treatment.var()/n_t)
    z_stat = diff / se

    boundary = boundaries[look_index]

    if abs(z_stat) > boundary:
        return "STOP", z_stat, boundary
    else:
        return "CONTINUE", z_stat, boundary
```

---

## When to Use Sequential Testing

| Scenario                       | Method                        |
| ------------------------------ | ----------------------------- |
| Fixed sample, single analysis  | Standard t-test               |
| Regular interim looks (weekly) | O'Brien-Fleming               |
| Continuous monitoring          | Always-valid p-values (mSPRT) |
| Want to stop for futility too  | Two-sided sequential          |
| Bayesian preference            | Bayesian stopping rules       |

Sequential testing is standard practice at tech companies running A/B tests.

---

## From Experiments to Observation

Sometimes you **cannot** randomise.

```
Question: Did the new MRT station increase nearby HDB prices?

Problem: You cannot randomly assign MRT stations to towns.

Solution: Causal inference methods for observational data
  → Difference-in-Differences (DiD)
  → Propensity Score Matching
  → Instrumental Variables (advanced)
```

---

## Difference-in-Differences (DiD)

Compare the **change** in treated vs control groups, before and after treatment.

```
                Before    After     Change
Treated (MRT):   $400k    $480k    +$80k
Control (no MRT): $390k    $420k    +$30k
                                    ─────
DiD effect:                         +$50k
```

The $50k is the estimated **causal** effect of the MRT station.

---

## DiD Visual

```
Price
  |
  |                    ●──── Treated (after)
  |                 ╱
  |              ╱     ●──── Control (after)
  |           ╱     ╱
  |    ●───╱─╱─ ─ ─ ─  Counterfactual (treated without MRT)
  |     ╱╱
  |  ●──── Control (before)
  |  ●──── Treated (before)
  └──────────────────────→ Time
       Before    After

  DiD = (Treated_after - Treated_before)
      - (Control_after - Control_before)
```

---

## DiD Implementation

```python
import polars as pl
from shared.data_loader import ASCENTDataLoader

loader = ASCENTDataLoader()
df = loader.load("ascent02", "mrt_impact.csv")

# Columns: town, price, period (before/after), treated (0/1)
did = df.group_by("treated", "period").agg(
    pl.col("price").mean().alias("avg_price")
).sort("treated", "period")

# DiD manually
treated_diff = (
    did.filter((pl.col("treated") == 1) & (pl.col("period") == "after"))["avg_price"][0]
    - did.filter((pl.col("treated") == 1) & (pl.col("period") == "before"))["avg_price"][0]
)
control_diff = (
    did.filter((pl.col("treated") == 0) & (pl.col("period") == "after"))["avg_price"][0]
    - did.filter((pl.col("treated") == 0) & (pl.col("period") == "before"))["avg_price"][0]
)
did_effect = treated_diff - control_diff
print(f"DiD causal effect: ${did_effect:,.0f}")
```

---

## DiD Assumptions

The **parallel trends assumption**: without treatment, both groups would have followed the same trend.

```
✅ Valid: Both groups trending similarly before treatment
  Check: plot pre-treatment trends — do they move together?

❌ Invalid: Groups already diverging before treatment
  Fix: Use synthetic control methods or find better controls
```

Always plot pre-treatment trends to assess this assumption.

---

## Propensity Score Matching

When groups differ on observable characteristics, **match** similar units.

```
Problem: Towns with MRT stations differ from towns without
  → Higher income, more amenities, denser population
  → Direct comparison confounds treatment with town characteristics

Solution: Match each treated town with a "similar" control town
  → Similar income, similar amenities, similar density
  → Compare outcomes only among matched pairs
```

---

## Propensity Score: The Matching Key

The propensity score = P(treated | covariates) -- a single number summarising all confounders.

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Estimate propensity scores
features = ["median_income", "population", "num_amenities"]
X = df.select(features).to_numpy()
y = df["treated"].to_numpy()

model = LogisticRegression()
model.fit(X, y)
df = df.with_columns(
    pl.Series("propensity", model.predict_proba(X)[:, 1])
)
```

---

## Nearest-Neighbour Matching

```python
from scipy.spatial import KDTree

treated = df.filter(pl.col("treated") == 1)
control = df.filter(pl.col("treated") == 0)

# Build KD-tree on control propensity scores
tree = KDTree(control["propensity"].to_numpy().reshape(-1, 1))

# Find nearest control match for each treated unit
distances, indices = tree.query(
    treated["propensity"].to_numpy().reshape(-1, 1), k=1
)

# Matched control units
matched_control = control[indices.flatten()]

# ATT (Average Treatment effect on the Treated)
att = treated["price"].mean() - matched_control["price"].mean()
print(f"ATT: ${att:,.0f}")
```

---

## Checking Match Quality

```python
# Before matching: standardised mean differences
for col in features:
    treat_mean = treated[col].mean()
    control_mean = control[col].mean()
    pooled_std = np.sqrt((treated[col].var() + control[col].var()) / 2)
    smd = (treat_mean - control_mean) / pooled_std
    print(f"{col}: SMD = {smd:.3f}")

# After matching: SMD should be < 0.1
# (repeat with matched_control instead of control)
```

Standardised Mean Difference < 0.1 = good balance.

---

## Causal Inference Method Selection

```
Can you randomise?
├─ YES → A/B test (+ CUPED, sequential testing)
│
└─ NO → Observational study
   │
   ├─ Have before/after + control group?
   │  └─ YES → Difference-in-Differences
   │
   ├─ Have rich covariates?
   │  └─ YES → Propensity Score Matching
   │
   └─ Have instrumental variable?
      └─ YES → IV Regression (advanced)
```

---

## Exercise Preview

**Exercise 2.6: Causal Impact of MRT Stations**

You will:

1. Implement a sequential test with O'Brien-Fleming boundaries
2. Apply DiD to estimate the causal effect of MRT openings
3. Use propensity score matching to control for confounders
4. Assess parallel trends and match quality

Scaffolding level: **Moderate+ (~60% code provided)**

---

## Common Pitfalls

| Mistake                                    | Fix                                          |
| ------------------------------------------ | -------------------------------------------- |
| Peeking at p-values without correction     | Use sequential testing boundaries            |
| DiD without checking parallel trends       | Always plot pre-treatment trends             |
| Matching on outcomes instead of covariates | Match on propensity scores only              |
| Ignoring unobserved confounders            | Propensity matching only handles observables |
| Confusing correlation with causation       | Use the method selection guide above         |

---

## Summary

- Sequential testing allows valid early stopping with controlled error
- DiD estimates causal effects by comparing changes across groups
- Propensity score matching creates comparable groups from observational data
- Each method has key assumptions that must be verified
- Randomised experiments remain the gold standard when feasible

---

## Next Lesson

**Lesson 2.7: Feature Engineering**

We will learn:

- Automated feature generation with `FeatureEngineer`
- Feature selection methods (filter, wrapper, embedded)
- The Kailash ML feature engineering workflow
