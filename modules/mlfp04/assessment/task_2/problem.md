# MLFP04 — Task 2: Reduction, Embeddings and Anomaly Screening

**Weight**: 20 marks · **Outcomes**: 4.3 (PCA, variance explained, loadings, non-linear embeddings), 4.4 (anomaly detection, combining detectors, what each detector cannot see)
**Data**: `mlfp02/sg_credit_scoring.parquet` (real Singapore credit applications), loaded with `shared.MLFPDataLoader`. The grader draws its own secret samples and builds its own screening batches.

## Scenario

A lender's application-review team works with 20 numeric fields per
application (listed in `starter.py`): ages and tenures in years, balances and
loan amounts in S$, counts of credit lines and late payments, ratios such as
credit utilisation and debt-to-income. Every frame you receive has an
`application_id` column plus these 20 fields, **in no fixed column order**.

1. **How many dimensions does an application really have?** The team wants
   the share of variance carried by each principal component and the smallest
   number of components that keeps at least 90% of it. A component must not
   be dominated by a field merely because of the unit it is recorded in. They
   also want the direction of the first component (its loading on each field)
   so they can name it.
2. **A map for reviewers.** Reviewers want a 2-D picture of a batch in which
   applications that are similar across all 20 fields sit next to each other.
3. **Screening.** Before manual review, every application in a batch gets a
   suspicion score. Fraud and data-entry problems in this book come in three
   kinds, and the team needs all three ranked near the top:
   - a single field keyed in absurdly high (fat-finger entry, inflated balance);
   - a "synthetic identity": every field is copied from a different real
     applicant, so each value looks normal but the combination does not;
   - an application ring: a tight group of near-identical applications that
     are unusual as a group.

   There are no fraud labels to train on.

## What to submit

`starter.py` with three functions (signatures fixed). Each receives a polars
DataFrame of applications (`application_id` + the 20 fields).

| Function                          | Returns                                                                                                                                                                                                                         |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `component_profile(applications)` | dict: `explained_variance_ratio` (list, one per component, largest first, all 20), `n_components_90` (int), `pc1_loadings` (dict field → loading of the first component, a unit-length direction; the overall sign is free) |
| `embed_2d(applications)`          | numpy array (n × 2), one row per application in row order                                                                                                                                                                       |
| `anomaly_scores(applications)`    | list of floats, one per row in row order; higher = more suspicious                                                                                                                                                              |

## Acceptance criteria

- `component_profile` is checked on two secret samples (1,500–4,000
  applications, shuffled columns) against the grader's own principal
  components: every variance share within 0.002, the 90% component count
  exact, every loading within 0.02.
- `embed_2d` is checked on a secret sample of 1,000 applications: the
  trustworthiness (10 neighbours) of your map with respect to the 20 fields
  on a common scale must be at least 0.90.
- `anomaly_scores` is checked on two secret batches of about 3,000 real
  applications with 30 + 30 + 20 injected anomalies of the three kinds above.
  ROC-AUC against the injection labels must be at least 0.85 for **each**
  kind and at least 0.90 overall, on both batches.
- Structural checks (framework use) earn marks only when the outcome checks
  of the same part pass.

## Rules

- Reduction and embedding run through kailash-ml `DimReductionEngine`; at
  least one detector runs through kailash-ml `AnomalyDetectionEngine`. You
  may add your own numpy scores.
- Polars for data handling (no pandas). Your functions must work from their
  arguments alone: refer to fields by name, never by position.
