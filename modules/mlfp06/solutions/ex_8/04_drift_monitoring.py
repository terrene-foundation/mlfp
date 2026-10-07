# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 8.4: Production Drift Monitoring + Agent Test Harness
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Wire kailash-ml DriftMonitor against in-distribution AND shifted
#     production traffic, and measure PSI for each
#   - Read PSI thresholds: < 0.1 no drift, 0.1-0.2 moderate, > 0.2 alert
#   - Debug an agent call by extracting input/output/governance traces
#   - Run a test harness whose tests can FAIL, including deny cases
#   - Apply drift monitoring to a regional e-commerce dispute scenario
#
# PREREQUISITES: Exercises 8.1-8.3
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Rebuild the governed agent stack via build_capstone_stack(engine)
#   2. Configure DriftMonitor with validated QA traffic as the reference
#   3. Debug a single governed call (input -> output -> governance trace)
#   4. Run the automated test harness (5 tests, allow AND deny paths)
#   5. Visualise measured PSI under a traffic shift and apply it
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import polars as pl
from kailash.db.connection import ConnectionManager
from kailash_ml import DriftMonitor

from shared.mlfp06._ollama_bootstrap import DEFAULT_CHAT_MODEL, preflight_ollama
from shared.mlfp06.ex_6 import load_squad_corpus
from shared.mlfp06.ex_8 import (
    OUTPUT_DIR,
    build_capstone_stack,
    compile_capstone_governance,
    handle_qa,
    load_mmlu_eval,
    run_async,
)

# Tasks 3-4 make real LLM calls; fail loudly now if Ollama is not running.
preflight_ollama(required_models=[DEFAULT_CHAT_MODEL])

# ════════════════════════════════════════════════════════════════════════
# THEORY — Drift and Observability
# ════════════════════════════════════════════════════════════════════════
# A model ships. On day 1, its prediction distribution matches the
# training set. On day 30, a product launch, a new customer cohort,
# or a simple calendar effect silently shifts the input distribution.
# The model still returns answers — just wrong ones. Drift monitoring
# turns "the dashboard looks fine" into "PSI=0.35, ALERT, retrain".
#
# Population Stability Index (PSI) compares two histograms:
#   PSI < 0.10  no significant drift
#   PSI 0.10–0.20  moderate drift (investigate)
#   PSI > 0.20  significant drift (retrain)
#
# Drift monitoring is the *proof* that the governance envelope is
# still aligned to the data the model was trained on. Without it,
# governance silently degrades.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Rebuild the governed stack
# ════════════════════════════════════════════════════════════════════════

eval_data = load_mmlu_eval(n_rows=100)

governance_engine, _loaded = compile_capstone_governance()
agents_by_role, tiers = build_capstone_stack(governance_engine)
tier_by_role = {t.role: t for t in tiers}

print("Governed stack rebuilt:")
for tier in tiers:
    print(
        f"  {tier.role:6s} -> budget=${tier.budget_usd:>5.1f}  "
        f"clearance={tier.clearance}"
    )

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert len(agents_by_role) == 3, "Task 1: governed stack should rebuild"
print("✓ Checkpoint 1 passed — governed stack rebuilt\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Configure DriftMonitor and measure PSI on real traffic
# ════════════════════════════════════════════════════════════════════════
#
# kailash-ml's DriftMonitor persists reference distributions and drift
# reports through a `ConnectionManager` (a local SQLite file here).
#
# Feature: question length in characters — numeric, cheap, and it moves
# when the population of users or question types changes.
#   reference       = SQuAD 2.0 questions 0-149   (the QA traffic the
#                     model was validated on: short factual questions)
#   in-distribution = SQuAD 2.0 questions 150-299 (same population, unseen)
#   shifted         = the 100 MMLU exam questions (a different population:
#                     long multiple-choice questions with four options)
# Comparing both samples against the same reference is the test that the
# monitor can tell "same" from "shifted".

_DRIFT_DB_PATH = os.path.abspath("data/mlfp06/ex8_drift.db")
os.makedirs(os.path.dirname(_DRIFT_DB_PATH), exist_ok=True)
# Start from a clean database (stale -wal/-shm files cause I/O errors).
for _sfx in ("", "-wal", "-shm"):
    _p = _DRIFT_DB_PATH + _sfx
    if os.path.exists(_p):
        os.remove(_p)


def question_length(df: pl.DataFrame, col: str) -> pl.DataFrame:
    return df.select(pl.col(col).str.len_chars().cast(pl.Float64).alias("question_length"))


squad = load_squad_corpus()
reference_df = question_length(squad.head(150), "question")
in_dist_df = question_length(squad.tail(150), "question")
shifted_df = question_length(eval_data, "instruction")

# A traffic shift: the share of exam-style questions in a window of 150
# requests grows from 0% to 60%. The mix is constructed; every PSI below
# is MEASURED by DriftMonitor on the real rows in each window.
WINDOW = 150
SHIFT_FRACTIONS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6]
windows = [
    pl.concat(
        [in_dist_df.head(WINDOW - int(WINDOW * f)), shifted_df.head(int(WINDOW * f))]
    )
    for f in SHIFT_FRACTIONS
]


async def measure_drift() -> tuple[dict, list[float]]:
    conn = ConnectionManager(f"sqlite:///{_DRIFT_DB_PATH}")
    await conn.initialize()
    try:
        monitor = DriftMonitor(conn, tenant_id="mlfp_demo", psi_threshold=0.2)
        await monitor.set_reference_data(
            "capstone_qa_model", reference_df, ["question_length"]
        )
        reports = {}
        for name, prod in [("in_distribution", in_dist_df), ("shifted", shifted_df)]:
            rep = await monitor.check_drift("capstone_qa_model", prod)
            reports[name] = rep
            print(
                f"  {name:<16} n={prod.height}  PSI={rep.feature_results[0].psi:.3f}  "
                f"severity={rep.overall_severity}  drift={rep.overall_drift_detected}"
            )
        window_psi = []
        for frac, win in zip(SHIFT_FRACTIONS, windows):
            rep = await monitor.check_drift("capstone_qa_model", win)
            window_psi.append(rep.feature_results[0].psi)
        return reports, window_psi
    finally:
        # Close inside the loop so the pool finaliser does not hang at exit.
        await conn.close()


print("DriftMonitor reports (reference = SQuAD questions 0-149):")
drift_reports, window_psi = run_async(measure_drift())

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
psi_in = drift_reports["in_distribution"].feature_results[0].psi
psi_shift = drift_reports["shifted"].feature_results[0].psi
assert drift_reports["shifted"].overall_drift_detected, "Task 2: shift must alert"
assert not drift_reports["in_distribution"].overall_drift_detected, (
    "Task 2: same-population traffic must not alert"
)
assert psi_shift > psi_in, "Task 2: shifted traffic must score higher PSI"
print(
    f"✓ Checkpoint 2 passed — PSI {psi_in:.3f} (same population) vs "
    f"{psi_shift:.3f} (shifted)\n"
)
# INTERPRETATION: with 150 rows per sample, same-population PSI is still
# not 0 (sampling noise can put it in the 'moderate' band). Read PSI
# against a baseline measured on known-good traffic, not against 0 — and
# use samples of at least a few hundred rows in production.


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Debug a single governed call
# ════════════════════════════════════════════════════════════════════════


async def debug_agent_call() -> dict:
    question = eval_data["instruction"][0]
    tier = tier_by_role["qa"]
    print(f"\nDebugging call for: {question[:80]}...")

    print("\n  INPUT TRACE:")
    print(f"    Question:  {question[:100]}...")
    print(f"    Role:      {tier.role} ({tier.address})")
    print(f"    Budget:    ${tier.budget_usd:.2f}")
    print(f"    Clearance: {tier.clearance}")

    result = await handle_qa(
        question, role="qa", agents_by_role=agents_by_role, engine=governance_engine
    )

    print("\n  OUTPUT TRACE:")
    if result["blocked"]:
        print(f"    Status:    BLOCKED ({result['error']})")
    else:
        print(f"    Answer:    {result['answer'][:150]}...")
        print(f"    Confidence (self-reported): {result['confidence']}")
        print(f"    Sources:   {result['sources'][:3]}")
        print(f"    Latency:   {result['latency_ms']:.0f} ms")

    print("\n  GOVERNANCE TRACE:")
    print(f"    Role:      {result['role']}")
    print(f"    Verdict:   {result['verdict']}")
    print(f"    Blocked:   {result['blocked']}")
    return result


debug_result = run_async(debug_agent_call())

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert debug_result["governed"] is True
assert debug_result["verdict"] == "served", "Task 3: a normal qa call is served"
assert debug_result["answer"].strip(), "Task 3: the LLM must return an answer"
print("✓ Checkpoint 3 passed — debug trace produced\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Automated test harness (every test can fail)
# ════════════════════════════════════════════════════════════════════════
# Each test states the EXPECTED governance outcome and compares it with
# what handle_qa actually returned. Two of the five are deny cases.


async def run_test_harness() -> pl.DataFrame:
    async def ask(question: str, role: str, action: str = "generate_answer") -> dict:
        return await handle_qa(
            question,
            role=role,
            agents_by_role=agents_by_role,
            engine=governance_engine,
            action=action,
        )

    cases = [
        ("Normal QA served", "qa", "generate_answer", "served"),
        ("Unknown role refused", "invalid_role", "generate_answer", "unknown_role"),
        ("Admin may update_model", "admin", "update_model", "served"),
        ("QA may NOT update_model", "qa", "update_model", "blocked"),
        ("QA may NOT read audit log", "qa", "access_audit_log", "blocked"),
    ]
    results: list[dict] = []
    for name, role, action, expected in cases:
        r = await ask(eval_data["instruction"][1], role, action)
        results.append(
            {
                "test": name,
                "expected": expected,
                "actual": r["verdict"],
                "passed": r["verdict"] == expected,
                "detail": r.get("error", "") or r.get("answer", "")[:40],
            }
        )

    df = pl.DataFrame(results)
    print("\n--- Test Results ---")
    print(df.select("test", "expected", "actual", "passed"))
    print(f"\n  Result: {int(df['passed'].sum())}/{df.height} passed")
    return df


test_df = run_async(run_test_harness())
test_df.write_parquet(OUTPUT_DIR / "test_harness_results.parquet")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert test_df.height >= 5, "Task 4: at least 5 tests should run"
assert test_df["passed"].all(), (
    f"Task 4: failing tests: {test_df.filter(~pl.col('passed'))['test'].to_list()}"
)
assert (test_df["expected"] != "served").sum() >= 2, "Task 4: include deny cases"
print("✓ Checkpoint 4 passed — automated test harness complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise and Apply: Regional E-commerce Dispute Handling
# ════════════════════════════════════════════════════════════════════════


def psi_zone(psi: float) -> str:
    return "Safe" if psi < 0.10 else ("Investigate" if psi <= 0.20 else "Alert")


psi_dashboard = pl.DataFrame(
    {
        "window": [f"{int(f * 100)}% shifted" for f in SHIFT_FRACTIONS],
        "PSI": window_psi,
        "zone": [psi_zone(p) for p in window_psi],
    }
)
psi_dashboard.write_parquet(OUTPUT_DIR / "psi_dashboard.parquet")
print("\nPSI by traffic window (measured):")
print(psi_dashboard)
first_alert = psi_dashboard.filter(pl.col("zone") == "Alert")["window"].to_list()
print(f"  First window in the Alert zone: {first_alert[0] if first_alert else 'none'}")

# SCENARIO: A regional e-commerce marketplace uses the governed QA agent
# in a merchant-dispute bot. During its biggest sale events the mix of
# questions changes sharply — new merchants, new fraud patterns, much
# longer, multi-part questions — and the model's training distribution no longer
# matches what it sees. The window table above is exactly that kind of
# shift: PSI rises as the share of "new population" questions grows.
#
# BUSINESS IMPACT (illustrative figures): if one 72-hour window of
# degraded automated dispute handling costs ~S$250,000 in merchant
# goodwill, while routing those disputes to human agents for the window
# costs ~S$5,000, a PSI > 0.2 alert that triggers the re-route is worth
# ~S$245,000 per event.

print("\n" + "=" * 70)
print("  APPLY — Sale-Event Dispute Handling")
print("=" * 70)
print(
    f"""
  Same-population PSI:  {psi_in:.3f} ({psi_zone(psi_in)})
  Fully shifted PSI:    {psi_shift:.3f} ({psi_zone(psi_shift)})
  Auto action on Alert: route to human agents + queue a retrain job

  Illustrative economics per alert window:
    goodwill protected ~S$250,000 vs human surge cost ~S$5,000
"""
)


# ════════════════════════════════════════════════════════════════════════
# VISUALISATION — Measured PSI as the traffic shifts
# ════════════════════════════════════════════════════════════════════════

threshold = 0.2
x = [int(f * 100) for f in SHIFT_FRACTIONS]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, window_psi, "o-", color="#1976D2", linewidth=2, label="Measured PSI")
ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1.5, label=f"Alert ({threshold})")
top = max(max(window_psi) * 1.1, 0.4)
ax.axhspan(0, 0.1, alpha=0.1, color="green", label="Safe")
ax.axhspan(0.1, 0.2, alpha=0.1, color="orange", label="Investigate")
ax.axhspan(0.2, top, alpha=0.1, color="red", label="Alert")
ax.set_xlabel("Share of exam-style (MMLU) questions in the window (%)")
ax.set_ylabel("PSI (question length)")
ax.set_yscale("symlog", linthresh=0.5)
ax.set_title("Drift Monitoring: Measured PSI vs Traffic Shift")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "04_drift_timeline.png", dpi=150)
plt.close(fig)
print(f"\nSaved: {OUTPUT_DIR / '04_drift_timeline.png'}")


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — Governance lens over the qa tier's audit trail
# ══════════════════════════════════════════════════════════════════
from shared.mlfp06.diagnostics import LLMObservatory

obs = LLMObservatory(governance=agents_by_role["qa"].audit, run_id="ex_8_4_drift")
print("\n── LLM Observatory: qa-tier audit snapshot ──")
print(obs.governance.audit_snapshot(last_n=20).select("action", "verdict"))
print(f"  qa audit chain verifies: {agents_by_role['qa'].audit.verify_chain()}")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  [x] Wired DriftMonitor with validated QA traffic as the reference
  [x] Measured PSI on same-population vs shifted traffic
  [x] Debugged a governed agent call end-to-end
  [x] Ran a test harness whose 5 tests include deny paths
  [x] Applied drift monitoring to a regional e-commerce scenario

  KEY INSIGHT: Drift monitoring and testing are the only things
  standing between "the model shipped" and "the model is still
  correct in production". Governance says what the envelope IS;
  drift + tests prove the envelope still MATCHES reality.

  Next: 05_compliance_audit.py closes the loop with a regulatory
  audit report mapping evidence from this run to EU AI Act, AI Verify,
  and MAS TRM requirements.
"""
)
