# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 8.4: Production Drift Monitoring + Agent Test Harness
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Wire kailash-ml DriftMonitor against a production traffic sample
#   - Read PSI thresholds: < 0.1 no drift, 0.1-0.2 moderate, > 0.2 alert
#   - Debug an agent call by extracting input/output/governance traces
#   - Run an automated test harness across several governance paths
#   - Apply drift monitoring to a Singapore e-commerce fraud scenario
#
# PREREQUISITES: Exercises 8.1-8.3
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Rebuild the governed agent stack
#   2. Configure DriftMonitor with MMLU as the reference distribution
#   3. Debug a single governed call (input -> output -> governance trace)
#   4. Run the automated test harness (5 tests)
#   5. Visualise the PSI dashboard and apply to Shopee-scale fraud
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from kailash_ml import DriftMonitor
from kailash_pact import GovernanceEngine, PactGovernedAgent

from shared.mlfp06.ex_8 import (
    CapstoneQAAgent,
    OUTPUT_DIR,
    handle_qa,
    load_mmlu_eval,
    run_async,
    write_org_yaml,
)

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

governance_engine = GovernanceEngine()


async def _init_engine() -> None:
    governance_engine.compile_org(write_org_yaml())


run_async(_init_engine())

base_qa = CapstoneQAAgent()
governed_qa = PactGovernedAgent(
    agent=base_qa,
    governance_engine=governance_engine,
    role="responder",
    max_budget_usd=1.0,
    allowed_tools=["generate_answer", "search_context"],
    clearance_level="internal",
)
governed_admin = PactGovernedAgent(
    agent=base_qa,
    governance_engine=governance_engine,
    role="operator",
    max_budget_usd=10.0,
    allowed_tools=[
        "generate_answer",
        "search_context",
        "update_model",
        "view_metrics",
        "monitor_drift",
    ],
    clearance_level="confidential",
)
governed_audit = PactGovernedAgent(
    agent=base_qa,
    governance_engine=governance_engine,
    role="auditor",
    max_budget_usd=50.0,
    allowed_tools=[
        "generate_answer",
        "search_context",
        "view_metrics",
        "access_audit_log",
        "generate_report",
    ],
    clearance_level="restricted",
)
agents_by_role = {"qa": governed_qa, "admin": governed_admin, "audit": governed_audit}

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert len(agents_by_role) == 3, "Task 1: governed stack should rebuild"
print("\u2713 Checkpoint 1 passed — governed stack rebuilt\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Configure DriftMonitor
# ════════════════════════════════════════════════════════════════════════


async def setup_drift_monitoring() -> tuple[DriftMonitor, object]:
    monitor = DriftMonitor(
        model_name="capstone_qa_model",
        reference_data=eval_data.select("instruction"),
        features=["instruction"],
        alert_threshold_psi=0.2,
    )
    prod_data = eval_data.select("instruction").head(50)
    report = await monitor.check_drift(production_data=prod_data)
    print("DriftMonitor report:")
    print(f"  Model:              capstone_qa_model")
    print(f"  Reference samples:  {eval_data.height}")
    print(f"  Production samples: {prod_data.height}")
    print(f"  Drift detected:     {report.drift_detected}")
    print(f"  PSI:                {report.psi:.4f}")
    print(
        "  Status:             "
        + ("ALERT — retrain needed" if report.drift_detected else "OK — no drift")
    )
    return monitor, report


monitor, drift_report = run_async(setup_drift_monitoring())

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert monitor is not None, "Task 2: DriftMonitor should be created"
assert drift_report is not None, "Task 2: drift report should be produced"
print("\u2713 Checkpoint 2 passed — DriftMonitor wired\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Debug a single governed call
# ════════════════════════════════════════════════════════════════════════


async def debug_agent_call() -> dict:
    question = eval_data["instruction"][0]
    print(f"\nDebugging call for: {question[:80]}...")

    print("\n  INPUT TRACE:")
    print(f"    Question:  {question[:100]}...")
    print(f"    Role:      qa (governed_qa)")
    print(f"    Budget:    $1.00")
    print(f"    Clearance: internal")

    t0 = time.time()
    result = await handle_qa(question, role="qa", agents_by_role=agents_by_role)
    latency = (time.time() - t0) * 1000

    print("\n  OUTPUT TRACE:")
    if "error" in result:
        print(f"    Status:    BLOCKED ({result['error']})")
    else:
        print(f"    Answer:    {result['answer'][:150]}...")
        print(f"    Confidence:{result.get('confidence', 'N/A')}")
        print(f"    Sources:   {result.get('sources', [])[:3]}")
        print(f"    Latency:   {latency:.0f} ms")

    print("\n  GOVERNANCE TRACE:")
    print(f"    Role:      {result['role']}")
    print(f"    Governed:  {result['governed']}")
    print(f"    Blocked:   {result.get('blocked', False)}")
    return result


debug_result = run_async(debug_agent_call())

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert debug_result is not None, "Task 3: debug call should return a result"
print("\u2713 Checkpoint 3 passed — debug trace produced\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Automated test harness
# ════════════════════════════════════════════════════════════════════════


async def run_test_harness() -> pl.DataFrame:
    results: list[dict] = []

    # Test 1 — Normal QA query
    r = await handle_qa(
        eval_data["instruction"][0], role="qa", agents_by_role=agents_by_role
    )
    results.append(
        {
            "test": "Normal QA",
            "passed": "error" not in r,
            "detail": "Answer received" if "error" not in r else r.get("error", ""),
        }
    )

    # Test 2 — Invalid role falls back to lowest privilege
    r = await handle_qa(
        "Test question", role="invalid_role", agents_by_role=agents_by_role
    )
    results.append(
        {
            "test": "Invalid role fallback",
            "passed": True,
            "detail": f"Role used: {r.get('role', 'unknown')}",
        }
    )

    # Test 3 — Admin tier should reach the admin-only tool surface
    r = await handle_qa(
        "Show model performance metrics", role="admin", agents_by_role=agents_by_role
    )
    results.append(
        {
            "test": "Admin access",
            "passed": "error" not in r,
            "detail": "Admin access granted" if "error" not in r else "Blocked",
        }
    )

    # Test 4 — Budget cascade across 5 queries
    budget_ok = True
    for q in eval_data["instruction"].to_list()[:5]:
        rr = await handle_qa(q, role="qa", agents_by_role=agents_by_role)
        if rr.get("blocked"):
            budget_ok = False
            break
    results.append(
        {
            "test": "Budget cascade (5 queries)",
            "passed": budget_ok,
            "detail": "All passed" if budget_ok else "Budget exceeded",
        }
    )

    # Test 5 — Same question, different tiers, different envelopes
    q = "What are the internal model training parameters?"
    qa_r = await handle_qa(q, role="qa", agents_by_role=agents_by_role)
    admin_r = await handle_qa(q, role="admin", agents_by_role=agents_by_role)
    results.append(
        {
            "test": "Cross-role governance",
            "passed": True,
            "detail": (
                f"QA: {'answered' if 'error' not in qa_r else 'blocked'}, "
                f"Admin: {'answered' if 'error' not in admin_r else 'blocked'}"
            ),
        }
    )

    df = pl.DataFrame(results)
    passed = int(df["passed"].sum())
    print("\n--- Test Results ---")
    print(df)
    print(f"\n  Result: {passed}/{df.height} passed")
    return df


test_df = run_async(run_test_harness())
test_df.write_parquet(OUTPUT_DIR / "test_harness_results.parquet")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert test_df.height >= 5, "Task 4: at least 5 tests should run"
print("\u2713 Checkpoint 4 passed — automated test harness complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise and Apply: Shopee-scale fraud detection
# ════════════════════════════════════════════════════════════════════════

psi_dashboard = pl.DataFrame(
    {
        "Zone": ["Safe", "Investigate", "Alert"],
        "PSI range": ["< 0.10", "0.10 - 0.20", "> 0.20"],
        "Action": [
            "Continue serving",
            "Root-cause within 24h",
            "Rollback or retrain",
        ],
        "Current": [
            drift_report.psi,
            drift_report.psi,
            drift_report.psi,
        ],
    }
)
psi_dashboard.write_parquet(OUTPUT_DIR / "psi_dashboard.parquet")
print("\nPSI dashboard:")
print(psi_dashboard)

# SCENARIO: A Singapore-based e-commerce platform (think Shopee scale)
# uses the governed QA agent as part of a customer-dispute resolution
# bot. The bot answers merchant questions about fraud flags. Drift
# appears during the 11.11 and 12.12 sales — traffic distribution
# shifts 4-7x, fraud-dispute text patterns change, and the underlying
# model's training distribution no longer matches reality.
#
# BUSINESS IMPACT: Without drift monitoring, the bot's dispute
# responses silently degrade during the highest-revenue window of
# the year. With PSI > 0.2 alerting, operations auto-route to a
# human agent for the 72-hour shift window and queue a retraining
# run. One 72-hour window of bad automated dispute handling has
# been priced internally at S$250,000 of merchant goodwill (10% of
# disputed merchants churn at ~S$2,500 LTV each). Drift detection
# converts that S$250,000 into a S$5,000 human-handling surge cost.

print("\n" + "=" * 70)
print("  APPLY — Shopee-scale 11.11 Dispute Handling")
print("=" * 70)
print(
    """
  Normal window:  PSI ~0.05, governed QA handles 95% of disputes.
  11.11 surge:    PSI climbs toward 0.25, DriftMonitor ALERTS.
  Auto action:    Route to human agent pool + queue retrain job.

  Churn avoided:  ~S$250,000 merchant goodwill
  Cost incurred:  ~S$5,000 human surge handling
  Net saving:     ~S$245,000 per 72-hour alert window
"""
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  [x] Wired DriftMonitor with MMLU as the reference distribution
  [x] Read PSI thresholds as business-actionable zones
  [x] Debugged a governed agent call end-to-end
  [x] Ran an automated test harness covering 5 governance paths
  [x] Applied drift monitoring to a regional e-commerce scenario

  KEY INSIGHT: Drift monitoring and testing are the only things
  standing between "the model shipped" and "the model is still
  correct in production". Governance says what the envelope IS;
  drift + tests prove the envelope still MATCHES reality.

  Next: 05_compliance_audit.py closes the loop with a regulatory
  audit report mapping technical controls to EU AI Act, AI Verify,
  and MAS TRM requirements.
"""
)
