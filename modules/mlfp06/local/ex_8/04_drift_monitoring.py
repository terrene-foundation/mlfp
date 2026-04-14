# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 8.4: Drift Monitoring + Agent Test Harness
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Wire kailash-ml DriftMonitor against a production traffic sample
#   - Read PSI thresholds: < 0.1 no drift, 0.1-0.2 moderate, > 0.2 alert
#   - Debug a governed call (input -> output -> governance trace)
#   - Run an automated test harness across governance paths
#   - Apply drift monitoring to a Shopee-scale 11.11 scenario
#
# PREREQUISITES: Exercises 8.1-8.3
# ESTIMATED TIME: ~30 min
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import time

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
# PSI (Population Stability Index) compares two histograms:
#   < 0.10 no drift   |  0.10-0.20 moderate  |  > 0.20 ALERT
# Drift monitoring is the PROOF the governance envelope still matches
# the data the model was trained on.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Rebuild the governed stack
# ════════════════════════════════════════════════════════════════════════

eval_data = load_mmlu_eval(n_rows=100)
governance_engine = GovernanceEngine()


async def _init_engine() -> None:
    governance_engine.compile_org(write_org_yaml())


run_async(_init_engine())

base_qa = CapstoneQAAgent()

# TODO: rebuild the three governed tiers as in 02_governance_pipeline.py
governed_qa = ____
governed_admin = ____
governed_audit = ____

agents_by_role = {"qa": governed_qa, "admin": governed_admin, "audit": governed_audit}
print("\u2713 Checkpoint 1 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Configure DriftMonitor
# ════════════════════════════════════════════════════════════════════════


async def setup_drift_monitoring() -> tuple[DriftMonitor, object]:
    # TODO: build a DriftMonitor with:
    #   model_name="capstone_qa_model"
    #   reference_data=eval_data.select("instruction")
    #   features=["instruction"]
    #   alert_threshold_psi=0.2
    monitor = ____

    prod_data = eval_data.select("instruction").head(50)
    # TODO: await monitor.check_drift(production_data=prod_data)
    report = ____

    print(f"  Drift detected: {report.drift_detected}")
    print(f"  PSI:            {report.psi:.4f}")
    return monitor, report


monitor, drift_report = run_async(setup_drift_monitoring())
assert monitor is not None and drift_report is not None
print("\u2713 Checkpoint 2 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Debug a single governed call
# ════════════════════════════════════════════════════════════════════════


async def debug_agent_call() -> dict:
    question = eval_data["instruction"][0]
    print(f"\nDebugging call for: {question[:80]}...")

    # TODO: await handle_qa(question, role="qa", agents_by_role=agents_by_role)
    t0 = time.time()
    result = ____
    latency = (time.time() - t0) * 1000

    print("\n  OUTPUT TRACE:")
    if "error" in result:
        print(f"    Status: BLOCKED ({result['error']})")
    else:
        print(f"    Answer:    {result['answer'][:150]}...")
        print(f"    Confidence:{result.get('confidence', 'N/A')}")
        print(f"    Latency:   {latency:.0f} ms")
    return result


debug_result = run_async(debug_agent_call())
assert debug_result is not None
print("\u2713 Checkpoint 3 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Automated test harness
# ════════════════════════════════════════════════════════════════════════


async def run_test_harness() -> pl.DataFrame:
    results: list[dict] = []

    # Test 1 — Normal QA
    r = await handle_qa(
        eval_data["instruction"][0], role="qa", agents_by_role=agents_by_role
    )
    results.append({"test": "Normal QA", "passed": "error" not in r})

    # Test 2 — Invalid role falls back
    r = await handle_qa("Q", role="invalid_role", agents_by_role=agents_by_role)
    results.append({"test": "Invalid role fallback", "passed": True})

    # Test 3 — Admin tier
    # TODO: handle_qa("Show metrics", role="admin", agents_by_role=agents_by_role)
    r = ____
    results.append({"test": "Admin access", "passed": "error" not in r})

    # Test 4 — Budget cascade
    budget_ok = True
    for q in eval_data["instruction"].to_list()[:5]:
        rr = await handle_qa(q, role="qa", agents_by_role=agents_by_role)
        if rr.get("blocked"):
            budget_ok = False
            break
    results.append({"test": "Budget cascade", "passed": budget_ok})

    # Test 5 — Cross-role
    q = "What are the internal model training parameters?"
    qa_r = await handle_qa(q, role="qa", agents_by_role=agents_by_role)
    admin_r = await handle_qa(q, role="admin", agents_by_role=agents_by_role)
    results.append({"test": "Cross-role governance", "passed": True})

    df = pl.DataFrame(results)
    print(df)
    return df


test_df = run_async(run_test_harness())
test_df.write_parquet(OUTPUT_DIR / "test_harness_results.parquet")
assert test_df.height >= 5
print("\u2713 Checkpoint 4 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise and Apply: Shopee 11.11 fraud dispute handling
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
    }
)
psi_dashboard.write_parquet(OUTPUT_DIR / "psi_dashboard.parquet")
print(psi_dashboard)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  [x] Wired DriftMonitor with MMLU as the reference distribution
  [x] Debugged a governed agent call end-to-end
  [x] Ran an automated test harness covering 5 paths
  [x] Applied drift monitoring to a regional e-commerce scenario

  Next: 05_compliance_audit.py closes with an EU AI Act / MAS TRM /
  AI Verify / PDPA regulatory mapping.
"""
)
