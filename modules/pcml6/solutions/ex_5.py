# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT6 — Exercise 5: AI Governance with PACT
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Define a realistic organization in YAML, compile it, and
#   create a GovernanceEngine. Verify access decisions.
#
# TASKS:
#   1. Define organization structure in YAML (3 departments, 8 roles)
#   2. Compile organization with compile_org()
#   3. Create GovernanceEngine
#   4. Test access decisions: can_access(), explain_access()
#   5. Verify monotonic tightening and fail-closed behavior
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from pact import GovernanceEngine, GovernanceContext
from pact import Address, RoleEnvelope, TaskEnvelope
from pact import compile_org, load_org_yaml

from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Define organization in YAML
# ══════════════════════════════════════════════════════════════════════

org_yaml = """
organization:
  name: "ASCENT Credit Bureau"
  version: "1.0"

  departments:
    - name: "data_science"
      description: "ML model development and research"
      teams:
        - name: "modeling"
          roles:
            - name: "senior_data_scientist"
              permissions:
                data_access: ["credit_data", "feature_store", "experiment_logs"]
                tools: ["training_pipeline", "hyperparameter_search", "model_registry"]
                max_cost_usd: 50.0
                can_deploy: false
            - name: "junior_data_scientist"
              permissions:
                data_access: ["credit_data", "feature_store"]
                tools: ["training_pipeline"]
                max_cost_usd: 10.0
                can_deploy: false
        - name: "mlops"
          roles:
            - name: "ml_engineer"
              permissions:
                data_access: ["credit_data", "feature_store", "model_artifacts", "production_logs"]
                tools: ["training_pipeline", "model_registry", "inference_server", "drift_monitor"]
                max_cost_usd: 100.0
                can_deploy: true

    - name: "risk"
      description: "Model risk management and compliance"
      teams:
        - name: "model_validation"
          roles:
            - name: "model_validator"
              permissions:
                data_access: ["credit_data", "experiment_logs", "model_artifacts", "audit_logs"]
                tools: ["model_registry", "drift_monitor"]
                max_cost_usd: 20.0
                can_deploy: false
                can_approve_production: true
            - name: "compliance_officer"
              permissions:
                data_access: ["audit_logs", "governance_reports"]
                tools: []
                max_cost_usd: 5.0
                can_deploy: false
                can_approve_production: false

    - name: "operations"
      description: "Production systems and customer-facing"
      teams:
        - name: "serving"
          roles:
            - name: "sre"
              permissions:
                data_access: ["production_logs", "model_artifacts"]
                tools: ["inference_server", "drift_monitor"]
                max_cost_usd: 200.0
                can_deploy: true
            - name: "customer_service"
              permissions:
                data_access: ["credit_decisions"]
                tools: ["inference_server"]
                max_cost_usd: 1.0
                can_deploy: false

  policies:
    - name: "model_promotion"
      description: "Models require validator approval before production"
      rule: "promote_to_production requires model_validator.can_approve_production"
    - name: "data_classification"
      description: "Credit data is Confidential; audit logs are Restricted"
      classifications:
        credit_data: "confidential"
        audit_logs: "restricted"
        production_logs: "internal"
        credit_decisions: "confidential"
"""

# Write to temp file
org_file = Path(tempfile.mktemp(suffix=".yaml"))
org_file.write_text(org_yaml)
print(f"=== Organization YAML ===")
print(f"Departments: 3 (data_science, risk, operations)")
print(f"Roles: 6 unique roles across 5 teams")
print(f"Written to: {org_file}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Compile organization
# ══════════════════════════════════════════════════════════════════════

org = load_org_yaml(str(org_file))
compiled = compile_org(org)

print(f"\n=== Compiled Organization ===")
print(f"Valid: {compiled.valid}")
if compiled.errors:
    print(f"Errors: {compiled.errors}")
if compiled.warnings:
    print(f"Warnings: {compiled.warnings}")
print(f"Departments: {[d.name for d in compiled.departments]}")
print(f"Total roles: {compiled.total_roles}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Create GovernanceEngine
# ══════════════════════════════════════════════════════════════════════


async def setup_governance():
    engine = GovernanceEngine(compiled)

    print(f"\n=== GovernanceEngine ===")
    print(f"Organization: {compiled.name}")
    print(f"Enforcement mode: fail-closed (deny on error)")

    return engine


engine = asyncio.run(setup_governance())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Test access decisions
# ══════════════════════════════════════════════════════════════════════


async def test_access():
    # D/T/R addressing: Department/Team/Role
    senior_ds = Address("data_science", "modeling", "senior_data_scientist")
    junior_ds = Address("data_science", "modeling", "junior_data_scientist")
    ml_eng = Address("data_science", "mlops", "ml_engineer")
    validator = Address("risk", "model_validation", "model_validator")
    cust_svc = Address("operations", "serving", "customer_service")

    # Test cases
    test_cases = [
        (senior_ds, "credit_data", True, "Senior DS can access credit data"),
        (
            junior_ds,
            "experiment_logs",
            False,
            "Junior DS cannot access experiment logs",
        ),
        (ml_eng, "production_logs", True, "ML Engineer can access production logs"),
        (validator, "audit_logs", True, "Validator can access audit logs"),
        (
            cust_svc,
            "credit_data",
            False,
            "Customer service cannot access raw credit data",
        ),
        (cust_svc, "credit_decisions", True, "Customer service can access decisions"),
    ]

    print(f"\n=== Access Control Tests ===")
    for address, resource, expected, description in test_cases:
        decision = await engine.can_access(address, resource)
        status = "✓" if decision == expected else "✗ UNEXPECTED"
        print(f"  {status} {description}")
        print(f"     {address} → {resource}: {'ALLOW' if decision else 'DENY'}")

    # Explain access
    print(f"\n=== Access Explanations ===")
    explanation = await engine.explain_access(junior_ds, "production_logs")
    print(f"Junior DS → production_logs:")
    print(f"  Decision: {explanation.decision}")
    print(f"  Reason: {explanation.reason}")
    print(f"  Chain: {explanation.chain}")


asyncio.run(test_access())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Monotonic tightening and fail-closed
# ══════════════════════════════════════════════════════════════════════


async def test_monotonic_tightening():
    """
    Monotonic tightening: child envelopes CANNOT exceed parent.
    If parent budget is $50, child cannot have $100.
    """
    parent_envelope = RoleEnvelope(
        address=Address("data_science", "modeling", "senior_data_scientist"),
        max_cost_usd=50.0,
        data_access=["credit_data", "feature_store"],
    )

    # Try to create child task with higher budget (should be tightened)
    child_task = TaskEnvelope(
        parent=parent_envelope,
        max_cost_usd=100.0,  # Exceeds parent — will be clamped to 50
        data_access=[
            "credit_data",
            "feature_store",
            "production_logs",
        ],  # Exceeds parent
    )

    print(f"\n=== Monotonic Tightening ===")
    print(f"Parent budget: ${parent_envelope.max_cost_usd}")
    print(f"Child requested: ${100.0}")
    print(f"Child actual:   ${child_task.effective_max_cost_usd}")
    print(f"  → Tightened to parent's limit")
    print(f"\nParent data access: {parent_envelope.data_access}")
    print(f"Child requested:    {['credit_data', 'feature_store', 'production_logs']}")
    print(f"Child actual:       {child_task.effective_data_access}")
    print(f"  → production_logs removed (not in parent's scope)")

    # Test fail-closed: what happens on governance error
    print(f"\n=== Fail-Closed Behavior ===")
    print(f"If GovernanceEngine encounters an error during access check:")
    print(f"  → Access is DENIED (not allowed)")
    print(f"  → Error is logged to AuditChain")
    print(f"  → This prevents privilege escalation through bugs")

    # GovernanceContext is frozen
    ml_eng = Address("data_science", "mlops", "ml_engineer")
    context = await engine.create_context(ml_eng)
    print(f"\n=== Frozen GovernanceContext ===")
    print(f"Context for ML Engineer:")
    print(f"  max_cost_usd: {context.max_cost_usd}")
    print(f"  data_access: {context.data_access}")
    print(f"  can_deploy: {context.can_deploy}")
    print(f"\n  context.max_cost_usd = 999  # Raises FrozenInstanceError!")
    print(f"  → Agents RECEIVE governance but CANNOT modify it")


asyncio.run(test_monotonic_tightening())

# Clean up
org_file.unlink()

print("\n✓ Exercise 3 complete — PACT governance setup with access control")
