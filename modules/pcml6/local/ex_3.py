# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT6 — Exercise 3: PACT Governance Setup
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

# TODO: Write a PACT organization YAML string with 3 departments and 6+ roles.
# Hint: The org_yaml string must follow this structure:
#
#   organization:
#     name: "ASCENT Credit Bureau"
#     version: "1.0"
#     departments:
#       - name: "data_science"
#         description: "..."
#         teams:
#           - name: "modeling"
#             roles:
#               - name: "senior_data_scientist"
#                 permissions:
#                   data_access: ["credit_data", "feature_store", "experiment_logs"]
#                   tools: ["training_pipeline", "hyperparameter_search", "model_registry"]
#                   max_cost_usd: 50.0
#                   can_deploy: false
#               - name: "junior_data_scientist"
#                 permissions:
#                   data_access: ["credit_data", "feature_store"]
#                   tools: ["training_pipeline"]
#                   max_cost_usd: 10.0
#                   can_deploy: false
#           - name: "mlops"
#             roles:
#               - name: "ml_engineer"
#                 permissions:
#                   data_access: ["credit_data", "feature_store", "model_artifacts", "production_logs"]
#                   tools: ["training_pipeline", "model_registry", "inference_server", "drift_monitor"]
#                   max_cost_usd: 100.0
#                   can_deploy: true
#       - name: "risk"   (model_validator: can_approve_production: true, compliance_officer: read-only)
#       - name: "operations"   (sre: can_deploy: true, customer_service: decisions only)
#     policies:
#       - name: "model_promotion"
#         rule: "promote_to_production requires model_validator.can_approve_production"
#       - name: "data_classification"
#         classifications:
#           credit_data: "confidential"
#           audit_logs: "restricted"
org_yaml = ____

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

# TODO: Load the YAML file and compile it.
# Hint: org = load_org_yaml(str(org_file))
#   compiled = compile_org(org)
#   compiled has: .valid (bool), .errors (list), .warnings (list),
#                 .departments (list), .total_roles (int), .name (str)
org = ____
compiled = ____

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
    # TODO: Create a GovernanceEngine from the compiled organization.
    # Hint: engine = GovernanceEngine(compiled)
    #   The engine is fail-closed: any error during access check → DENY
    engine = ____

    print(f"\n=== GovernanceEngine ===")
    print(f"Organization: {compiled.name}")
    print(f"Enforcement mode: fail-closed (deny on error)")

    return engine


engine = asyncio.run(setup_governance())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Test access decisions
# ══════════════════════════════════════════════════════════════════════


async def test_access():
    # TODO: Use Address to create D/T/R addresses, then call engine.can_access()
    # and engine.explain_access() to test access control decisions.
    #
    # Hint: Address("department", "team", "role") — matches the YAML structure.
    #   decision = await engine.can_access(address, resource_name)  → bool
    #   explanation = await engine.explain_access(address, resource_name)
    #     explanation has: .decision (str), .reason (str), .chain (list)
    #
    # Test cases to verify:
    #   senior_data_scientist → "credit_data": ALLOW (in permissions)
    #   junior_data_scientist → "experiment_logs": DENY (not in permissions)
    #   ml_engineer → "production_logs": ALLOW
    #   model_validator → "audit_logs": ALLOW
    #   customer_service → "credit_data": DENY (can only access credit_decisions)
    #   customer_service → "credit_decisions": ALLOW

    # TODO: Create Address objects for each role.
    # Hint: Address("data_science", "modeling", "senior_data_scientist")
    senior_ds = ____
    junior_ds = ____
    ml_eng = ____
    validator = ____
    cust_svc = ____

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
        # TODO: Call engine.can_access() to get the access decision.
        # Hint: decision = await engine.can_access(address, resource)  → bool
        decision = await ____
        status = "✓" if decision == expected else "✗ UNEXPECTED"
        print(f"  {status} {description}")
        print(f"     {address} → {resource}: {'ALLOW' if decision else 'DENY'}")

    # Explain access
    print(f"\n=== Access Explanations ===")
    # TODO: Call engine.explain_access() to get a detailed explanation.
    # Hint: explanation = await engine.explain_access(junior_ds, "production_logs")
    #   explanation.decision, explanation.reason, explanation.chain
    explanation = await ____
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
    # TODO: Create a RoleEnvelope and a TaskEnvelope that tries to exceed it.
    # Hint: RoleEnvelope(
    #   address=Address("data_science", "modeling", "senior_data_scientist"),
    #   max_cost_usd=50.0,
    #   data_access=["credit_data", "feature_store"],
    # )
    # TaskEnvelope(
    #   parent=parent_envelope,
    #   max_cost_usd=100.0,  # will be clamped to 50 (parent's limit)
    #   data_access=["credit_data", "feature_store", "production_logs"],  # production_logs removed
    # )
    # Check: child_task.effective_max_cost_usd and child_task.effective_data_access
    parent_envelope = ____
    child_task = ____

    print(f"\n=== Monotonic Tightening ===")
    print(f"Parent budget: ${parent_envelope.max_cost_usd}")
    print(f"Child requested: ${100.0}")
    print(f"Child actual:   ${child_task.effective_max_cost_usd}")
    print(f"  → Tightened to parent's limit")
    print(f"\nParent data access: {parent_envelope.data_access}")
    print(f"Child requested:    {['credit_data', 'feature_store', 'production_logs']}")
    print(f"Child actual:       {child_task.effective_data_access}")
    print(f"  → production_logs removed (not in parent's scope)")

    print(f"\n=== Fail-Closed Behavior ===")
    print(f"If GovernanceEngine encounters an error during access check:")
    print(f"  → Access is DENIED (not allowed)")
    print(f"  → Error is logged to AuditChain")
    print(f"  → This prevents privilege escalation through bugs")

    # TODO: Create a GovernanceContext using engine.create_context() and demonstrate
    # that it is frozen (cannot be modified).
    # Hint: ml_eng = Address("data_science", "mlops", "ml_engineer")
    #   context = await engine.create_context(ml_eng)
    #   context.max_cost_usd, context.data_access, context.can_deploy
    ml_eng = Address("data_science", "mlops", "ml_engineer")
    context = await ____
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
