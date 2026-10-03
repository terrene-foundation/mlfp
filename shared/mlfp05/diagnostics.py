# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Print the Prescription Pad for a kailash-ml DL diagnostic run.

``kailash_ml.diagnostics.run_diagnostic_checkpoint`` and
``DLDiagnostics.report()`` RETURN a findings dict; neither prints it.
Exercises call :func:`print_prescription_pad` so students actually see
the three automated readings (gradient flow, dead neurons, loss trend).

    from kailash_ml.diagnostics import run_diagnostic_checkpoint
    from shared.mlfp05.diagnostics import print_prescription_pad

    diag, findings = run_diagnostic_checkpoint(model, loader, loss_fn,
                                               title="CNN", show=False)
    print_prescription_pad(findings, "CNN")
"""
from __future__ import annotations

import textwrap
from typing import Any, Mapping

_MARKS = {"HEALTHY": "[ok]", "WARNING": "[!] ", "CRITICAL": "[X] ", "UNKNOWN": "[?] "}
_LABELS = {
    "gradient_flow": "Gradient flow",
    "dead_neurons": "Dead neurons",
    "loss_trend": "Loss trend",
}


def print_prescription_pad(findings: Mapping[str, Any], title: str) -> None:
    """Print every instrument reading in a ``report()`` findings dict.

    A reading is any entry whose value is a mapping with a ``severity``
    key. Readings the library could not compute appear as UNKNOWN with the
    library's own explanation, so a missing instrument is visible rather
    than silently dropped.
    """
    readings = [
        (key, value)
        for key, value in findings.items()
        if isinstance(value, Mapping) and "severity" in value
    ]
    print("=" * 66)
    print(f"  Prescription Pad — {title}")
    print("=" * 66)
    if not readings:
        print("  (the findings dict contains no instrument readings)")
    for key, value in readings:
        severity = str(value.get("severity", "UNKNOWN"))
        label = _LABELS.get(key, key.replace("_", " ").capitalize())
        print(f"  {_MARKS.get(severity, '[?] ')} {label} ({severity})")
        for line in textwrap.wrap(str(value.get("message", "")), width=58):
            print("        " + line)
    print("=" * 66)
