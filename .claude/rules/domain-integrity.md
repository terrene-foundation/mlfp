# Domain Integrity — Assessment Quality

Adapted from CO for Education (COE). Assessments must test genuine understanding, not pattern matching.

## AI-Resilient Assessment

Questions MUST require one or more of:
- **Context-specific application** — Use the student's own exercise outputs, not generic examples
- **Process documentation** — Show the reasoning journey (DataExplorer output → feature decision → model choice)
- **Code debugging** — Find and fix real SDK errors (missing `.build()`, wrong import path)
- **Architecture decisions** — "Which two frameworks would you combine for X, and why?"
- **Output interpretation** — "What does this DriftMonitor alert mean for your production model?"

## MUST NOT

- Ask pure recall questions ("What does PSI stand for?")
- Accept answers that a language model could generate without running the code
- Test generic ML theory instead of Kailash SDK patterns
- Grade on code syntax alone — assess the decision behind the code

## Quiz-to-Module Alignment

Every quiz question MUST map to a specific module learning outcome. If a question doesn't test a stated outcome, either the question or the outcome needs revision.
