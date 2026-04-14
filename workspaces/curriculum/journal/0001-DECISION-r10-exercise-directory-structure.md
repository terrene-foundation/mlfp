---
type: DECISION
date: 2026-04-13
created_at: 2026-04-13T16:00:00Z
author: co-authored
project: curriculum
topic: Redline 10 — exercise directory structure with per-technique files and 5-phase narrative arc
phase: codify
tags: [R10, exercise-structure, directory-layout, narrative-arc, two-format]
---

## Decision

Adopted Redline 10 (R10): exercises are now structured as directories (not monolithic files) with per-technique files following a 5-phase narrative arc: Theory, Build, Train, Visualise, Apply. This applies to all modules and both delivery formats (VS Code local, Colab notebooks).

The directory structure is:

```
modules/mlfpNN/solutions/ex_N/
  01_technique_a.py
  02_technique_b.py
  03_technique_c.py
```

Each file follows the 5-phase arc internally, providing a complete learning journey per technique rather than a single monolithic exercise covering everything superficially.

## Alternatives Considered

1. **Monolithic single-file exercises** (status quo before R10) — one `ex_N.py` per exercise covering all techniques. Rejected because files grew too large, students lost narrative thread, and per-technique visual proof was impossible to maintain.

2. **Phase-based files** (one file per phase across all techniques) — `01_theory.py`, `02_build.py`, etc. Rejected because it fragments the per-technique narrative and makes it impossible for students to see a complete technique journey in one file.

3. **Technique directories with phase files** (nested two levels) — `ex_N/technique_a/01_theory.py`. Rejected as over-engineered; the 5-phase arc fits naturally within a single file per technique.

## Rationale

- Each technique file is self-contained: students can complete one technique per sitting without losing context
- The 5-phase arc (Theory, Build, Train, Visualise, Apply) ensures every technique includes visual proof and real-world application (satisfying R9)
- Per-technique files map cleanly to Colab notebooks (one notebook per technique file)
- The exercise-designer agent, two-format rule, and exercise-standards rule were all updated to enforce R10 consistently

## Consequences

- All 8 M5 exercises were restructured from monolithic files to R10 directories
- The exercise-designer agent now generates per-technique files instead of monolithic exercises
- The two-format rule was updated to handle directory-based exercises
- A STUDENT-REPO.md and build script were created to document the student/instructor content boundary
- Readings directories were created in each module for student-facing PDFs (deck, textbook, notes)

## For Discussion

1. If a technique is too small to justify a full 5-phase arc (e.g., a simple preprocessing step), should it be merged with an adjacent technique file, or should the arc phases be abbreviated? What is the minimum viable technique size?

2. The M5 restructuring produced 8 exercise directories. Given that M5 covers deep learning (CNNs, autoencoders, transfer learning, ONNX), does the per-technique file count per exercise (typically 3-5 files) provide enough depth, or would some techniques benefit from being split further? Specifically, does the autoencoder exercise (ex_2) with its 4 variant files adequately separate vanilla, denoising, convolutional, and variational approaches?

3. How will progressive disclosure (Directive 5: M5 provides ~30% scaffolding) interact with the per-technique file structure? If each technique file is self-contained, does the reduced scaffolding mean students navigate between files with less guidance than in monolithic exercises where the narrative flow was linear?
