#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Grader for MLFP04 Assessment Task 4 — Topics from News Text
(instructor-side; not distributed to students).

    python grader.py submission.py [--seed N]

Two secret corpora are drawn from the AG News pool (train + test files): a
random 3 or 4 of the four news sections, ~350 raw articles each, shuffled.
The section labels never reach the submission. The grader scores the
submission's topics against those labels (NMI), recomputes NPMI coherence of
its top words on the same documents, checks the topics are distinct, and
checks every topic's words actually describe the documents assigned to it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _corpus import news_pool, npmi, secret_sample, tokens  # noqa: E402
from grading_harness import Checks, finalize, load_student_module, main, uses_engine  # noqa: E402

WEIGHT = 15
NMI_FLOOR = 0.20
NPMI_MEAN_FLOOR = 0.15
NPMI_MIN_FLOOR = 0.0
NPMI_WORST_TOLERANCE = -0.10  # one borderline topic may dip below zero on hard draws
OVERLAP_MAX = 2      # a word may appear in at most this many topics' top-10
LIFT_FLOOR = 2.0     # topic words must be >= 2x more frequent in the topic's own documents


def grade(path: Path, seed: int) -> dict:
    checks = Checks()
    try:
        st = load_student_module(path, "student_m4_task4")
    except Exception as e:
        return finalize(checks, WEIGHT, seed, f"Failed to import: {type(e).__name__}: {e}")
    if not callable(getattr(st, "discover_topics", None)):
        return finalize(checks, WEIGHT, seed, "Missing function: discover_topics")
    rng = np.random.default_rng(seed)
    pool = news_pool()
    corpora = [secret_sample(rng, pool) for _ in range(2)]

    checks.add("uses_dim_reduction_engine", uses_engine(path, "DimReductionEngine"),
               "topic factorisation must run through kailash-ml DimReductionEngine")
    names = ["output_format", "topics_track_sections", "topics_coherent", "topics_distinct", "words_describe_documents"]

    def run():
        ok = dict.fromkeys(names, True)
        notes = []
        for docs, labels, k in corpora:
            out = st.discover_topics(list(docs), k)
            assign = np.asarray(out["doc_topics"])
            words = [[str(w).lower() for w in t] for t in out["top_words"]]
            fmt = (assign.shape == (len(docs),) and np.issubdtype(assign.dtype, np.integer)
                   and assign.min() >= 0 and assign.max() < k and len(words) == k
                   and all(len(t) == 10 and len(set(t)) == 10 for t in words))
            if not fmt:
                return {n: (False, f"need {len(docs)} topic ids in [0, {k}) and {k} lists of 10 distinct words") for n in names}
            nmi = normalized_mutual_info_score(labels, assign)
            dt = [tokens(d) for d in docs]
            coh = [npmi(dt, t) for t in words]
            counts: dict[str, int] = {}
            for t in words:
                for w in t:
                    counts[w] = counts.get(w, 0) + 1
            worst_overlap = max(counts.values())
            lifts = []
            for j, t in enumerate(words):
                mine = [d for d, a in zip(dt, assign) if a == j]
                rest = [d for d, a in zip(dt, assign) if a != j]
                if not mine or not rest:
                    lifts.append(0.0)
                    continue
                f_in = np.mean([np.mean([w in d for d in mine]) for w in t])
                f_out = np.mean([np.mean([w in d for d in rest]) for w in t])
                lifts.append(float(f_in / max(f_out, 1e-9)))
            ok["topics_track_sections"] &= nmi >= NMI_FLOOR
            worst = min(coh)
            ok["topics_coherent"] &= (
                np.mean(coh) >= NPMI_MEAN_FLOOR
                and (worst >= NPMI_MIN_FLOOR or worst > NPMI_WORST_TOLERANCE)
            )
            ok["topics_distinct"] &= worst_overlap <= OVERLAP_MAX
            ok["words_describe_documents"] &= min(lifts) >= LIFT_FLOOR
            notes.append(f"{k} sections: NMI {nmi:.3f}; NPMI {np.round(coh, 3).tolist()}; "
                         f"max topics sharing a word {worst_overlap}; word lift {np.round(lifts, 2).tolist()}")
        return {n: (ok[n], "; ".join(notes)) for n in names}

    checks.guarded(names, run)
    checks.require(["topics_track_sections", "topics_coherent"],
                   ["uses_dim_reduction_engine", "output_format", "topics_distinct", "words_describe_documents"])
    return finalize(checks, WEIGHT, seed)


if __name__ == "__main__":
    main(grade)
