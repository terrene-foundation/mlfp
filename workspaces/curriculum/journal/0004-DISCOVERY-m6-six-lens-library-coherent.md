---
type: DISCOVERY
status: pending
session_id: m6-holistic-2026-04-17
created: 2026-04-17
---

# M6 holistic audit: six-lens library fully wired, 39 exercises coherent with 8 lessons

**Discovery**: M6 (LLMs and Agentic Workflows) is in a healthy state end-to-end. The LLM Observatory six-lens library implements the spec from `specs/m6-diagnostics-design.md`, all exercises AST-parse, and the exercise runtime degrades gracefully when API keys are absent.

## Evidence

### Library structure matches spec

The diagnostics subpackage at `shared/mlfp06/diagnostics/` has the spec's six lenses:

- Output Lens — `LLMDiagnostics` (output.py, 615 LOC)
- Attention Lens — `InterpretabilityDiagnostics` (interpretability.py, 529 LOC)
- Retrieval Lens — `RAGDiagnostics` (retrieval.py, 746 LOC)
- Agent Trace Lens — `AgentDiagnostics` (agent.py, 668 LOC)
- Alignment Lens — `AlignmentDiagnostics` (alignment.py, 649 LOC)
- Governance Lens — `GovernanceDiagnostics` (governance.py, 716 LOC)

Plus leaf utilities (_judges, _plots, _traces) and the top-level facade (observatory.py, 452 LOC).

### Facade is wired, not orphaned

`LLMObservatory` is imported by 78 solution files — every lesson's exercises compose through the facade rather than reaching into lens classes directly. Satisfies `rules/orphan-detection.md` + `rules/facade-manager-detection.md`. All six lens classes construct without arguments; the Observatory facade exposes them as `obs.{output,attention,retrieval,agent,alignment,governance}` plus `report()`, `plot_dashboard()`, `close()`.

### Exercise-lesson coherence

| Lesson | Exercise dir | Technique files | Helper LOC |
|---|---|---|---|
| 6.1 Prompting | ex_1 | 6 | 419 |
| 6.2 Fine-tuning | ex_2 | 6 | 182 |
| 6.3 Alignment (DPO/GRPO) | ex_3 | 4 | 301 |
| 6.4 RAG | ex_4 | 5 | 341 |
| 6.5 Agents (ReAct) | ex_5 | 4 | 302 |
| 6.6 Multi-agent + MCP | ex_6 | 5 | 286 |
| 6.7 PACT governance | ex_7 | 4 | 515 |
| 6.8 Nexus deployment | ex_8 | 5 | 665 |

39/39 M6 solutions AST-parse; 39/39 colab-selfcontained notebooks regenerated with the fixed generator (transitive imports + Colab-safe REPO_ROOT + subpackage flattening). Cell 1 exec tested clean on every module.

### Offline-graceful design

`shared/mlfp06/ex_1.py::run_delegate` wraps Kaizen `Delegate` calls in try/except and falls back to a deterministic stub when no API key is present. Smoke-tested ex_1/01_zero_shot end-to-end on 3 SST-2 docs: ran to REFLECTION in ~1 second, producing an accuracy plot and the canonical "What you've mastered" block. This is the design intent — exercises never burn API budget in CI or for students without keys — and it's the same pattern I'd expect on every LLM-calling helper.

## For Discussion

1. The six-lens facade uses `obs.attention` (pragmatic shortening) even though the class is named `InterpretabilityDiagnostics` and the spec calls it the "Attention Lens / X-Ray". Is the attribute name a good bridge between "what you call it in the deck" and "what the class does under the hood", or should it rename to `obs.interp` for consistency with `InterpretabilityDiagnostics`?
2. The offline-graceful fallback silently returns accuracy=0% when API keys are absent. A student running the notebook without a key gets a visually-complete result with zero accuracy — they may not realize the LLM never actually fired. Should the fallback print a loud banner (`⚠ OFFLINE MODE — results are stubbed`) at the top of the exercise run?
3. The LLM Observatory report() returns a plain dict with `summary` + `severity` per lens. That's right for programmatic use but hard for a student to read interactively. Is a Rich-formatted table or an HTML report a worthwhile next increment, or is the dict sufficient given the lens classes each have their own `plot_dashboard()`?
