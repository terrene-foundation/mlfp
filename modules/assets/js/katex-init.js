// ── MLFP KaTeX renderer (idempotent, single-pass) ────────────────────
// Root cause of prior double-render: our authored class `katex-display`
// collides with the KaTeX internal class of the same name. After KaTeX
// renders into an element, that element contains a NESTED
// <span class="katex-display">. A naive
// `querySelectorAll('.katex-display')` on a second pass picks up both
// the authored outer span AND the KaTeX-created inner span,
// double-rendering the inner span using its polluted textContent
// (which now includes invisible glyphs U+200B / U+2063 / U+2009 that
// KaTeX emits as spacing, plus a nested <annotation> copy of the LaTeX
// source). Each slidechanged event compounds the corruption and
// eventually wedges Chromium's layout engine during PDF export.
//
// Fix: only process authored spans that live inside `.equation-box`
// (displays) or `span.katex-inline` not inside an already-rendered
// `.katex` subtree (inlines). Capture the original LaTeX source in a
// data-mlfp-source attribute on first sight and always render from
// that stored source, never from el.textContent on a subsequent pass.
// Idempotence via data-mlfp-rendered.
//
// Contract for deck authors:
//   Display math   → <div class="equation-box"><span class="katex-display">LATEX</span></div>
//   Inline math    → <span class="katex-inline">LATEX</span>
// Dollar-delimiter syntax ($…$, $$…$$) is NOT supported by this
// renderer — use the class-based markup above so the idempotence guard
// has something to latch onto.
//
// Usage (after Reveal.initialize has been called):
//   <script src="../assets/js/katex-init.js"></script>

(function () {
  "use strict";

  // Pre-capture every authored span's LaTeX source BEFORE any rendering
  // runs. Authored spans are children of `.equation-box` (displays) or
  // inline `span.katex-inline`. We tag them with
  // data-mlfp-authored="true" so the render loop can unambiguously
  // distinguish authored spans from KaTeX-inserted nested spans (which
  // KaTeX creates with the same classes).
  function mlfpTagAuthored() {
    document
      .querySelectorAll(".equation-box > span.katex-display")
      .forEach((el) => {
        if (!el.dataset.mlfpAuthored) {
          el.dataset.mlfpAuthored = "true";
          el.dataset.mlfpSource = el.textContent;
        }
      });
    document.querySelectorAll("span.katex-inline").forEach((el) => {
      if (el.dataset.mlfpAuthored) return;
      if (el.closest(".katex")) return;
      el.dataset.mlfpAuthored = "true";
      el.dataset.mlfpSource = el.textContent;
    });
  }

  function mlfpRenderMath() {
    if (!window.katex) return;
    mlfpTagAuthored();
    document
      .querySelectorAll('span.katex-display[data-mlfp-authored="true"]')
      .forEach((el) => {
        if (el.dataset.mlfpRendered === "true") return;
        try {
          window.katex.render(el.dataset.mlfpSource, el, {
            displayMode: true,
            throwOnError: false,
            strict: "ignore",
          });
        } catch (e) {
          console.warn(
            "KaTeX display render failed:",
            e.message,
            el.dataset.mlfpSource,
          );
        }
        el.dataset.mlfpRendered = "true";
      });
    document
      .querySelectorAll('span.katex-inline[data-mlfp-authored="true"]')
      .forEach((el) => {
        if (el.dataset.mlfpRendered === "true") return;
        try {
          window.katex.render(el.dataset.mlfpSource, el, {
            displayMode: false,
            throwOnError: false,
            strict: "ignore",
          });
        } catch (e) {
          console.warn(
            "KaTeX inline render failed:",
            e.message,
            el.dataset.mlfpSource,
          );
        }
        el.dataset.mlfpRendered = "true";
      });
  }

  // Tag immediately so authored source is captured before any Reveal
  // event can trigger a render pass against polluted textContent.
  mlfpTagAuthored();

  if (window.Reveal) {
    window.Reveal.on("ready", () => mlfpRenderMath());
    window.Reveal.on("slidechanged", () => mlfpRenderMath());
  }
  requestAnimationFrame(() => mlfpRenderMath());
})();
