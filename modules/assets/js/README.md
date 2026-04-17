# Shared deck scripts

## `katex-init.js`

Idempotent KaTeX renderer for Reveal.js decks. Replaces per-deck renderer blocks so the double-render fix lives in one place.

### Why this exists

KaTeX's rendered output wraps the target element in a nested `<span class="katex-display">`. A naive `querySelectorAll('.katex-display')` pass on `Reveal.on('slidechanged')` picks up both the authored span AND the nested one, re-rendering from polluted `textContent` that now includes invisible spacing glyphs and the LaTeX `<annotation>` copy. The corruption compounds per slide change and can wedge Chromium's layout engine during `decktape` PDF export — that was the M6 crash.

This script fixes it by:

1. Pre-capturing each authored span's LaTeX source into `data-mlfp-source` on first sight (before any render pass can pollute `textContent`).
2. Tagging authored spans with `data-mlfp-authored="true"` so the render loop can distinguish them from KaTeX-created nested spans.
3. Marking rendered spans with `data-mlfp-rendered="true"` so repeated invocations are no-ops.

### Contract for deck authors

- **Display math:** `<div class="equation-box"><span class="katex-display">LATEX</span></div>`
- **Inline math:** `<span class="katex-inline">LATEX</span>`

Dollar-delimiter syntax (`$…$`, `$$…$$`) is **not** supported — the idempotence guard relies on class-based markup.

### Usage

After `Reveal.initialize({...})` in your deck, include:

```html
<!-- Idempotent KaTeX renderer — shared across decks. See modules/assets/js/README.md. -->
<script src="../assets/js/katex-init.js"></script>
```

The script auto-registers `Reveal.on('ready')` and `Reveal.on('slidechanged')` handlers and runs one `requestAnimationFrame` pass as a fallback.

### Decks using this script

- `modules/mlfp05/deck.html`
- `modules/mlfp06/deck.html`

### Decks NOT yet migrated

- `mlfp01`, `mlfp02` — use `$…$` / `$$…$$` delimiter syntax via KaTeX auto-render. Migrating requires rewriting every equation to the class-based markup.
- `mlfp03` — class-based markup but a simpler non-idempotent init; can be migrated by swapping the script tag.
- `mlfp04` — uses the `RevealMath.KaTeX` plugin (different approach entirely).

None of these have crashed decktape in practice, so migration is a cleanup task, not a blocker. If a future M1–M4 regeneration hits the Chromium wedge, migrate that deck to this pattern.
