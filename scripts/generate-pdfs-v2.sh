#!/bin/bash
# V2 PDF generation — uses decktape for Reveal.js decks (reliable)
# Chrome headless for textbook/notes (portrait HTML)
# pandoc → Chrome for markdown
set -eu

BASE="/Users/esperie/repos/lyceum/courses/mlfp"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
DECKTAPE="$BASE/node_modules/.bin/decktape"
OUT="$BASE/pdf"
LOG="$BASE/scripts/pdf-build.log"
TMP="/tmp/mlfp-pdf-build"

mkdir -p "$OUT"/{decks,textbooks,notes,lessons}
mkdir -p "$TMP"
: > "$LOG"

deck_pdf() {
  local src="$1" dst="$2"
  "$DECKTAPE" reveal --size 1280x720 "file://$src" "$dst" >>"$LOG" 2>&1
}

portrait_pdf() {
  local src="$1" dst="$2"
  "$CHROME" --headless=new --disable-gpu --no-sandbox \
    --print-to-pdf="$dst" --print-to-pdf-no-header \
    --virtual-time-budget=15000 --run-all-compositor-stages-before-draw \
    "file://$src" 2>>"$LOG"
}

md_to_pdf() {
  local src="$1" dst="$2" title="$3"
  local html_tmp="$TMP/$(basename "$src" .md).html"
  pandoc "$src" -o "$html_tmp" --standalone --toc --toc-depth=3 \
    --metadata title="$title" \
    --css=https://cdn.jsdelivr.net/npm/github-markdown-css@5/github-markdown.min.css \
    2>>"$LOG"
  portrait_pdf "$html_tmp" "$dst"
}

echo "[$(date)] V2 PDF build starting" | tee -a "$LOG"

# ── 1. Master decks (6 decktape) ─────────────────────────
echo "[1/4] Master decks with decktape..."
for m in mlfp01 mlfp02 mlfp03 mlfp04 mlfp05 mlfp06; do
  src="$BASE/modules/$m/deck.html"
  dst="$OUT/decks/$m-master.pdf"
  if [ -f "$src" ]; then
    echo "  $m..."
    deck_pdf "$src" "$dst" && echo "    → $(du -h "$dst" | awk '{print $1}')"
  fi
done

# ── 2. Master textbooks (markdown → PDF) ─────────────────
echo "[2/4] Master textbooks..."
for m in mlfp01 mlfp02 mlfp03 mlfp04 mlfp05 mlfp06; do
  src="$BASE/modules/$m/textbook.md"
  dst="$OUT/textbooks/$m-textbook.pdf"
  if [ -f "$src" ]; then
    echo "  $m..."
    md_to_pdf "$src" "$dst" "MLFP $m — Textbook"
  fi
done

# ── 3. Master speaker notes (markdown → PDF) ─────────────
echo "[3/4] Master speaker notes..."
for m in mlfp01 mlfp02 mlfp03 mlfp04 mlfp05 mlfp06; do
  src="$BASE/modules/$m/speaker-notes.md"
  dst="$OUT/notes/$m-speaker-notes.pdf"
  if [ -f "$src" ]; then
    echo "  $m..."
    md_to_pdf "$src" "$dst" "MLFP $m — Speaker Notes"
  fi
done

# ── 4. Per-lesson: textbook + notes (portrait) + slides (deck) ──
echo "[4/4] Per-lesson PDFs (144 files)..."
for m in mlfp01 mlfp02 mlfp03 mlfp04 mlfp05 mlfp06; do
  for lesson_dir in "$BASE/modules/$m/lessons"/*/; do
    lesson_num=$(basename "$lesson_dir")
    mkdir -p "$OUT/lessons/$m/$lesson_num"

    if [ -f "$lesson_dir/textbook.html" ]; then
      portrait_pdf "$lesson_dir/textbook.html" "$OUT/lessons/$m/$lesson_num/textbook.pdf"
    fi
    if [ -f "$lesson_dir/slides.html" ]; then
      deck_pdf "$lesson_dir/slides.html" "$OUT/lessons/$m/$lesson_num/slides.pdf"
    fi
    if [ -f "$lesson_dir/notes.html" ]; then
      portrait_pdf "$lesson_dir/notes.html" "$OUT/lessons/$m/$lesson_num/notes.pdf"
    fi
  done
  echo "  $m done"
done

echo "[$(date)] V2 PDF build complete" | tee -a "$LOG"
total=$(find "$OUT" -name "*.pdf" | wc -l | tr -d ' ')
size=$(du -sh "$OUT" | awk '{print $1}')
echo "Total PDFs: $total, Size: $size"
