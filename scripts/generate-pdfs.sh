#!/bin/bash
# Generate PDFs for all MLFP course deliverables
# Decks (landscape 1280x720) via ?print-pdf via Chrome headless
# Textbooks + notes (portrait) via Chrome headless
# Markdown via pandoc → standalone HTML → Chrome
set -eu

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
BASE="/Users/esperie/repos/lyceum/courses/mlfp"
OUT="$BASE/pdf"
LOG="$BASE/scripts/pdf-build.log"
TMP="/tmp/mlfp-pdf-build"

mkdir -p "$OUT"/{decks,textbooks,notes,lessons}
mkdir -p "$TMP"
echo "[$(date)] Starting PDF build" > "$LOG"

# ── Helpers ──────────────────────────────────────────────

pdf_landscape() {
  # Reveal.js deck with ?print-pdf query param
  local src="$1" dst="$2"
  "$CHROME" --headless --disable-gpu --no-sandbox \
    --print-to-pdf="$dst" \
    --print-to-pdf-no-header \
    --virtual-time-budget=15000 \
    --run-all-compositor-stages-before-draw \
    "file://$src?print-pdf" 2>>"$LOG"
}

pdf_portrait() {
  local src="$1" dst="$2"
  "$CHROME" --headless --disable-gpu --no-sandbox \
    --print-to-pdf="$dst" \
    --print-to-pdf-no-header \
    --virtual-time-budget=15000 \
    --run-all-compositor-stages-before-draw \
    "file://$src" 2>>"$LOG"
}

pdf_markdown() {
  # Markdown → standalone HTML → portrait PDF
  local src="$1" dst="$2" title="$3"
  local html_tmp="$TMP/$(basename "$src" .md).html"
  pandoc "$src" \
    -o "$html_tmp" \
    --standalone \
    --toc \
    --toc-depth=3 \
    --metadata title="$title" \
    --css=https://cdn.jsdelivr.net/npm/github-markdown-css@5/github-markdown.min.css \
    2>>"$LOG"
  pdf_portrait "$html_tmp" "$dst"
}

# ── 1. Master decks (6 landscape) ───────────────────────
echo "[$(date)] [1/4] Building master decks..." | tee -a "$LOG"
for m in mlfp01 mlfp02 mlfp03 mlfp04 mlfp05 mlfp06; do
  src="$BASE/decks/$m/deck.html"
  dst="$OUT/decks/$m-master.pdf"
  if [ -f "$src" ]; then
    echo "  $m master deck"
    pdf_landscape "$src" "$dst"
  fi
done

# ── 2. Master textbooks (6 portrait) ─────────────────────
echo "[$(date)] [2/4] Building master textbooks (markdown)..." | tee -a "$LOG"
for m in mlfp01 mlfp02 mlfp03 mlfp04 mlfp05 mlfp06; do
  src="$BASE/decks/$m/textbook.md"
  dst="$OUT/textbooks/$m-textbook.pdf"
  if [ -f "$src" ]; then
    echo "  $m textbook"
    pdf_markdown "$src" "$dst" "MLFP $m — Textbook Chapter"
  fi
done

# ── 3. Master speaker notes (6 portrait) ─────────────────
echo "[$(date)] [3/4] Building master speaker notes (markdown)..." | tee -a "$LOG"
for m in mlfp01 mlfp02 mlfp03 mlfp04 mlfp05 mlfp06; do
  src="$BASE/decks/$m/speaker-notes.md"
  dst="$OUT/notes/$m-speaker-notes.pdf"
  if [ -f "$src" ]; then
    echo "  $m speaker notes"
    pdf_markdown "$src" "$dst" "MLFP $m — Speaker Notes"
  fi
done

# ── 4. Per-lesson HTML (144 files) ───────────────────────
echo "[$(date)] [4/4] Building per-lesson PDFs (144 files)..." | tee -a "$LOG"
for m in mlfp01 mlfp02 mlfp03 mlfp04 mlfp05 mlfp06; do
  mkdir -p "$OUT/lessons/$m"
  count=0
  for lesson_dir in "$BASE/decks/$m/lessons"/*/; do
    lesson_num=$(basename "$lesson_dir")
    mkdir -p "$OUT/lessons/$m/$lesson_num"

    # Textbook (portrait)
    if [ -f "$lesson_dir/textbook.html" ]; then
      pdf_portrait "$lesson_dir/textbook.html" "$OUT/lessons/$m/$lesson_num/textbook.pdf"
      count=$((count+1))
    fi

    # Slides (landscape)
    if [ -f "$lesson_dir/slides.html" ]; then
      pdf_landscape "$lesson_dir/slides.html" "$OUT/lessons/$m/$lesson_num/slides.pdf"
      count=$((count+1))
    fi

    # Notes (portrait)
    if [ -f "$lesson_dir/notes.html" ]; then
      pdf_portrait "$lesson_dir/notes.html" "$OUT/lessons/$m/$lesson_num/notes.pdf"
      count=$((count+1))
    fi
  done
  echo "  $m: $count lesson PDFs"
done

echo "[$(date)] PDF build complete" | tee -a "$LOG"
echo ""
echo "=== PDF inventory ==="
total=$(find "$OUT" -name "*.pdf" | wc -l | tr -d ' ')
echo "  Total PDFs: $total"
echo "  Location: $OUT"
du -sh "$OUT" | awk '{print "  Total size:", $1}'
