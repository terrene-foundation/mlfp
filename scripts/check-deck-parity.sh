#!/usr/bin/env bash
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
#
# Rebuild M5 + M6 deck PDFs and compare their text content against the
# committed baseline. Catches rendering regressions in modules/assets/js/
# (e.g. a broken edit to katex-init.js) that would silently ship.
#
# Usage:
#   scripts/check-deck-parity.sh           # check against pdf/decks/*.pdf
#   scripts/check-deck-parity.sh --update  # refresh the baseline text files
set -euo pipefail

BASE="$(cd "$(dirname "$0")/.." && pwd)"
DECKTAPE="$BASE/node_modules/.bin/decktape"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

UPDATE=false
if [ "${1:-}" = "--update" ]; then
    UPDATE=true
fi

BASELINE_DIR="$BASE/scripts/deck-parity-baselines"
mkdir -p "$BASELINE_DIR"

# Decks that use the idempotent class-based renderer (katex-init.js).
# M1-M4 use older init patterns that are incompatible with this guard.
DECKS=(mlfp05 mlfp06)

fail=0
for m in "${DECKS[@]}"; do
    src="$BASE/modules/$m/deck.html"
    pdf="$TMP/$m.pdf"
    txt="$TMP/$m.txt"
    baseline="$BASELINE_DIR/$m.txt"

    "$DECKTAPE" reveal --size 1280x720 "file://$src" "$pdf" >/dev/null 2>&1
    pdftotext "$pdf" "$txt" 2>/dev/null

    if [ "$UPDATE" = true ] || [ ! -f "$baseline" ]; then
        cp "$txt" "$baseline"
        echo "✓ $m: baseline refreshed ($(wc -l < "$baseline" | tr -d ' ') lines)"
        continue
    fi

    if diff -q "$baseline" "$txt" >/dev/null 2>&1; then
        echo "✓ $m: deck content unchanged ($(wc -l < "$baseline" | tr -d ' ') lines)"
    else
        echo "✗ $m: deck content diverged from baseline"
        diff "$baseline" "$txt" | head -40
        fail=1
    fi
done

exit $fail
