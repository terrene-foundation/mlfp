#!/usr/bin/env bash
# Build or sync the student-facing repository from the source repo.
#
# Usage:
#   ./scripts/build-student-repo.sh <target-directory>          # Fresh build
#   ./scripts/build-student-repo.sh --sync <target-directory>   # Sync to existing repo
#
# Copies ONLY student-facing content (see downstream.yaml for the full spec).
set -euo pipefail

SYNC_MODE=false
if [ "${1:-}" = "--sync" ]; then
    SYNC_MODE=true
    shift
fi

TARGET="${1:?Usage: $0 [--sync] <target-directory>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ "$SYNC_MODE" = false ] && [ -d "$TARGET" ] && [ "$(ls -A "$TARGET" 2>/dev/null)" ]; then
    echo "Error: $TARGET exists and is not empty. Use --sync to update an existing repo."
    exit 1
fi

echo "$([ "$SYNC_MODE" = true ] && echo "Syncing" || echo "Building") student repo"
echo "  Source: $SOURCE"
echo "  Target: $TARGET"

mkdir -p "$TARGET"

# ── Data (custom datasets only — standard ones auto-download) ──
echo "Syncing data/..."
if [ -d "$SOURCE/data" ]; then
    rsync -a --delete \
        --exclude='mlfp05/mnist/' \
        --exclude='mlfp05/fashion_mnist/' \
        --exclude='mlfp05/cifar10/' \
        --exclude='mlfp05/cora/' \
        --exclude='mlfp05/agnews/' \
        "$SOURCE/data/" "$TARGET/data/"
fi

# ── Modules (student-facing content only) ──
echo "Syncing modules/..."
for module_dir in "$SOURCE"/modules/mlfp*/; do
    mod=$(basename "$module_dir")
    dest="$TARGET/modules/$mod"
    mkdir -p "$dest"

    # readings/ — PDFs
    [ -d "$module_dir/readings" ] && rsync -a "$module_dir/readings/" "$dest/readings/"

    # local/ — exercise .py files (R10 directories)
    [ -d "$module_dir/local" ] && rsync -a --delete "$module_dir/local/" "$dest/local/"

    # colab-selfcontained/ — canonical Colab format (self-contained, no git clone)
    [ -d "$module_dir/colab-selfcontained" ] && rsync -a --delete "$module_dir/colab-selfcontained/" "$dest/colab-selfcontained/"

    # diagnostic-reference/ — captured outputs (plots + reports)
    [ -d "$module_dir/diagnostic-reference" ] && rsync -a --delete "$module_dir/diagnostic-reference/" "$dest/diagnostic-reference/"


    # solutions/ — reference solutions (students can consult after attempting)
    [ -d "$module_dir/solutions" ] && rsync -a --delete --exclude='__pycache__' "$module_dir/solutions/" "$dest/solutions/"

    # quiz/ — graded assessments (exclude *_solutions.ipynb + quiz_harness_solutions)
    [ -d "$module_dir/quiz" ] && rsync -a --delete \
        --exclude='*_solutions.ipynb' \
        --exclude='_solutions/' \
        --exclude='__pycache__' \
        "$module_dir/quiz/" "$dest/quiz/"

    # assessment/ — ship the WORK artifacts only (problem.md + starter.py).
    # Students complete the starters, zip, and submit to the external grading
    # portal. NEVER ship answer keys or anything that reveals them:
    #   - solution.py     reference answer
    #   - grader.py       re-derives the answer from data (leaks the formula)
    #   - exam.py         the fully-solved module exam
    #   - README.md       self-grade instructions referencing the withheld grader
    if [ -d "$module_dir/assessment" ]; then
        rsync -a --delete \
            --exclude='__pycache__' \
            --exclude='solution.py' \
            --exclude='grader.py' \
            --exclude='exam.py' \
            --exclude='README.md' \
            --exclude='*solution*' \
            --exclude='.claude/' \
            --exclude='.kailash_ml/' \
            --exclude='.env' \
            --exclude='*.onnx' \
            --exclude='*.jsonl' \
            --exclude='*.pkl' \
            --exclude='*.db' \
            --exclude='.DS_Store' \
            "$module_dir/assessment/" "$dest/assessment/"
        # Drop any assessment subdir left empty after answer-key exclusion.
        find "$dest/assessment" -type d -empty -delete 2>/dev/null || true
    fi

    # NOTE: colab-selfcontained-solutions/ is NEVER synced to students (instructor-only).

    echo "  $mod: $(ls "$dest" 2>/dev/null | tr '\n' ' ')"
done

# ── Shared utilities ──
echo "Syncing shared/..."
[ -d "$SOURCE/shared" ] && rsync -a --delete --delete-excluded \
    --exclude='__pycache__' \
    --exclude='.claude/' \
    --exclude='.coc/' \
    --exclude='.codex/' \
    --exclude='.gemini/' \
    "$SOURCE/shared/" "$TARGET/shared/"

# ── Config files ──
echo "Syncing config files..."
for f in pyproject.toml uv.lock LICENSE .env.example .gitignore; do
    [ -f "$SOURCE/$f" ] && cp "$SOURCE/$f" "$TARGET/$f"
done

# ── README (student-specific if it exists) ──
if [ -f "$SOURCE/README-student.md" ]; then
    cp "$SOURCE/README-student.md" "$TARGET/README.md"
fi

# ── Clean artifacts ──
find "$TARGET" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$TARGET" -name ".DS_Store" -delete 2>/dev/null || true
find "$TARGET" -name "*.pyc" -delete 2>/dev/null || true

# ── Safety scrub: stray secrets / COC artifacts / built models must never ship ──
# (.env.example is intentionally kept as a template; only real .env is scrubbed.)
find "$TARGET" -name ".env" -delete 2>/dev/null || true
find "$TARGET" -name "*.onnx" -delete 2>/dev/null || true
find "$TARGET" -path "*/.claude/*" -delete 2>/dev/null || true
find "$TARGET" -type d -name ".claude" -exec rm -rf {} + 2>/dev/null || true
find "$TARGET" -type d -name ".coc" -exec rm -rf {} + 2>/dev/null || true
find "$TARGET" -type d -name ".kailash_ml" -exec rm -rf {} + 2>/dev/null || true

# ── Summary ──
echo ""
echo "=== Student repo $([ "$SYNC_MODE" = true ] && echo "synced" || echo "built") ==="
echo "Location: $TARGET"
echo ""
echo "Modules:"
for mod_dir in "$TARGET"/modules/mlfp*/; do
    mod=$(basename "$mod_dir")
    exercises=$(find "$mod_dir/local" -name "*.py" 2>/dev/null | wc -l | tr -d ' ')
    notebooks=$(find "$mod_dir/colab-selfcontained" -name "*.ipynb" 2>/dev/null | wc -l | tr -d ' ')
    pdfs=$(find "$mod_dir/readings" -name "*.pdf" 2>/dev/null | wc -l | tr -d ' ')
    echo "  $mod: ${exercises} exercises, ${notebooks} notebooks, ${pdfs} PDFs"
done

if [ "$SYNC_MODE" = true ]; then
    echo ""
    echo "Next: cd $TARGET && git add -A && git diff --cached --stat"
    echo "Then: git commit -m 'sync: update from source' && git push"
else
    echo ""
    echo "Next: cd $TARGET && git init && git add -A && git commit -m 'Initial student repo'"
fi
