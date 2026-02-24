#!/usr/bin/env bash
#
# Prune from git history all files that:
# 1. Are >100KB (non-code files)
# 2. Are NOT in the current repo (HEAD)
#
# These are typically: apps/api/runs/*, old assets, etc.
#
# Usage:
#   ./scripts/prune-deleted-large-files.sh [--dry-run] [--yes] [--runs-only] [--repo DIR]
#
#   --runs-only  Prune entire apps/api/runs/ directory from history (catches all files)
#   --repo DIR   Run on this repo (default: parent of script dir). Use for fresh clones.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_OVERRIDE=""
SIZE_KB=100
SIZE_BYTES=$((SIZE_KB * 1024))

DRY_RUN=false
YES=false
RUNS_ONLY=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)   DRY_RUN=true; shift ;;
    --yes)       YES=true; shift ;;
    --runs-only) RUNS_ONLY=true; shift ;;
    --repo)
      [[ -z "${2:-}" ]] && { echo "Error: --repo requires DIR"; exit 1; }
      REPO_OVERRIDE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--dry-run] [--yes] [--runs-only] [--repo DIR]"
      echo "  Prune large files (>100KB) no longer in current repo from git history."
      echo "  --dry-run    Show what would be removed, don't modify"
      echo "  --yes       Skip confirmation prompt"
      echo "  --runs-only Prune entire apps/api/runs/ dir (catches all, not just >100KB)"
      echo "  --repo DIR  Run on this repo (e.g. fresh clone in mm/)"
      exit 0
      ;;
    *) shift ;;
  esac
done

[[ -n "$REPO_OVERRIDE" ]] && REPO_ROOT="$(cd "$REPO_OVERRIDE" && pwd)"

cd "$REPO_ROOT"
[[ -n "$REPO_OVERRIDE" ]] && echo "Using repo: $REPO_ROOT" && echo ""

if [[ ! -d .git ]]; then
  echo "Error: Not a git repository. Run from apex-studio root."
  exit 1
fi

if ! command -v git-filter-repo &>/dev/null; then
  echo "Error: git-filter-repo required. Install with: pip install git-filter-repo"
  exit 1
fi

if [[ "$RUNS_ONLY" == true ]]; then
  echo "=== Prune apps/api/runs/ directory from git history ==="
  echo "Extracting all paths under apps/api/runs/ from history..."
  PATHS_FILE=$(mktemp)
  git rev-list --all --objects 2>/dev/null | \
    git cat-file --batch-check='%(objecttype) %(objectname) %(rest)' 2>/dev/null | \
    awk '/^blob / {path=$3; for(i=4;i<=NF;i++) path=path" "$i; if(path ~ /^apps\/api\/runs\//) print path}' | \
    sort -u > "$PATHS_FILE"
  RUNS_COUNT=$(wc -l < "$PATHS_FILE")
  echo "  Found $RUNS_COUNT unique paths under apps/api/runs/"
  echo ""
  if [[ $RUNS_COUNT -eq 0 ]]; then
    echo "Nothing to prune."
    rm -f "$PATHS_FILE"
    exit 0
  fi
  if [[ "$DRY_RUN" == true ]]; then
    echo "[DRY RUN] Would remove these paths (first 10):"
    head -10 "$PATHS_FILE"
    echo "[DRY RUN] ... and $((RUNS_COUNT - 10)) more"
    echo "[DRY RUN] Full list: $PATHS_FILE"
    exit 0
  fi
  if [[ "$YES" != true ]]; then
    echo "Sample of paths to remove:"
    head -5 "$PATHS_FILE"
    echo "..."
    read -p "Continue? (y/N) " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && echo "Aborted." && rm -f "$PATHS_FILE" && exit 1
  fi
  git filter-repo --invert-paths --paths-from-file "$PATHS_FILE" --force
  rm -f "$PATHS_FILE"
  echo ""
  echo "Done. Run:"
  echo "  git push --force --all"
  echo "  git push --force --tags   # CRITICAL: tags (v0.1.x) hold the runs history; without this, clones stay huge"
  echo "  git reflog expire --expire=now --all && git gc --prune=now --aggressive"
  exit 0
fi

echo "=== Prune deleted large files (>${SIZE_KB}KB) from git history ==="
echo ""

# 1. Files currently in repo (HEAD)
echo "Building list of files in current repo (HEAD)..."
CURRENT=$(mktemp)
git ls-tree -r HEAD --name-only 2>/dev/null | sort -u > "$CURRENT"
CURRENT_COUNT=$(wc -l < "$CURRENT")
echo "  $CURRENT_COUNT files in HEAD"
echo ""

# 2. All large blobs in history (output: path only, one per line)
echo "Scanning git history for blobs >${SIZE_KB}KB..."
ALL_LARGE=$(mktemp)
git rev-list --all --objects 2>/dev/null | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' 2>/dev/null | \
  awk -v threshold="$SIZE_BYTES" '
    /^blob / {
      if ($3 > threshold && $4 != "") {
        path = $4
        for (i=5; i<=NF; i++) path = path " " $i
        print path
      }
    }
  ' | sort -u > "$ALL_LARGE"
ALL_COUNT=$(wc -l < "$ALL_LARGE")
echo "  $ALL_COUNT unique large file paths in history"
echo ""

# 3. Paths to remove = in history but NOT in current repo
TO_REMOVE=$(mktemp)
comm -23 "$ALL_LARGE" "$CURRENT" > "$TO_REMOVE"
REMOVE_COUNT=$(wc -l < "$TO_REMOVE")

echo "Paths to prune (in history but not in current repo): $REMOVE_COUNT"
echo ""

if [[ $REMOVE_COUNT -eq 0 ]]; then
  echo "Nothing to prune."
  rm -f "$CURRENT" "$ALL_LARGE" "$TO_REMOVE"
  exit 0
fi

echo "Sample of paths to remove:"
head -30 "$TO_REMOVE"
if [[ $REMOVE_COUNT -gt 30 ]]; then
  echo "... and $((REMOVE_COUNT - 30)) more"
fi
echo ""

if [[ "$DRY_RUN" == true ]]; then
  echo "[DRY RUN] Would run: git filter-repo --invert-paths --paths-from-file $TO_REMOVE --force"
  echo "[DRY RUN] Full list: $TO_REMOVE"
  rm -f "$CURRENT" "$ALL_LARGE"
  exit 0
fi

if [[ "$YES" != true ]]; then
  echo "WARNING: This rewrites history. Backup first. Force-push required after."
  read -p "Continue? (y/N) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted. Path list saved to: $TO_REMOVE"
    rm -f "$CURRENT" "$ALL_LARGE"
    exit 1
  fi
fi

echo "Running git filter-repo..."
git filter-repo --invert-paths --paths-from-file "$TO_REMOVE" --force

rm -f "$CURRENT" "$ALL_LARGE" "$TO_REMOVE"

echo ""
echo "Done. Next steps:"
echo "  git push --force --all"
echo "  git push --force --tags   # CRITICAL: tags hold history; without this, clones stay huge"
echo "  git reflog expire --expire=now --all && git gc --prune=now --aggressive"
