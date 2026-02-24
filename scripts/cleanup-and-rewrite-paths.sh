#!/usr/bin/env bash
#
# Comprehensive cleanup and path rewrite script for apex-studio
#
# Part 1: Git history cleanup
#   - Iterates through all git branches
#   - Finds files >100KB that are no longer present in the current workspace
#   - Prunes them from git history (requires git-filter-repo)
#
# Part 2: Models path rewrite
#   - Updates all references to model files from old long paths to current short paths
#   - Target: apps/app/packages/renderer/public/models/* -> models/*
#
# Usage:
#   ./scripts/cleanup-and-rewrite-paths.sh [--git-only] [--paths-only] [--dry-run] [--rewrite-history]
#
#   --rewrite-history  Replace model file blobs in history with current (smaller) versions
#                      (all commits, all branches). Requires git-filter-repo.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SIZE_THRESHOLD_KB=100
SIZE_THRESHOLD_BYTES=$((SIZE_THRESHOLD_KB * 1024))

# Parse arguments
GIT_ONLY=false
PATHS_ONLY=false
DRY_RUN=false
REWRITE_HISTORY=false
for arg in "$@"; do
  case "$arg" in
    --git-only)        GIT_ONLY=true ;;
    --paths-only)      PATHS_ONLY=true ;;
    --dry-run)         DRY_RUN=true ;;
    --rewrite-history) REWRITE_HISTORY=true ;;
    -h|--help)
      echo "Usage: $0 [--git-only] [--paths-only] [--dry-run] [--rewrite-history] [--prune-all]"
      echo "  --git-only        Only run git history cleanup (large deleted files)"
      echo "  --paths-only      Only run models path rewrite (current working tree)"
      echo "  --dry-run         Show what would be done without making changes"
      echo "  --rewrite-history Replace model blobs in history with current (smaller) versions"
      echo "  --prune-all       Prune ALL deleted files (not just >100KB). Use with caution."
      exit 0
      ;;
  esac
done

# =============================================================================
# Part 1: Git history cleanup - remove large deleted files from all branches
# =============================================================================
run_git_cleanup() {
  echo "=============================================="
  echo "Part 1: Git history cleanup"
  echo "=============================================="

  cd "$REPO_ROOT"

  if [[ ! -d .git ]]; then
    echo "Error: Not a git repository. Run from apex-studio root."
    exit 1
  fi

  # Check for git-filter-repo (recommended) or git filter-branch
  if command -v git-filter-repo &>/dev/null; then
    echo "Using git-filter-repo for history rewrite."
    USE_FILTER_REPO=true
  else
    echo "Warning: git-filter-repo not found. Install with: pip install git-filter-repo"
    echo "Falling back to generating path list only. Run git-filter-repo manually."
    USE_FILTER_REPO=false
  fi

  # Get all files currently in workspace (from HEAD of main)
  echo "Building list of files currently in workspace..."
  CURRENT_FILES=$(mktemp)
  git ls-tree -r HEAD --name-only 2>/dev/null | sort -u > "$CURRENT_FILES" || true

  # Also include files from current working tree (in case HEAD is behind)
  if [[ -d . ]]; then
    find . -type f 2>/dev/null | sed 's|^\./||' | grep -v '^\.git/' | sort -u >> "$CURRENT_FILES" || true
    sort -u "$CURRENT_FILES" -o "$CURRENT_FILES"
  fi

  # Find all paths in git history that are NOT in current workspace
  if [[ "$PRUNE_ALL" == true ]]; then
    echo "Scanning git history for ALL files not in workspace (--prune-all)..."
    THRESHOLD=0
  else
    echo "Scanning git history for large files (>${SIZE_THRESHOLD_KB}KB) not in workspace..."
    THRESHOLD="$SIZE_THRESHOLD_BYTES"
  fi
  PATHS_TO_REMOVE=$(mktemp)

  # Get all blobs (optionally > 100KB) and their paths across all branches
  git rev-list --all --objects 2>/dev/null | \
    git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' 2>/dev/null | \
    awk -v threshold="$THRESHOLD" '
      /^blob / {
        if ($3 > threshold && $4 != "") {
          path = $4
          for (i=5; i<=NF; i++) path = path " " $i
          print path
        }
      }
    ' | sort -u > "${PATHS_TO_REMOVE}.all" 2>/dev/null || true

  # Filter to only paths NOT in current workspace
  if [[ -s "${PATHS_TO_REMOVE}.all" ]]; then
    comm -23 "${PATHS_TO_REMOVE}.all" "$CURRENT_FILES" > "$PATHS_TO_REMOVE" 2>/dev/null || cp "${PATHS_TO_REMOVE}.all" "$PATHS_TO_REMOVE"
  else
    touch "$PATHS_TO_REMOVE"
  fi

  REMOVE_COUNT=$(wc -l < "$PATHS_TO_REMOVE" 2>/dev/null || echo 0)
  echo "Found $REMOVE_COUNT paths to remove from git history (large files no longer in workspace)."

  if [[ $REMOVE_COUNT -gt 0 ]]; then
    echo "Sample of paths to remove:"
    head -20 "$PATHS_TO_REMOVE"
    if [[ $REMOVE_COUNT -gt 20 ]]; then
      echo "... and $((REMOVE_COUNT - 20)) more"
    fi

    if [[ "$DRY_RUN" == true ]]; then
      echo ""
      echo "[DRY RUN] Would run: git filter-repo --invert-paths --paths-from-file $PATHS_TO_REMOVE"
      echo "[DRY RUN] Full list saved to: $PATHS_TO_REMOVE"
    elif [[ "$USE_FILTER_REPO" == true ]]; then
      echo ""
      echo "Running git-filter-repo to prune these paths from history..."
      echo "WARNING: This rewrites history. Ensure you have a backup. Force-push will be required."
      read -p "Continue? (y/N) " -n 1 -r
      echo
      if [[ $REPLY =~ ^[Yy]$ ]]; then
        git filter-repo --invert-paths --paths-from-file "$PATHS_TO_REMOVE" --force
        echo "Done. Run 'git push --force --all' to update remote."
      else
        echo "Aborted. Path list saved to: $PATHS_TO_REMOVE"
      fi
    else
      echo ""
      echo "To complete manually, run:"
      echo "  git filter-repo --invert-paths --paths-from-file $PATHS_TO_REMOVE --force"
      echo "  git push --force --all"
    fi
  else
    echo "No large deleted files to prune."
  fi

  rm -f "$CURRENT_FILES" "${PATHS_TO_REMOVE}.all"
  if [[ "$USE_FILTER_REPO" == false ]] && [[ $REMOVE_COUNT -gt 0 ]]; then
    echo "Path list saved to: $PATHS_TO_REMOVE (not deleted for manual use)"
  else
    rm -f "$PATHS_TO_REMOVE"
  fi
}

# =============================================================================
# Part 2: Models path rewrite - update old paths to current short format
# Searches through ALL git branches in history to find every file that needs rewriting
# =============================================================================
run_path_rewrite() {
  echo ""
  echo "=============================================="
  echo "Part 2: Models path rewrite"
  echo "=============================================="

  MODELS_DIR="$REPO_ROOT/apps/app/packages/renderer/public/models"
  if [[ ! -d "$MODELS_DIR" ]]; then
    echo "Error: Models directory not found: $MODELS_DIR"
    exit 1
  fi

  # Step 1: Search through ALL branches in git history for files containing old path patterns
  echo "Searching git history across all branches for files with old model paths..."
  FILES_TO_UPDATE=$(mktemp)

  # Patterns that indicate old/long paths (current format is "models/filename.ext")
  OLD_PATTERNS=(
    "apps/app/packages/renderer/public/models"
    "packages/renderer/public/models"
    "renderer/public/models"
    "app/packages/renderer/public/models"
  )

  # Get unique branch/ref names (local + remote, no HEAD)
  REFS=$(git branch -a 2>/dev/null | sed 's/^[* ]*//' | sed 's|remotes/||' | grep -v HEAD | sort -u)

  for ref in $REFS; do
    for pattern in "${OLD_PATTERNS[@]}"; do
      (git grep -l -F "$pattern" "$ref" 2>/dev/null || true) | cut -d: -f2-
    done
  done | sort -u > "$FILES_TO_UPDATE"

  # Also search current working tree (in case of uncommitted or new files)
  grep -r -l -E "apps/app/packages/renderer/public/models|packages/renderer/public/models|renderer/public/models|app/packages/renderer/public/models" \
    --include="*.yml" --include="*.yaml" --include="*.ts" --include="*.tsx" --include="*.js" --include="*.cjs" \
    --include="*.json" --include="*.md" --include="*.html" \
    "$REPO_ROOT" 2>/dev/null | grep -v node_modules | grep -v "/.git/" | grep -v "/dist/" | grep -v ".egg-info" >> "$FILES_TO_UPDATE" || true

  # Add ALL manifest files that have demo_path (they may need path normalization)
  find "$REPO_ROOT/apps/api/manifest" -type f \( -name "*.yml" -o -name "*.yaml" \) 2>/dev/null >> "$FILES_TO_UPDATE" || true

  # Add known files that reference renderer paths (from branch search)
  echo ".gitmodules" >> "$FILES_TO_UPDATE"
  echo "apps/app/packages/preload/src/filters/fetch.ts" >> "$FILES_TO_UPDATE"
  echo "apps/app/electron-builder.cjs" >> "$FILES_TO_UPDATE"
  echo ".gitignore" >> "$FILES_TO_UPDATE"

  sort -u "$FILES_TO_UPDATE" -o "$FILES_TO_UPDATE"
  FILE_COUNT=$(wc -l < "$FILES_TO_UPDATE")
  echo "Found $FILE_COUNT candidate files to check."

  # Step 2: Rewrite files - replace old long paths with "models/" (current short format)
  # Skip .gitmodules and fetch.ts - they use full paths for filesystem resolution, not model URLs
  SKIP_FILES=".gitmodules|fetch\.ts|electron-builder|\.gitignore"
  UPDATED_COUNT=0

  while IFS= read -r file; do
    [[ -f "$file" ]] || continue
    # Skip binary and generated files
    file "$file" 2>/dev/null | grep -qE "text|empty" || continue
    # Skip files that need full paths for resolution
    [[ "$file" == *".gitmodules"* ]] && continue
    [[ "$file" == *"fetch.ts"* ]] && continue
    [[ "$file" == *"electron-builder"* ]] && continue
    [[ "$file" == *".gitignore"* ]] && continue

    # Check if file contains any old path pattern that should be rewritten
    CHANGED=false
    if grep -qE "apps/app/packages/renderer/public/models|packages/renderer/public/models|renderer/public/models|app/packages/renderer/public/models" "$file" 2>/dev/null; then
      CHANGED=true
    fi
    # Also check demo_path with long paths (any path longer than "models/")
    if grep -qE "demo_path:.*(apps/|packages/|renderer/public|/renderer/)" "$file" 2>/dev/null; then
      CHANGED=true
    fi

    if [[ "$CHANGED" == true ]]; then
      echo "Updating: $file"
      if [[ "$DRY_RUN" != true ]]; then
        # Replace from longest to shortest to avoid partial replacements
        sed -i 's|apps/app/packages/renderer/public/models/|models/|g' "$file"
        sed -i 's|apps/app/packages/renderer/public/models|models|g' "$file"
        sed -i 's|packages/renderer/public/models/|models/|g' "$file"
        sed -i 's|app/packages/renderer/public/models/|models/|g' "$file"
        sed -i 's|renderer/public/models/|models/|g' "$file"
        sed -i 's|public/models/|models/|g' "$file"
        # Fix demo_path: strip any leading path to leave models/filename
        sed -i -E 's|demo_path: *[^m]*models/|demo_path: models/|g' "$file"
        UPDATED_COUNT=$((UPDATED_COUNT + 1))
      else
        echo "  [DRY RUN] Would replace old path patterns with 'models/'"
        UPDATED_COUNT=$((UPDATED_COUNT + 1))
      fi
    fi
  done < "$FILES_TO_UPDATE"

  echo "Path rewrite complete. Updated $UPDATED_COUNT files."

  # Step 3: Process ALL manifest files - ensure demo_path uses "models/xxx" and fix extensions
  echo ""
  echo "Processing all manifest files in apps/api/manifest..."
  MANIFEST_COUNT=0
  for yml in $(find "$REPO_ROOT/apps/api/manifest" -type f \( -name "*.yml" -o -name "*.yaml" \) 2>/dev/null); do
    [[ -f "$yml" ]] || continue
    demo_path=$(grep -E "^  demo_path:" "$yml" 2>/dev/null | sed 's/.*demo_path: *//' | tr -d '"' | tr -d "'" | xargs)
    [[ -z "$demo_path" ]] && continue
    MANIFEST_COUNT=$((MANIFEST_COUNT + 1))

    # Normalize: ensure demo_path starts with models/ (strip any leading path)
    if [[ "$demo_path" != models/* ]] && [[ "$demo_path" == *models/* ]]; then
      new_demo="models/${demo_path#*models/}"
      if [[ "$DRY_RUN" != true ]]; then
        sed -i "s|demo_path: *$demo_path|demo_path: $new_demo|g" "$yml"
        echo "  Normalized: $demo_path -> $new_demo (in $(basename "$yml"))"
      fi
      demo_path="$new_demo"
    fi

    # Validate: check if referenced file exists
    if [[ "$demo_path" == models/* ]]; then
      model_file="$MODELS_DIR/${demo_path#models/}"
      if [[ -f "$model_file" ]]; then
        continue
      fi
      # Try alternate extensions (e.g. .mov when .mp4 is referenced)
      base="${model_file%.*}"
      found=""
      for ext in .mov .mp4 .webm .png .jpg .jpeg .gif .webp; do
        if [[ -f "${base}${ext}" ]]; then
          found="${base}${ext}"
          break
        fi
      done
      if [[ -n "$found" ]] && [[ "$DRY_RUN" != true ]]; then
        new_demo="models/$(basename "$found")"
        echo "  Fixing extension: $demo_path -> $new_demo (in $(basename "$yml"))"
        sed -i "s|demo_path: *$demo_path|demo_path: $new_demo|g" "$yml"
      elif [[ -z "$found" ]]; then
        echo "  Missing: $demo_path (referenced in $(basename "$yml"))"
      fi
    fi
  done 2>/dev/null || true
  echo "  Processed $MANIFEST_COUNT manifest files with demo_path."

  rm -f "$FILES_TO_UPDATE"
}

# =============================================================================
# Part 3: Rewrite model files in history - replace old large blobs with current smaller ones
# Uses git filter-repo with commit callback to replace blob IDs for model files.
# =============================================================================
run_history_rewrite() {
  echo ""
  echo "=============================================="
  echo "Part 3: Replacing model files in history with current (smaller) versions"
  echo "=============================================="

  cd "$REPO_ROOT"
  MODELS_SRC="$REPO_ROOT/apps/app/packages/renderer/public/models"
  MODELS_GIT_PREFIX="apps/app/packages/renderer/public/models/"

  if [[ ! -d "$MODELS_SRC" ]]; then
    echo "Error: Models directory not found: $MODELS_SRC"
    exit 1
  fi

  if ! command -v git-filter-repo &>/dev/null; then
    echo "Error: git-filter-repo required. Install with: pip install git-filter-repo"
    exit 1
  fi

  # filter-repo works best with clean state; warn if dirty
  if [[ -n $(git status --porcelain 2>/dev/null | grep -v '^??') ]]; then
    echo "Warning: You have uncommitted changes. Stash or commit them first:"
    echo "  git stash -u   # or git add -A && git commit -m 'WIP'"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && echo "Aborted." && return
  fi

  echo "Source (current smaller files): $MODELS_SRC"
  echo "Target path in repo: $MODELS_GIT_PREFIX"
  echo ""
  echo "This will replace the model file contents in EVERY commit with the current"
  echo "versions from your workspace (shrinking the repo by removing old large blobs)."
  echo ""

  if [[ "$DRY_RUN" == true ]]; then
    echo "[DRY RUN] Would run: git filter-repo with commit callback to replace model blobs"
    echo "[DRY RUN] Backup first. Force-push required after."
    return
  fi

  echo "WARNING: This rewrites git history. Create a backup first (e.g. git clone --mirror)."
  echo "Force-push will be required: git push --force --all"
  read -p "Continue? (y/N) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    return
  fi

  # Build blob map: create new blobs for each current model file
  echo "Creating blobs for current model files..."
  BLOB_MAP_FILE=$(mktemp)
  for f in "$MODELS_SRC"/*; do
    [[ -f "$f" ]] || continue
    name=$(basename "$f")
    blob=$(git hash-object -w "$f")
    echo "${MODELS_GIT_PREFIX}${name} $blob" >> "$BLOB_MAP_FILE"
  done
  MAP_COUNT=$(wc -l < "$BLOB_MAP_FILE")
  echo "  Mapped $MAP_COUNT model files to new blobs"

  # Build Python callback file (filter-repo accepts path to file)
  CALLBACK_FILE=$(mktemp --suffix=.py)
  {
    echo "blob_map = {"
    while read -r path blob; do
      # Escape backslash and single quote for Python string
      path_escaped="${path//\\/\\\\}"
      path_escaped="${path_escaped//\'/\\\\\'}"
      echo "    b'$path_escaped': b'$blob',"
    done < "$BLOB_MAP_FILE"
    echo "}"
    echo "for change in commit.file_changes:"
    echo "    if change.type != b'D' and change.filename in blob_map:"
    echo "        change.blob_id = blob_map[change.filename]"
  } > "$CALLBACK_FILE"

  echo "Running git filter-repo (this may take a while)..."
  git filter-repo --commit-callback "$CALLBACK_FILE" --force

  rm -f "$BLOB_MAP_FILE" "$CALLBACK_FILE"

  echo ""
  echo "Done. Run 'git push --force --all' to update remote."
  echo "Then run 'git reflog expire --expire=now --all && git gc --prune=now --aggressive' to reclaim space."
}

# =============================================================================
# Main
# =============================================================================
main() {
  echo "Apex Studio - Cleanup and Path Rewrite Script"
  echo "Repository: $REPO_ROOT"
  echo ""

  if [[ "$PATHS_ONLY" != true ]]; then
    run_git_cleanup
  fi

  if [[ "$GIT_ONLY" != true ]]; then
    run_path_rewrite
  fi

  if [[ "$REWRITE_HISTORY" == true ]]; then
    run_history_rewrite
  fi

  echo ""
  echo "Done."
}

main "$@"
