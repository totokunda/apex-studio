#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/apply_patches.sh [options]

Apply all .patch files from a patches directory to the repository working tree.

Options:
  --dir <dir>      Path to patches directory (default: patches)
  --dry-run | -n   Validate patches without applying changes
  --reverse | -R   Reverse-apply patches (unapply)
  --3way           Use 3-way merge when applying with git (default)
  --no-3way        Disable 3-way merge for git apply
  --python-cmd CMD Python command to resolve site-packages (e.g. "conda run -n ENV python")
  -h | --help      Show this help and exit

Environment:
  PATCHES_DIR      Same as --dir

Notes:
  - Only files ending with .patch are processed; other files are ignored.
  - If inside a git repo, uses 'git apply' (with optional 3-way). If that fails
    or not in a git repo, falls back to the 'patch' command with -p1.
  - In --dry-run mode, the script exits non-zero if any patch would fail.
EOF
}

PATCHES_DIR=${PATCHES_DIR:-patches}
DRY_RUN=false
REVERSE=false
THREE_WAY=true
PYTHON_CMD=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)
      [[ $# -ge 2 ]] || { echo "Missing value for --dir" >&2; exit 1; }
      PATCHES_DIR="$2"; shift 2 ;;
    --dry-run|-n)
      DRY_RUN=true; shift ;;
    --reverse|-R)
      REVERSE=true; shift ;;
    --3way)
      THREE_WAY=true; shift ;;
    --no-3way)
      THREE_WAY=false; shift ;;
    --python-cmd)
      [[ $# -ge 2 ]] || { echo "Missing value for --python-cmd" >&2; exit 1; }
      PYTHON_CMD="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1 ;;
  esac
done

# Move to repo root if in a git repository
ROOT_DIR=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$ROOT_DIR"

if [[ ! -d "$PATCHES_DIR" ]]; then
  echo "Patches directory not found: $PATCHES_DIR" >&2
  exit 1
fi

# Collect .patch files only, sorted for deterministic order
mapfile -t PATCH_FILES < <(find "$PATCHES_DIR" -maxdepth 1 -type f -name "*.patch" | sort)

if [[ ${#PATCH_FILES[@]} -eq 0 ]]; then
  echo "No .patch files found in $PATCHES_DIR. Nothing to do."
  exit 0
fi

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  IN_GIT=true
else
  IN_GIT=false
fi

# Choose a python command if not provided, used to locate site-packages for third-party packages
if [[ -z "$PYTHON_CMD" ]]; then
  # Prefer current conda environment python if available
  if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON_CMD="$CONDA_PREFIX/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
  else
    PYTHON_CMD=""
  fi
fi

apply_with_git() {
  local patch_file="$1"
  local args=()
  $THREE_WAY && args+=(--3way)
  $REVERSE && args+=(-R)
  if $DRY_RUN; then
    git apply --check "${args[@]}" "$patch_file"
  else
    git apply "${args[@]}" "$patch_file"
  fi
}

apply_with_patch_cmd() {
  local patch_file="$1"
  # Non-interactive + idempotent:
  # - --batch: never prompt (safe for CI / unattended runs)
  # - -N: ignore patches already applied / reversed (best-effort idempotency)
  local args=(--batch -N -p1)
  $REVERSE && args+=(-R)
  if $DRY_RUN; then
    patch --dry-run "${args[@]}" < "$patch_file"
  else
    patch "${args[@]}" < "$patch_file"
  fi
}

success_count=0
failure_count=0

echo "Using patches directory: $PATCHES_DIR"
echo "Mode: ${DRY_RUN:+dry-run }${REVERSE:+reverse }${THREE_WAY:+3-way }apply"
[[ -n "$PYTHON_CMD" ]] && echo "Python resolver: $PYTHON_CMD"

# Determine the base directory to apply a patch from its first file path
determine_base_dir() {
  local patch_file="$1"
  # Extract first path from 'diff --git a/xxx b/xxx'
  local first_line
  first_line=$(grep -m1 '^diff --git ' "$patch_file" || true)
  if [[ -z "$first_line" ]]; then
    echo "$ROOT_DIR"; return 0
  fi
  # Get the 'a/xxx' token and strip leading 'a/'
  local a_path
  a_path=$(echo "$first_line" | awk '{print $3}')
  a_path=${a_path#a/}
  # Top-level segment before '/'
  local top_level="${a_path%%/*}"

  # If path already starts with 'thirdparty/', stay at repo root
  if [[ "$a_path" == thirdparty/* ]]; then
    echo "$ROOT_DIR"; return 0
  fi

  # Try to resolve as a Python package (e.g., huggingface_hub, sam2)
  if [[ -n "$PYTHON_CMD" && -n "$top_level" ]]; then
    local site_parent
    # The parent of the package directory contains the top-level path (e.g., site-packages)
    site_parent=$($PYTHON_CMD - <<PYCODE 2>/dev/null || true
import importlib, pathlib, sys
name = "$top_level"
try:
    m = importlib.import_module(name)
    p = pathlib.Path(m.__file__).resolve()
    # Return the directory containing the top-level package directory
    print(str(p.parent.parent if p.is_file() else p.parent))
except Exception:
    pass
PYCODE
)
    if [[ -n "$site_parent" && -d "$site_parent/$top_level" ]]; then
      echo "$site_parent"; return 0
    fi
  fi

  # Default to repo root
  echo "$ROOT_DIR"
}

first_patch_a_path() {
  local patch_file="$1"
  local first_line
  first_line=$(grep -m1 '^diff --git ' "$patch_file" || true)
  if [[ -z "$first_line" ]]; then
    echo ""
    return 0
  fi
  local a_path
  a_path=$(echo "$first_line" | awk '{print $3}')
  a_path=${a_path#a/}
  echo "$a_path"
}

is_python_ident() {
  local s="$1"
  [[ "$s" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]
}

apply_in_dir() {
  local base_dir="$1"
  local patch_file="$2"
  # Ensure patch file path is absolute so we can read it after changing directories
  local abs_patch_file
  if [[ "$patch_file" = /* ]]; then
    abs_patch_file="$patch_file"
  else
    abs_patch_file="$ROOT_DIR/$patch_file"
  fi
  pushd "$base_dir" >/dev/null
  # Prefer git apply only if base_dir is inside a git work tree
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if apply_with_git "$abs_patch_file"; then
      popd >/dev/null
      return 0
    fi
    echo "git apply failed in $base_dir, attempting 'patch' fallback..."
  fi
  if apply_with_patch_cmd "$abs_patch_file"; then
    popd >/dev/null
    return 0
  fi
  popd >/dev/null
  return 1
}

for patch_path in "${PATCH_FILES[@]}"; do
  patch_name=$(basename "$patch_path")
  echo "Applying $patch_name ..."
  base_dir=$(determine_base_dir "$patch_path")
  echo " → Base directory resolved to: $base_dir"

  # If this patch looks like it targets a third-party Python package (top-level dir is a
  # Python identifier), but we couldn't resolve it into site-packages AND the target file
  # doesn't exist in the repo, skip it gracefully. This keeps CPU/mac installs working
  # when optional deps (e.g. xformers) aren't installed.
  a_path=$(first_patch_a_path "$patch_path")
  top_level="${a_path%%/*}"
  if [[ -n "$PYTHON_CMD" && "$base_dir" == "$ROOT_DIR" && -n "$a_path" && -n "$top_level" ]]; then
    if is_python_ident "$top_level" && [[ "$a_path" == "$top_level/"* ]]; then
      if [[ ! -e "$ROOT_DIR/$a_path" ]] && ! grep -q '^new file mode ' "$patch_path"; then
        echo " ⚠ Skipping $patch_name (package '$top_level' not importable in python-cmd; target not found)"
        ((success_count++))
        continue
      fi
    fi
  fi

  if apply_in_dir "$base_dir" "$patch_path"; then
    ((success_count++))
    echo "✔ Applied $patch_name in $base_dir"
  else
    ((failure_count++))
    echo "✖ Failed to apply $patch_name in $base_dir" >&2
  fi
done

echo "Summary: ${success_count} applied, ${failure_count} failed${DRY_RUN:+ (dry-run)}"

if [[ $failure_count -gt 0 ]]; then
  exit 1
fi

exit 0


