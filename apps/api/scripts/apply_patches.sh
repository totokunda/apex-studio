#!/usr/bin/env bash
set -euo pipefail

# Compatibility entrypoint.
# The implementation lives under `scripts/updates/apply_patches.sh`.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/updates/apply_patches.sh" "$@"


