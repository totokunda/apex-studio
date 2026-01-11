#!/usr/bin/env bash
set -euo pipefail

# Compatibility entrypoint.
# The implementation lives under `scripts/setup/install.sh`.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/setup/install.sh" "$@"


