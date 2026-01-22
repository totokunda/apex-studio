#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR"
export PATH="$SCRIPT_DIR/apex-studio/bin:$PATH"
exec "$SCRIPT_DIR/apex-studio/bin/python" -m uvicorn src.api.main:app --host 127.0.0.1 --port 8765
