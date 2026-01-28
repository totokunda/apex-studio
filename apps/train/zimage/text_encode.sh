#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

PYTHON="$SCRIPT_DIR/../../api/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="$SCRIPT_DIR/.venv/bin/python"
fi
if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi

# Defaults (override by exporting env vars before running)
DATASET_DIR="${DATASET_DIR:-$SCRIPT_DIR/datasets}"
CAPTIONS_CSV="${CAPTIONS_CSV:-$SCRIPT_DIR/captions.csv}"
TRAINING_INPUTS_DIR="${TRAINING_INPUTS_DIR:-$SCRIPT_DIR/training_inputs}"
MANIFEST_YAML="${MANIFEST_YAML:-$SCRIPT_DIR/../../api/manifest/image/zimage-1.0.0.v1.yml}"
TEXT_DEVICE="${TEXT_DEVICE:-cuda}"
TEXT_OUT_FILE="${TEXT_OUT_FILE:-text_encodings.safetensors}"

mkdir -p "$TRAINING_INPUTS_DIR"

"$PYTHON" "$SCRIPT_DIR/text_encode.py" \
  --dataset-dir "$DATASET_DIR" \
  --captions-csv "$CAPTIONS_CSV" \
  --out-dir "$TRAINING_INPUTS_DIR" \
  --out-file "$TEXT_OUT_FILE" \
  --device "$TEXT_DEVICE" \
  --yaml-path "$MANIFEST_YAML"

