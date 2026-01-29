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
VAE_OUT_FILE="${VAE_OUT_FILE:-vae_encodings.safetensors}"

VAE_MAX_AREA="${VAE_MAX_AREA:-1572864}" # 1024*1024*1.5
VAE_MOD_VALUE="${VAE_MOD_VALUE:-16}"
VAE_OFFLOAD="${VAE_OFFLOAD:-0}"

mkdir -p "$TRAINING_INPUTS_DIR"

VAE_OFFLOAD_FLAG=()
if [ "$VAE_OFFLOAD" = "1" ]; then
  VAE_OFFLOAD_FLAG=(--vae-offload)
fi

"$PYTHON" "$SCRIPT_DIR/vae_encode.py" \
  --dataset-dir "$DATASET_DIR" \
  --captions-csv "$CAPTIONS_CSV" \
  --out-dir "$TRAINING_INPUTS_DIR" \
  --out-file "$VAE_OUT_FILE" \
  --yaml-path "$MANIFEST_YAML" \
  --max-area "$VAE_MAX_AREA" \
  --mod-value "$VAE_MOD_VALUE" \
  "${VAE_OFFLOAD_FLAG[@]}"

