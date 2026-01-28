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

CAPTION_GLOB="${CAPTION_GLOB:-*}"
CAPTION_MODEL="${CAPTION_MODEL:-fancyfeast/llama-joycaption-beta-one-hf-llava}"
CAPTION_PROMPT="${CAPTION_PROMPT:-Write a brief caption for this image in a formal tone.}"
CAPTION_MAX_NEW_TOKENS="${CAPTION_MAX_NEW_TOKENS:-512}"

"$PYTHON" "$SCRIPT_DIR/caption.py" \
  --dataset-dir "$DATASET_DIR" \
  --out-csv "$CAPTIONS_CSV" \
  --glob "$CAPTION_GLOB" \
  --model "$CAPTION_MODEL" \
  --prompt "$CAPTION_PROMPT" \
  --max-new-tokens "$CAPTION_MAX_NEW_TOKENS"

