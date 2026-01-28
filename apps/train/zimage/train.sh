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

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Defaults (override by exporting env vars before running this script)
CAPTIONS_CSV="${CAPTIONS_CSV:-$SCRIPT_DIR/captions.csv}"
TRAINING_INPUTS_DIR="${TRAINING_INPUTS_DIR:-$SCRIPT_DIR/training_inputs}"
OPTIMIZER="${OPTIMIZER:-adamw}"
RUN_NAME="${RUN_NAME:-run}"
MAX_STEPS="${MAX_STEPS:-5000}"

$PYTHON "$SCRIPT_DIR/train.py" \
  --vae_encodings "$TRAINING_INPUTS_DIR/vae_encodings.safetensors" \
  --text_encodings "$TRAINING_INPUTS_DIR/text_encodings.safetensors" \
  --captions_csv "$CAPTIONS_CSV" \
  --caption_dropout 0.05 \
  --lora_rank 32 \
  --learning_rate 1e-4 \
  --optimizer "$OPTIMIZER" \
  --batch_size 1 \
  --mixed_precision bf16 \
  --gradient_accumulation_steps 4 \
  --max_steps "$MAX_STEPS" \
  --gradient_checkpointing \
  --run_name "$RUN_NAME" \
  --save_every 250