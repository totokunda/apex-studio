#!/usr/bin/env bash
set -euo pipefail

# End-to-end Z-Image data prep + training.
#
# Stages:
#  1) Caption images -> captions.csv
#  2) Text-encode captions.csv -> training_inputs/text_encodings.safetensors
#  3) VAE-encode images in captions.csv -> training_inputs/vae_encodings.safetensors
#  4) Train LoRA -> lora/<run_name>/
#
# All artifacts default to this folder (`apps/train/zimage/`) for simplicity.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

PYTHON="$SCRIPT_DIR/../../api/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="$SCRIPT_DIR/.venv/bin/python"
fi
if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Required: where your images live
DATASET_DIR="${DATASET_DIR:-$SCRIPT_DIR/datasets}"

# Outputs (kept under this folder by default)
CAPTIONS_CSV="${CAPTIONS_CSV:-$SCRIPT_DIR/captions.csv}"
TRAINING_INPUTS_DIR="${TRAINING_INPUTS_DIR:-$SCRIPT_DIR/training_inputs}"

# Captioning controls
CAPTION_GLOB="${CAPTION_GLOB:-*}"
CAPTION_MODEL="${CAPTION_MODEL:-fancyfeast/llama-joycaption-beta-one-hf-llava}"
CAPTION_PROMPT="${CAPTION_PROMPT:-Write a brief caption for this image in a formal tone.}"
CAPTION_MAX_NEW_TOKENS="${CAPTION_MAX_NEW_TOKENS:-512}"

# Encoding controls
MANIFEST_YAML="${MANIFEST_YAML:-$SCRIPT_DIR/../../api/manifest/image/zimage-1.0.0.v1.yml}"
TEXT_DEVICE="${TEXT_DEVICE:-cuda}"
VAE_MAX_AREA="${VAE_MAX_AREA:-921600}" # 720*1280
VAE_MOD_VALUE="${VAE_MOD_VALUE:-16}"
VAE_OFFLOAD="${VAE_OFFLOAD:-0}"

# Training controls
RUN_NAME="${RUN_NAME:-run}"
MAX_STEPS="${MAX_STEPS:-5000}"
OPTIMIZER="${OPTIMIZER:-adamw}"

# Stage toggles (set to 1 to skip)
SKIP_CAPTION="${SKIP_CAPTION:-0}"
SKIP_TEXT_ENCODE="${SKIP_TEXT_ENCODE:-0}"
SKIP_VAE_ENCODE="${SKIP_VAE_ENCODE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

echo "[run_all] python: $PYTHON"
echo "[run_all] dataset: $DATASET_DIR"
echo "[run_all] captions_csv: $CAPTIONS_CSV"
echo "[run_all] training_inputs: $TRAINING_INPUTS_DIR"
echo "[run_all] run_name: $RUN_NAME"

mkdir -p "$TRAINING_INPUTS_DIR"

if [ "$SKIP_CAPTION" != "1" ]; then
  echo "[run_all] stage: caption"
  "$PYTHON" "$SCRIPT_DIR/caption.py" \
    --dataset-dir "$DATASET_DIR" \
    --out-csv "$CAPTIONS_CSV" \
    --glob "$CAPTION_GLOB" \
    --model "$CAPTION_MODEL" \
    --prompt "$CAPTION_PROMPT" \
    --max-new-tokens "$CAPTION_MAX_NEW_TOKENS"
else
  echo "[run_all] stage: caption (skipped)"
fi

if [ "$SKIP_TEXT_ENCODE" != "1" ]; then
  echo "[run_all] stage: text_encode"
  "$PYTHON" "$SCRIPT_DIR/text_encode.py" \
    --dataset-dir "$DATASET_DIR" \
    --captions-csv "$CAPTIONS_CSV" \
    --out-dir "$TRAINING_INPUTS_DIR" \
    --out-file "text_encodings.safetensors" \
    --device "$TEXT_DEVICE" \
    --yaml-path "$MANIFEST_YAML"
else
  echo "[run_all] stage: text_encode (skipped)"
fi

if [ "$SKIP_VAE_ENCODE" != "1" ]; then
  echo "[run_all] stage: vae_encode"
  VAE_OFFLOAD_FLAG=()
  if [ "$VAE_OFFLOAD" = "1" ]; then
    VAE_OFFLOAD_FLAG=(--vae-offload)
  fi
  "$PYTHON" "$SCRIPT_DIR/vae_encode.py" \
    --dataset-dir "$DATASET_DIR" \
    --captions-csv "$CAPTIONS_CSV" \
    --out-dir "$TRAINING_INPUTS_DIR" \
    --out-file "vae_encodings.safetensors" \
    --yaml-path "$MANIFEST_YAML" \
    --max-area "$VAE_MAX_AREA" \
    --mod-value "$VAE_MOD_VALUE" \
    "${VAE_OFFLOAD_FLAG[@]}"
else
  echo "[run_all] stage: vae_encode (skipped)"
fi

if [ "$SKIP_TRAIN" != "1" ]; then
  echo "[run_all] stage: train"
  CAPTIONS_CSV="$CAPTIONS_CSV" \
  TRAINING_INPUTS_DIR="$TRAINING_INPUTS_DIR" \
  RUN_NAME="$RUN_NAME" \
  MAX_STEPS="$MAX_STEPS" \
  OPTIMIZER="$OPTIMIZER" \
    bash "$SCRIPT_DIR/train.sh"
else
  echo "[run_all] stage: train (skipped)"
fi

echo "[run_all] done"

