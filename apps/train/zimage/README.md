# Z-Image training (caption → encode → train)

This folder contains a simple end-to-end pipeline for training a Z-Image LoRA using:

- **Captioning**: `caption.py` → produces `captions.csv`
- **Text encoding**: `text_encode.py` → produces `training_inputs/text_encodings.safetensors`
- **VAE encoding**: `vae_encode.py` → produces `training_inputs/vae_encodings.safetensors`
- **Training**: `train.py` (wrapped by `train.sh`) → writes `lora/<run_name>/...`

All default outputs are kept **under this folder** for convenience.

## Quick start (run everything)

From the repo root:

```bash
DATASET_DIR="/path/to/your/images" RUN_NAME="my_run" bash apps/train/zimage/run_all.sh
```

On Windows (Command Prompt):

```bat
set DATASET_DIR=C:\path\to\your\images
set RUN_NAME=my_run
call apps\train\zimage\run_all.bat
```

Common options:

```bash
DATASET_DIR="/path/to/your/images" \
RUN_NAME="ss2" \
OPTIMIZER="adamw8bit" \
MAX_STEPS=8000 \
bash apps/train/zimage/run_all.sh
```

On Windows (Command Prompt):

```bat
set DATASET_DIR=C:\path\to\your\images
set RUN_NAME=ss2
set OPTIMIZER=adamw8bit
set MAX_STEPS=8000
call apps\train\zimage\run_all.bat
```

Skip stages if you already have artifacts:

```bash
DATASET_DIR="/path/to/your/images" \
SKIP_CAPTION=1 \
SKIP_TEXT_ENCODE=1 \
SKIP_VAE_ENCODE=1 \
bash apps/train/zimage/run_all.sh
```

On Windows (Command Prompt):

```bat
set DATASET_DIR=C:\path\to\your\images
set SKIP_CAPTION=1
set SKIP_TEXT_ENCODE=1
set SKIP_VAE_ENCODE=1
call apps\train\zimage\run_all.bat
```

## Outputs (defaults)

- **Captions CSV**: `apps/train/zimage/captions.csv`
- **Encodings**: `apps/train/zimage/training_inputs/`
  - `text_encodings.safetensors`
  - `vae_encodings.safetensors`
- **Training outputs**: `apps/train/zimage/lora/<run_name>/`

You can override paths with:

- `CAPTIONS_CSV` (default: `apps/train/zimage/captions.csv`)
- `TRAINING_INPUTS_DIR` (default: `apps/train/zimage/training_inputs`)

## Running stages individually

### 1) Caption images

```bash
python3 apps/train/zimage/caption.py \
  --dataset-dir "/path/to/your/images" \
  --out-csv "./captions.csv" \
  --glob "*" 
```

Or via the wrappers:

```bash
DATASET_DIR="/path/to/your/images" bash ./caption.sh
```

```bat
set DATASET_DIR=C:\path\to\your\images
call .\caption.bat
```

### 2) Text encode

```bash
python3 apps/train/zimage/text_encode.py \
  --dataset-dir "/path/to/your/images" \
  --captions-csv "./captions.csv" \
  --out-dir "apps/train/zimage/training_inputs" \
  --yaml-path "apps/api/manifest/image/zimage-1.0.0.v1.yml"
```

Or via the wrappers:

```bash
DATASET_DIR="/path/to/your/images" bash apps/train/zimage/text_encode.sh
```

```bat
set DATASET_DIR=C:\path\to\your\images
call apps\train\zimage\text_encode.bat
```

### 3) VAE encode

```bash
python3 apps/train/zimage/vae_encode.py \
  --dataset-dir "/path/to/your/images" \
  --captions-csv "./captions.csv" \
  --out-dir "./training_inputs" \
  --yaml-path "apps/api/manifest/image/zimage-1.0.0.v1.yml"
```

Or via the wrappers:

```bash
DATASET_DIR="/path/to/your/images" bash ./vae_encode.sh
```

```bat
set DATASET_DIR=C:\path\to\your\images
call .\vae_encode.bat
```

### 4) Train

The simplest way is via `train.sh` (it is location-independent):

```bash
RUN_NAME="my_run" MAX_STEPS=5000 OPTIMIZER="adamw" bash ./train.sh
```

On Windows (Command Prompt):

```bat
set RUN_NAME=my_run
set MAX_STEPS=5000
set OPTIMIZER=adamw
call .\train.bat
```

`train.sh` expects:

- `CAPTIONS_CSV` (defaults to `./captions.csv`)
- `TRAINING_INPUTS_DIR` (defaults to `./training_inputs`)

## Notes

- `run_all.sh` attempts to pick a Python interpreter in this order:
  - `apps/api/.venv/bin/python`
  - `./venv/bin/python`
  - `python3`
- `run_all.bat` attempts to pick a Python interpreter in this order:
  - `apps\api\.venv\Scripts\python.exe`
  - `.\venv\Scripts\python.exe`
  - `python`
- If you want VAE offloading during encoding, set `VAE_OFFLOAD=1` when running `run_all.sh`.
  - On Windows, set `VAE_OFFLOAD=1` before running `run_all.bat`.

