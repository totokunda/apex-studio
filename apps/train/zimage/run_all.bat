@echo off
setlocal enableextensions enabledelayedexpansion

rem End-to-end Z-Image pipeline for Windows:
rem  1) caption.py -> captions.csv
rem  2) text_encode.py -> training_inputs\text_encodings.safetensors
rem  3) vae_encode.py -> training_inputs\vae_encodings.safetensors
rem  4) train.py (via train.bat) -> lora\<run_name>\

set "SCRIPT_DIR=%~dp0"

set "PYTHON=%SCRIPT_DIR%..\..\api\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

rem Required: where your images live
if "%DATASET_DIR%"=="" set "DATASET_DIR=%SCRIPT_DIR%datasets"

rem Outputs (kept under this folder by default)
if "%CAPTIONS_CSV%"=="" set "CAPTIONS_CSV=%SCRIPT_DIR%captions.csv"
if "%TRAINING_INPUTS_DIR%"=="" set "TRAINING_INPUTS_DIR=%SCRIPT_DIR%training_inputs"

rem Captioning controls
if "%CAPTION_GLOB%"=="" set "CAPTION_GLOB=*"
if "%CAPTION_MODEL%"=="" set "CAPTION_MODEL=fancyfeast/llama-joycaption-beta-one-hf-llava"
if "%CAPTION_PROMPT%"=="" set "CAPTION_PROMPT=Write a brief caption for this image in a formal tone."
if "%CAPTION_MAX_NEW_TOKENS%"=="" set "CAPTION_MAX_NEW_TOKENS=512"

rem Encoding controls
if "%MANIFEST_YAML%"=="" set "MANIFEST_YAML=%SCRIPT_DIR%..\..\api\manifest\image\zimage-1.0.0.v1.yml"
if "%TEXT_DEVICE%"=="" set "TEXT_DEVICE=cuda"
if "%VAE_MAX_AREA%"=="" set "VAE_MAX_AREA=921600"
if "%VAE_MOD_VALUE%"=="" set "VAE_MOD_VALUE=16"
if "%VAE_OFFLOAD%"=="" set "VAE_OFFLOAD=0"

rem Training controls
if "%RUN_NAME%"=="" set "RUN_NAME=run"
if "%MAX_STEPS%"=="" set "MAX_STEPS=5000"
if "%OPTIMIZER%"=="" set "OPTIMIZER=adamw"

rem Stage toggles (set to 1 to skip)
if "%SKIP_CAPTION%"=="" set "SKIP_CAPTION=0"
if "%SKIP_TEXT_ENCODE%"=="" set "SKIP_TEXT_ENCODE=0"
if "%SKIP_VAE_ENCODE%"=="" set "SKIP_VAE_ENCODE=0"
if "%SKIP_TRAIN%"=="" set "SKIP_TRAIN=0"

echo [run_all] python: %PYTHON%
echo [run_all] dataset: %DATASET_DIR%
echo [run_all] captions_csv: %CAPTIONS_CSV%
echo [run_all] training_inputs: %TRAINING_INPUTS_DIR%
echo [run_all] run_name: %RUN_NAME%

if not exist "%TRAINING_INPUTS_DIR%" mkdir "%TRAINING_INPUTS_DIR%"

if "%SKIP_CAPTION%"=="1" (
  echo [run_all] stage: caption (skipped)
) else (
  echo [run_all] stage: caption
  "%PYTHON%" "%SCRIPT_DIR%caption.py" ^
    --dataset-dir "%DATASET_DIR%" ^
    --out-csv "%CAPTIONS_CSV%" ^
    --glob "%CAPTION_GLOB%" ^
    --model "%CAPTION_MODEL%" ^
    --prompt "%CAPTION_PROMPT%" ^
    --max-new-tokens "%CAPTION_MAX_NEW_TOKENS%"
)

if "%SKIP_TEXT_ENCODE%"=="1" (
  echo [run_all] stage: text_encode (skipped)
) else (
  echo [run_all] stage: text_encode
  "%PYTHON%" "%SCRIPT_DIR%text_encode.py" ^
    --dataset-dir "%DATASET_DIR%" ^
    --captions-csv "%CAPTIONS_CSV%" ^
    --out-dir "%TRAINING_INPUTS_DIR%" ^
    --out-file "text_encodings.safetensors" ^
    --device "%TEXT_DEVICE%" ^
    --yaml-path "%MANIFEST_YAML%"
)

if "%SKIP_VAE_ENCODE%"=="1" (
  echo [run_all] stage: vae_encode (skipped)
) else (
  echo [run_all] stage: vae_encode
  set "VAE_OFFLOAD_FLAG="
  if "%VAE_OFFLOAD%"=="1" set "VAE_OFFLOAD_FLAG=--vae-offload"

  "%PYTHON%" "%SCRIPT_DIR%vae_encode.py" ^
    --dataset-dir "%DATASET_DIR%" ^
    --captions-csv "%CAPTIONS_CSV%" ^
    --out-dir "%TRAINING_INPUTS_DIR%" ^
    --out-file "vae_encodings.safetensors" ^
    --yaml-path "%MANIFEST_YAML%" ^
    --max-area "%VAE_MAX_AREA%" ^
    --mod-value "%VAE_MOD_VALUE%" ^
    !VAE_OFFLOAD_FLAG!
)

if "%SKIP_TRAIN%"=="1" (
  echo [run_all] stage: train (skipped)
) else (
  echo [run_all] stage: train
  call "%SCRIPT_DIR%train.bat"
)

echo [run_all] done
endlocal

