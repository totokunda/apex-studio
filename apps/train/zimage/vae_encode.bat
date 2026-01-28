@echo off
setlocal enableextensions enabledelayedexpansion

rem Windows wrapper for vae_encode.py (location-independent).

set "SCRIPT_DIR=%~dp0"

set "PYTHON=%SCRIPT_DIR%..\..\api\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

rem Defaults (override by setting env vars before running)
if "%DATASET_DIR%"=="" set "DATASET_DIR=%SCRIPT_DIR%datasets"
if "%CAPTIONS_CSV%"=="" set "CAPTIONS_CSV=%SCRIPT_DIR%captions.csv"
if "%TRAINING_INPUTS_DIR%"=="" set "TRAINING_INPUTS_DIR=%SCRIPT_DIR%training_inputs"
if "%MANIFEST_YAML%"=="" set "MANIFEST_YAML=%SCRIPT_DIR%..\..\api\manifest\image\zimage-1.0.0.v1.yml"
if "%VAE_OUT_FILE%"=="" set "VAE_OUT_FILE=vae_encodings.safetensors"

if "%VAE_MAX_AREA%"=="" set "VAE_MAX_AREA=921600"
if "%VAE_MOD_VALUE%"=="" set "VAE_MOD_VALUE=16"
if "%VAE_OFFLOAD%"=="" set "VAE_OFFLOAD=0"

if not exist "%TRAINING_INPUTS_DIR%" mkdir "%TRAINING_INPUTS_DIR%"

set "VAE_OFFLOAD_FLAG="
if "%VAE_OFFLOAD%"=="1" set "VAE_OFFLOAD_FLAG=--vae-offload"

"%PYTHON%" "%SCRIPT_DIR%vae_encode.py" ^
  --dataset-dir "%DATASET_DIR%" ^
  --captions-csv "%CAPTIONS_CSV%" ^
  --out-dir "%TRAINING_INPUTS_DIR%" ^
  --out-file "%VAE_OUT_FILE%" ^
  --yaml-path "%MANIFEST_YAML%" ^
  --max-area "%VAE_MAX_AREA%" ^
  --mod-value "%VAE_MOD_VALUE%" ^
  !VAE_OFFLOAD_FLAG!

endlocal

