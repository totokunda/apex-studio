@echo off
setlocal enableextensions enabledelayedexpansion

rem Windows wrapper for text_encode.py (location-independent).

set "SCRIPT_DIR=%~dp0"

set "PYTHON=%SCRIPT_DIR%..\..\api\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

rem Defaults (override by setting env vars before running)
if "%DATASET_DIR%"=="" set "DATASET_DIR=%SCRIPT_DIR%datasets"
if "%CAPTIONS_CSV%"=="" set "CAPTIONS_CSV=%SCRIPT_DIR%captions.csv"
if "%TRAINING_INPUTS_DIR%"=="" set "TRAINING_INPUTS_DIR=%SCRIPT_DIR%training_inputs"
if "%MANIFEST_YAML%"=="" set "MANIFEST_YAML=%SCRIPT_DIR%..\..\api\manifest\image\zimage-1.0.0.v1.yml"
if "%TEXT_DEVICE%"=="" set "TEXT_DEVICE=cuda"
if "%TEXT_OUT_FILE%"=="" set "TEXT_OUT_FILE=text_encodings.safetensors"

if not exist "%TRAINING_INPUTS_DIR%" mkdir "%TRAINING_INPUTS_DIR%"

"%PYTHON%" "%SCRIPT_DIR%text_encode.py" ^
  --dataset-dir "%DATASET_DIR%" ^
  --captions-csv "%CAPTIONS_CSV%" ^
  --out-dir "%TRAINING_INPUTS_DIR%" ^
  --out-file "%TEXT_OUT_FILE%" ^
  --device "%TEXT_DEVICE%" ^
  --yaml-path "%MANIFEST_YAML%"

endlocal

