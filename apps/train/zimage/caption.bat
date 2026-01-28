@echo off
setlocal enableextensions enabledelayedexpansion

rem Windows wrapper for caption.py (location-independent).

set "SCRIPT_DIR=%~dp0"

set "PYTHON=%SCRIPT_DIR%..\..\api\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

rem Defaults (override by setting env vars before running)
if "%DATASET_DIR%"=="" set "DATASET_DIR=%SCRIPT_DIR%datasets"
if "%CAPTIONS_CSV%"=="" set "CAPTIONS_CSV=%SCRIPT_DIR%captions.csv"

if "%CAPTION_GLOB%"=="" set "CAPTION_GLOB=*"
if "%CAPTION_MODEL%"=="" set "CAPTION_MODEL=fancyfeast/llama-joycaption-beta-one-hf-llava"
if "%CAPTION_PROMPT%"=="" set "CAPTION_PROMPT=Write a brief caption for this image in a formal tone."
if "%CAPTION_MAX_NEW_TOKENS%"=="" set "CAPTION_MAX_NEW_TOKENS=512"

"%PYTHON%" "%SCRIPT_DIR%caption.py" ^
  --dataset-dir "%DATASET_DIR%" ^
  --out-csv "%CAPTIONS_CSV%" ^
  --glob "%CAPTION_GLOB%" ^
  --model "%CAPTION_MODEL%" ^
  --prompt "%CAPTION_PROMPT%" ^
  --max-new-tokens "%CAPTION_MAX_NEW_TOKENS%"

endlocal

