@echo off
setlocal enableextensions enabledelayedexpansion

rem Location-independent training wrapper for Windows.
rem Uses precomputed encodings under apps/train/zimage/ by default.

set "SCRIPT_DIR=%~dp0"

set "PYTHON=%SCRIPT_DIR%..\..\api\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

rem Defaults (override by setting env vars before running)
if "%CAPTIONS_CSV%"=="" set "CAPTIONS_CSV=%SCRIPT_DIR%captions.csv"
if "%TRAINING_INPUTS_DIR%"=="" set "TRAINING_INPUTS_DIR=%SCRIPT_DIR%training_inputs"
if "%OPTIMIZER%"=="" set "OPTIMIZER=adamw"
if "%RUN_NAME%"=="" set "RUN_NAME=run"
if "%MAX_STEPS%"=="" set "MAX_STEPS=5000"

rem Optional resume support:
rem - set RESUME_RUN=1 to resume latest checkpoint under lora/<RUN_NAME>/
rem - set RESUME_PATH=C:\path\to\lora\run_or_checkpoint to resume from a specific run/checkpoint
if "%RESUME_RUN%"=="" set "RESUME_RUN=0"
if not "%RESUME_PATH%"=="" if "%RESUME_RUN%"=="1" (
  echo ERROR: Set only one of RESUME_RUN=1 or RESUME_PATH=...
  exit /b 1
)
set "RESUME_ARGS="
if "%RESUME_RUN%"=="1" set "RESUME_ARGS=--resume_run"
if not "%RESUME_PATH%"=="" set "RESUME_ARGS=--resume ""%RESUME_PATH%"""

"%PYTHON%" "%SCRIPT_DIR%train.py" ^
  --vae_encodings "%TRAINING_INPUTS_DIR%\vae_encodings.safetensors" ^
  --text_encodings "%TRAINING_INPUTS_DIR%\text_encodings.safetensors" ^
  --captions_csv "%CAPTIONS_CSV%" ^
  --caption_dropout 0.05 ^
  --lora_rank 32 ^
  --learning_rate 1e-4 ^
  --optimizer "%OPTIMIZER%" ^
  --batch_size 1 ^
  --mixed_precision bf16 ^
  --gradient_accumulation_steps 4 ^
  --max_steps "%MAX_STEPS%" ^
  --gradient_checkpointing ^
  --run_name "%RUN_NAME%" ^
  --save_every 250 ^
  %RESUME_ARGS%

endlocal

