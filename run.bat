@echo off
setlocal

set VENV_DIR=venv
if exist "%VENV_DIR%\Scripts\activate.bat" (
  call "%VENV_DIR%\Scripts\activate.bat"
)

if /I "%1"=="ollama" goto run_ollama

if "%HF_TOKEN%"=="" (
  echo HF_TOKEN is required for Hugging Face Inference. Set it and retry.
  exit /b 1
)

python run_pipeline_v2.py
exit /b %errorlevel%

:run_ollama
shift
python run_pipeline_ollama.py %1 %2 %3 %4 %5 %6 %7 %8 %9
exit /b %errorlevel%
