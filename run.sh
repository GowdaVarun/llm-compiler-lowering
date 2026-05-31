#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
if [ -d "$VENV_DIR" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

if [ "${1:-}" = "ollama" ]; then
  shift
  python run_pipeline_ollama.py "$@"
  exit 0
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is required for Hugging Face Inference. Set it and retry."
  exit 1
fi

python run_pipeline_v2.py
