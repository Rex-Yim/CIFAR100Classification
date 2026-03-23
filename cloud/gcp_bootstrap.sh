#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$PWD}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"

if [[ -d "$ROOT_DIR/.git" ]]; then
  echo "Repository detected at $ROOT_DIR"
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but was not found." >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r "$ROOT_DIR/requirements.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi was not found. Verify the GPU driver stack on the VM before training." >&2
fi

"$VENV_DIR/bin/python" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
PY
