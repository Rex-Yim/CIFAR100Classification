#!/usr/bin/env bash
# Verify deps, dataset path, and PyTorch device availability (CUDA / MPS / CPU).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"
if [[ -x "$VENV_PYTHON" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$VENV_PYTHON}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

echo "ROOT_DIR=$ROOT_DIR"
echo "PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" --version

"$PYTHON_BIN" -m pip install -q -r "$ROOT_DIR/requirements.txt"

"$PYTHON_BIN" - <<'PY'
import sys
from pathlib import Path

try:
    import torch
except ImportError as e:
    print("ERROR: torch not installed:", e, file=sys.stderr)
    sys.exit(1)

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device:", torch.cuda.get_device_name(0))
mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
print("mps_available:", mps)

train = Path("data") / "cifar-100-python" / "train"
if train.exists():
    print("cifar100_data: OK", train)
else:
    print("cifar100_data: missing (train.py will download on first run if online)")
PY

echo "Done."
