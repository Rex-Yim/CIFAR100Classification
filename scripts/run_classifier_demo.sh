#!/usr/bin/env bash
# Serve the CIFAR-100 classifier demo at http://127.0.0.1:8765
#
# Usage:
#   CHECKPOINT=~/Downloads/best-3.pt bash scripts/run_classifier_demo.sh
#   DEMO_DEVICE=cpu bash scripts/run_classifier_demo.sh   # optional: cpu | cuda | mps
#
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="${PYTHONPATH:-}:$ROOT_DIR"
exec python3 -m uvicorn demo.app:app --host 127.0.0.1 --port 8765 "$@"
