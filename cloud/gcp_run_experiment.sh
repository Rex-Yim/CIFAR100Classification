#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: cloud/gcp_run_experiment.sh <gs://bucket> <run-name> [train args...]" >&2
  exit 1
fi

ROOT_DIR="${ROOT_DIR:-$PWD}"
export CLOUDSDK_CONFIG="${CLOUDSDK_CONFIG:-$ROOT_DIR/.gcloud}"
mkdir -p "$CLOUDSDK_CONFIG"

BUCKET_URI="${1%/}"
RUN_NAME="$2"
shift 2

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT_DIR/checkpoints}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/runs}"
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-300}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"

mkdir -p "$CHECKPOINT_ROOT" "$LOG_ROOT" "$ROOT_DIR/data"

if ! command -v gsutil >/dev/null 2>&1; then
  echo "gsutil is required on the VM but was not found." >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install -r "$ROOT_DIR/requirements.txt"

sync_inputs() {
  gsutil -m rsync -r "$BUCKET_URI/datasets" "$ROOT_DIR/data" || true
  gsutil -m rsync -r "$BUCKET_URI/checkpoints" "$CHECKPOINT_ROOT" || true
  gsutil -m rsync -r "$BUCKET_URI/logs" "$LOG_ROOT" || true
}

sync_outputs() {
  local run_dir="$CHECKPOINT_ROOT/$RUN_NAME"
  [[ -d "$ROOT_DIR/data" ]] && gsutil -m rsync -r "$ROOT_DIR/data" "$BUCKET_URI/datasets" || true
  [[ -d "$run_dir" ]] && gsutil -m rsync -r "$run_dir" "$BUCKET_URI/checkpoints/$RUN_NAME" || true
  [[ -f "$LOG_ROOT/$RUN_NAME.log" ]] && gsutil cp "$LOG_ROOT/$RUN_NAME.log" "$BUCKET_URI/logs/$RUN_NAME.log" || true
}

sync_inputs

background_sync() {
  while true; do
    sleep "$SYNC_INTERVAL_SECONDS"
    sync_outputs
  done
}

background_sync &
SYNC_PID=$!

cleanup() {
  kill "$SYNC_PID" >/dev/null 2>&1 || true
  sync_outputs
}
trap cleanup EXIT INT TERM

"$PYTHON_BIN" "$ROOT_DIR/train.py" \
  --save-dir "$CHECKPOINT_ROOT" \
  --run-name "$RUN_NAME" \
  --resume latest \
  "$@" 2>&1 | tee "$LOG_ROOT/$RUN_NAME.log"
