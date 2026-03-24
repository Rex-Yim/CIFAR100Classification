#!/usr/bin/env bash
# Run training workflows locally (Cursor / terminal). Same recipes as Colab; no google.colab.
# Usage: from repo root:  bash scripts/local_runner.sh <command>
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"
if [[ -x "$VENV_PYTHON" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$VENV_PYTHON}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

export DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"
export SAVE_DIR="${SAVE_DIR:-$ROOT_DIR/checkpoints}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export BATCH_SIZE="${BATCH_SIZE:-128}"

detect_device() {
  if [[ -n "${DEVICE:-}" ]]; then
    printf '%s\n' "$DEVICE"
    return
  fi
  "$PYTHON_BIN" - <<'PY'
try:
    import torch
    if torch.cuda.is_available():
        print("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("mps")
    else:
        print("cpu")
except Exception:
    print("cpu")
PY
}

export DEVICE="${DEVICE:-$(detect_device)}"
export PYTHON_BIN

usage() {
  cat <<EOF
Local runner (Cursor / terminal) — same scripts as Colab, default paths under this repo.

Environment (optional overrides):
  DATA_DIR   default: $ROOT_DIR/data
  SAVE_DIR   default: $ROOT_DIR/checkpoints
  DEVICE     default: auto (cuda > mps > cpu)
  BATCH_SIZE default: 128 (baseline repro uses 64 inside its script)
  NUM_WORKERS
  PYTHON_BIN default: .venv/bin/python if present
  RUN_NAME   baseline run name (default: repro_34pct_4epoch)
  BEST_RUN   for final — hydra_run_c, hydra_run_d, or hydra_run_e

Commands:
  env          Print resolved paths and device
  install      pip install -r requirements.txt
  baseline     4-epoch SE-ResNet + test eval (~34% class sanity check)
  ladder MODE  hydra_ladder.sh MODE  (run-c, run-d, run-e, run-f, select-best, ...)
  select-best  Compare c/d/e and print JSON winner
  final        Auto pick BEST_RUN from select-best, then run-f (full train + test)
  results PATH Run results.py on a checkpoint (default: SAVE_DIR/hydra_run_f_final/best.pt)
  stage-c      Stage C (Mixup/CutMix/EMA) from Stage B `se_resnet_from_34pct/best.pt`
               (same as `scripts/iterate_from_stageb_stage_c_mixema.sh`). Override INIT_CKPT, EPOCHS, RUN_NAME.

Examples:
  bash scripts/local_runner.sh install
  bash scripts/local_runner.sh env
  bash scripts/local_runner.sh baseline
  bash scripts/local_runner.sh ladder run-d
  bash scripts/local_runner.sh select-best
  bash scripts/local_runner.sh final
  bash scripts/local_runner.sh results checkpoints/hydra_run_f_final/best.pt
  INIT_CKPT=checkpoints/se_resnet_from_34pct/best.pt bash scripts/local_runner.sh stage-c
EOF
}

cmd="${1:-help}"
shift || true

case "$cmd" in
  help|-h|--help)
    usage
    ;;
  env)
    echo "ROOT_DIR=$ROOT_DIR"
    echo "DATA_DIR=$DATA_DIR"
    echo "SAVE_DIR=$SAVE_DIR"
    echo "DEVICE=$DEVICE"
    echo "PYTHON_BIN=$PYTHON_BIN"
    echo "BATCH_SIZE=$BATCH_SIZE"
    echo "NUM_WORKERS=$NUM_WORKERS"
    ;;
  install)
    "$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements.txt"
    ;;
  baseline)
    export RUN_NAME="${RUN_NAME:-repro_34pct_4epoch}"
    exec bash "$ROOT_DIR/scripts/repro_34pct_4epoch_baseline.sh" "$@"
    ;;
  ladder)
    mode="${1:?Usage: local_runner.sh ladder <run-c|run-d|run-e|run-f|select-best|...>}"
    shift || true
    exec bash "$ROOT_DIR/scripts/hydra_ladder.sh" "$mode" "$@"
    ;;
  select-best)
    exec "$PYTHON_BIN" "$ROOT_DIR/scripts/select_best_run.py" \
      "$SAVE_DIR/hydra_run_c" \
      "$SAVE_DIR/hydra_run_d" \
      "$SAVE_DIR/hydra_run_e"
    ;;
  final)
    # Pick winning run from history, then full-train + test (run-f).
    export BEST_RUN="$(
      "$PYTHON_BIN" - <<PY
import json
import subprocess
import sys
from pathlib import Path

root = Path("$ROOT_DIR")
save = Path("$SAVE_DIR")
r = subprocess.run(
    [sys.executable, str(root / "scripts" / "select_best_run.py"),
     str(save / "hydra_run_c"),
     str(save / "hydra_run_d"),
     str(save / "hydra_run_e")],
    cwd=str(root),
    capture_output=True,
    text=True,
)
if r.returncode != 0:
    sys.stderr.write(r.stderr)
    sys.exit(r.returncode)
data = json.loads(r.stdout)
print(Path(data["winner"]["run_dir"]).name)
PY
    )"
    echo "BEST_RUN=$BEST_RUN (from select-best)"
    exec bash "$ROOT_DIR/scripts/hydra_ladder.sh" run-f
    ;;
  results)
    ckpt="${1:-$SAVE_DIR/hydra_run_f_final/best.pt}"
    shift || true
    exec "$PYTHON_BIN" "$ROOT_DIR/results.py" \
      --checkpoint "$ckpt" \
      --data-dir "$DATA_DIR" \
      --batch-size "${RESULTS_BATCH_SIZE:-256}" \
      --num-workers "$NUM_WORKERS" \
      --device "$DEVICE" \
      "$@"
    ;;
  stage-c)
    export INIT_CKPT="${INIT_CKPT:-$SAVE_DIR/se_resnet_from_34pct/best.pt}"
    export RUN_NAME="${RUN_NAME:-se_resnet_stage_c_from65}"
    export EPOCHS="${EPOCHS:-30}"
    export LR="${LR:-0.015}"
    export WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
    exec bash "$ROOT_DIR/scripts/iterate_from_stageb_stage_c_mixema.sh" "$@"
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage >&2
    exit 1
    ;;
esac
