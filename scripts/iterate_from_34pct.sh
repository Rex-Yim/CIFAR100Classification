#!/usr/bin/env bash
# Train *on top of* the ~34% SE-ResNet baseline: load weights only, new optimizer/schedule.
# Requires a baseline checkpoint (same architecture as repro_34pct_4epoch_baseline.sh).
#
# Example:
#   INIT_CKPT="$SAVE_DIR/colab_repro_34pct_4epoch/best.pt" bash scripts/iterate_from_34pct.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"
SAVE_DIR="${SAVE_DIR:-$ROOT_DIR/checkpoints}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"

INIT_CKPT="${INIT_CKPT:-$SAVE_DIR/colab_repro_34pct_4epoch/best.pt}"
RUN_NAME="${RUN_NAME:-se_resnet_from_34pct}"
EPOCHS="${EPOCHS:-60}"

if [[ ! -f "$INIT_CKPT" ]]; then
  echo "Missing checkpoint: $INIT_CKPT" >&2
  echo "Train the baseline first (repro_34pct_4epoch) or set INIT_CKPT=... to your best.pt" >&2
  exit 1
fi

exec "$PYTHON_BIN" "$ROOT_DIR/train.py" \
  --save-dir "$SAVE_DIR" \
  --data-dir "$DATA_DIR" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --run-name "$RUN_NAME" \
  --init-from "$INIT_CKPT" \
  --model-name se_resnet \
  --base-width 32 \
  --blocks-per-stage 2 2 2 2 \
  --drop-path-rate 0 \
  --epochs "$EPOCHS" \
  --eval-split val \
  --val-ratio 0.1 \
  --split-seed 42 \
  --aug randaugment \
  --randaugment-n 2 \
  --randaugment-m 9 \
  --random-erasing-p 0.05 \
  --mix-mode none \
  --label-smoothing 0.05 \
  --optimizer sgd \
  --lr 0.05 \
  --weight-decay 5e-4 \
  --warmup-epochs 5 \
  --grad-clip 1.0 \
  --ema-decay 0 \
  "$@"
