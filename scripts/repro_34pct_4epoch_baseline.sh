#!/usr/bin/env bash
# Reproduce the local ~34% CIFAR-100 *test* accuracy in ~4 epochs (CPU reference run).
# This uses the small SE-ResNet recipe from checkpoints_cpu_4ep (not WRN-Hydra / hydra_ladder).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"
SAVE_DIR="${SAVE_DIR:-$ROOT_DIR/checkpoints}"
DEVICE="${DEVICE:-cpu}"
NUM_WORKERS="${NUM_WORKERS:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RUN_NAME="${RUN_NAME:-repro_34pct_4epoch}"

exec "$PYTHON_BIN" "$ROOT_DIR/train.py" \
  --save-dir "$SAVE_DIR" \
  --data-dir "$DATA_DIR" \
  --run-name "$RUN_NAME" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --model-name se_resnet \
  --base-width 32 \
  --blocks-per-stage 2 2 2 2 \
  --drop-path-rate 0 \
  --epochs 4 \
  --eval-split test \
  --aug basic \
  --random-erasing-p 0 \
  --mix-mode none \
  --label-smoothing 0 \
  --optimizer sgd \
  --lr 0.1 \
  --weight-decay 5e-4 \
  --warmup-epochs 1 \
  --grad-clip 1.0 \
  --ema-decay 0 \
  "$@"
