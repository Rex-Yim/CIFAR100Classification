#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SAVE_DIR="${SAVE_DIR:-$ROOT_DIR/checkpoints}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"
DEVICE="${DEVICE:-cpu}"
NUM_WORKERS="${NUM_WORKERS:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_THREADS="${NUM_THREADS:-8}"
RUN_NAME="${RUN_NAME:-hydra_m1_lite}"
EPOCHS="${EPOCHS:-120}"

"$PYTHON_BIN" "$ROOT_DIR/train.py" \
  --save-dir "$SAVE_DIR" \
  --data-dir "$DATA_DIR" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --num-threads "$NUM_THREADS" \
  --run-name "$RUN_NAME" \
  --model-name wrn_hydra \
  --depth 22 \
  --widen-factor 4 \
  --attention eca \
  --downsample-mode antialias \
  --aug randaugment \
  --randaugment-n 2 \
  --randaugment-m 7 \
  --random-erasing-p 0.1 \
  --optimizer sgd \
  --lr 0.08 \
  --weight-decay 5e-4 \
  --warmup-epochs 5 \
  --grad-clip 1.0 \
  --ema-decay 0.999 \
  --use-hierarchical-loss \
  --coarse-loss-weight 0.25 \
  --mix-mode both \
  --mixup-alpha 0.2 \
  --cutmix-alpha 1.0 \
  --mix-prob 1.0 \
  --label-smoothing 0.0 \
  --epochs "$EPOCHS" \
  --eval-split val \
  --val-ratio 0.1 \
  --split-seed 42 \
  "$@"
