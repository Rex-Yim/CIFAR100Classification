#!/usr/bin/env bash
# Stage D: continue from Stage C `best.pt` (same recipe as Stage C, typically lower LR).
# Run after Stage C has converged (early-stopped or reached max epochs).
#
# Example:
#   INIT_CKPT="$SAVE_DIR/se_resnet_stage_c_from65/best.pt" \
#   RUN_NAME=se_resnet_stage_d_from_stagec EPOCHS=120 EARLY_STOP_PATIENCE=12 LR=0.008 \
#   bash scripts/iterate_from_stagec_stage_d.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"
SAVE_DIR="${SAVE_DIR:-$ROOT_DIR/checkpoints}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"

INIT_CKPT="${INIT_CKPT:-$SAVE_DIR/se_resnet_stage_c_from65/best.pt}"
RUN_NAME="${RUN_NAME:-se_resnet_stage_d_from_stagec}"
EPOCHS="${EPOCHS:-120}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-12}"
LR="${LR:-0.008}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"

if [[ ! -f "$INIT_CKPT" ]]; then
  echo "Missing Stage C checkpoint: $INIT_CKPT" >&2
  echo "Finish Stage C first or set INIT_CKPT=..." >&2
  exit 1
fi

EARLY_STOP_ARGS=()
if [[ "${EARLY_STOP_PATIENCE}" != "0" ]]; then
  EARLY_STOP_ARGS=(--early-stopping-patience "${EARLY_STOP_PATIENCE}")
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
  --mix-mode both \
  --mixup-alpha 0.2 \
  --cutmix-alpha 1.0 \
  --mix-prob 0.5 \
  --label-smoothing 0.05 \
  --optimizer sgd \
  --lr "$LR" \
  --weight-decay 5e-4 \
  --warmup-epochs "$WARMUP_EPOCHS" \
  --grad-clip 1.0 \
  --ema-decay 0.999 \
  "${EARLY_STOP_ARGS[@]}" \
  "$@"
