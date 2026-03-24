#!/usr/bin/env bash
# Stage C from Stage B (~65%): init from se_resnet_from_34pct/best.pt + Mixup/CutMix + EMA.
# Uses a *lower* LR than training from the ~34% baseline so you do not destroy good weights.
#
# Requires: checkpoints/se_resnet_from_34pct/best.pt (run quick_se_resnet_results.sh first).
#
# Example (Colab / Drive):
#   INIT_CKPT="$SAVE_DIR/se_resnet_from_34pct/best.pt" \
#   RUN_NAME=se_resnet_stage_c_from65 EPOCHS=30 LR=0.015 \
#   bash scripts/iterate_from_stageb_stage_c_mixema.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"
SAVE_DIR="${SAVE_DIR:-$ROOT_DIR/checkpoints}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"

INIT_CKPT="${INIT_CKPT:-$SAVE_DIR/se_resnet_from_34pct/best.pt}"
RUN_NAME="${RUN_NAME:-se_resnet_stage_c_from65}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-0.015}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"

if [[ ! -f "$INIT_CKPT" ]]; then
  echo "Missing Stage B checkpoint: $INIT_CKPT" >&2
  echo "Train Stage B first (quick_se_resnet_results.sh) or set INIT_CKPT=..." >&2
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
  "$@"
