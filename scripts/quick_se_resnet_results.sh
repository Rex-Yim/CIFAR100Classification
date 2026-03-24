#!/usr/bin/env bash
# Fast path: 4-epoch SE-ResNet baseline (if missing) → finetune from best.pt → official test eval.
# Defaults tuned for Colab T4: 30 finetune epochs, batch 256.
#
#   SAVE_DIR=... DATA_DIR=... DEVICE=cuda bash scripts/quick_se_resnet_results.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"
SAVE_DIR="${SAVE_DIR:-$ROOT_DIR/checkpoints}"
DEVICE="${DEVICE:-cuda}"

BASE_RUN="${BASE_RUN:-colab_repro_34pct_4epoch}"
FINAL_RUN="${FINAL_RUN:-se_resnet_from_34pct}"
EPOCHS="${EPOCHS:-30}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"

BASE_PT="$SAVE_DIR/$BASE_RUN/best.pt"
FINAL_PT="$SAVE_DIR/$FINAL_RUN/best.pt"

if [[ ! -f "$BASE_PT" ]]; then
  echo ">>> No baseline at $BASE_PT — running repro_34pct_4epoch_baseline.sh (~4 epochs)"
  env \
    PYTHON_BIN="$PYTHON_BIN" \
    DATA_DIR="$DATA_DIR" \
    SAVE_DIR="$SAVE_DIR" \
    DEVICE="$DEVICE" \
    NUM_WORKERS="$NUM_WORKERS" \
    BATCH_SIZE="${BASE_BATCH_SIZE:-64}" \
    RUN_NAME="$BASE_RUN" \
    bash "$ROOT_DIR/scripts/repro_34pct_4epoch_baseline.sh"
else
  echo ">>> Using existing baseline: $BASE_PT"
fi

echo ">>> Finetune from baseline: $EPOCHS epochs → $FINAL_RUN"
env \
  PYTHON_BIN="$PYTHON_BIN" \
  DATA_DIR="$DATA_DIR" \
  SAVE_DIR="$SAVE_DIR" \
  DEVICE="$DEVICE" \
  NUM_WORKERS="$NUM_WORKERS" \
  BATCH_SIZE="$BATCH_SIZE" \
  INIT_CKPT="$BASE_PT" \
  RUN_NAME="$FINAL_RUN" \
  EPOCHS="$EPOCHS" \
  bash "$ROOT_DIR/scripts/iterate_from_34pct.sh"

echo ">>> Official test with results.py"
exec "$PYTHON_BIN" "$ROOT_DIR/results.py" \
  --checkpoint "$FINAL_PT" \
  --data-dir "$DATA_DIR" \
  --batch-size 256 \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE"
