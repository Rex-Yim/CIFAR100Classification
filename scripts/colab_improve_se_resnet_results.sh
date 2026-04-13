#!/usr/bin/env bash
# Colab-friendly improve path: Stage C -> optional Stage D -> pick better val checkpoint -> official test.
#
# Example:
#   PYTHON_BIN=python DATA_DIR=/content/drive/MyDrive/Colab_CIFAR/data \
#   SAVE_DIR=/content/drive/MyDrive/Colab_CIFAR/checkpoints DEVICE=cuda \
#   bash scripts/colab_improve_se_resnet_results.sh
#
# Optional:
#   RUN_STAGE_D=0                         # stop after Stage C
#   REUSE_EXISTING=1                      # skip stages whose best.pt already exists
#   STAGE_C_RUN=se_resnet_stage_c_from65  # override run names
#   STAGE_D_RUN=se_resnet_stage_d_from_stagec
#   ... bash scripts/colab_improve_se_resnet_results.sh --use-hierarchical-loss
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"
SAVE_DIR="${SAVE_DIR:-$ROOT_DIR/checkpoints}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
RESULTS_BATCH_SIZE="${RESULTS_BATCH_SIZE:-256}"
REUSE_EXISTING="${REUSE_EXISTING:-1}"
RUN_STAGE_D="${RUN_STAGE_D:-1}"

STAGE_B_RUN="${STAGE_B_RUN:-se_resnet_from_34pct}"
STAGE_C_RUN="${STAGE_C_RUN:-se_resnet_stage_c_from65}"
STAGE_D_RUN="${STAGE_D_RUN:-se_resnet_stage_d_from_stagec}"

STAGE_C_EPOCHS="${STAGE_C_EPOCHS:-150}"
STAGE_C_PATIENCE="${STAGE_C_PATIENCE:-12}"
STAGE_C_LR="${STAGE_C_LR:-0.015}"
STAGE_C_WARMUP="${STAGE_C_WARMUP:-3}"

STAGE_D_EPOCHS="${STAGE_D_EPOCHS:-120}"
STAGE_D_PATIENCE="${STAGE_D_PATIENCE:-12}"
STAGE_D_LR="${STAGE_D_LR:-0.008}"
STAGE_D_WARMUP="${STAGE_D_WARMUP:-2}"

STAGE_B_PT="$SAVE_DIR/$STAGE_B_RUN/best.pt"
STAGE_C_PT="$SAVE_DIR/$STAGE_C_RUN/best.pt"
STAGE_D_PT="$SAVE_DIR/$STAGE_D_RUN/best.pt"

if [[ ! -f "$STAGE_B_PT" ]]; then
  echo "Missing Stage B checkpoint: $STAGE_B_PT" >&2
  echo "Run scripts/quick_se_resnet_results.sh first, or point STAGE_B_RUN to your ~65% checkpoint." >&2
  exit 1
fi

run_stage_c() {
  if [[ "$REUSE_EXISTING" == "1" && -f "$STAGE_C_PT" ]]; then
    echo ">>> Reusing existing Stage C checkpoint: $STAGE_C_PT"
    return
  fi

  echo ">>> Running Stage C from $STAGE_B_PT"
  env \
    PYTHON_BIN="$PYTHON_BIN" \
    DATA_DIR="$DATA_DIR" \
    SAVE_DIR="$SAVE_DIR" \
    DEVICE="$DEVICE" \
    NUM_WORKERS="$NUM_WORKERS" \
    BATCH_SIZE="$BATCH_SIZE" \
    INIT_CKPT="$STAGE_B_PT" \
    RUN_NAME="$STAGE_C_RUN" \
    EPOCHS="$STAGE_C_EPOCHS" \
    EARLY_STOP_PATIENCE="$STAGE_C_PATIENCE" \
    LR="$STAGE_C_LR" \
    WARMUP_EPOCHS="$STAGE_C_WARMUP" \
    bash "$ROOT_DIR/scripts/iterate_from_stageb_stage_c_mixema.sh" "$@"
}

run_stage_d() {
  if [[ "$RUN_STAGE_D" != "1" ]]; then
    echo ">>> RUN_STAGE_D=$RUN_STAGE_D, skipping Stage D"
    return
  fi

  if [[ ! -f "$STAGE_C_PT" ]]; then
    echo "Stage C checkpoint missing: $STAGE_C_PT" >&2
    exit 1
  fi

  if [[ "$REUSE_EXISTING" == "1" && -f "$STAGE_D_PT" ]]; then
    echo ">>> Reusing existing Stage D checkpoint: $STAGE_D_PT"
    return
  fi

  echo ">>> Running Stage D from $STAGE_C_PT"
  env \
    PYTHON_BIN="$PYTHON_BIN" \
    DATA_DIR="$DATA_DIR" \
    SAVE_DIR="$SAVE_DIR" \
    DEVICE="$DEVICE" \
    NUM_WORKERS="$NUM_WORKERS" \
    BATCH_SIZE="$BATCH_SIZE" \
    INIT_CKPT="$STAGE_C_PT" \
    RUN_NAME="$STAGE_D_RUN" \
    EPOCHS="$STAGE_D_EPOCHS" \
    EARLY_STOP_PATIENCE="$STAGE_D_PATIENCE" \
    LR="$STAGE_D_LR" \
    WARMUP_EPOCHS="$STAGE_D_WARMUP" \
    bash "$ROOT_DIR/scripts/iterate_from_stagec_stage_d.sh" "$@"
}

select_best_checkpoint() {
  "$PYTHON_BIN" - "$STAGE_C_PT" "$STAGE_D_PT" "$RUN_STAGE_D" <<'PY'
import sys
from pathlib import Path

import torch


def score(path_str: str):
    path = Path(path_str)
    if not path.is_file():
        return None
    checkpoint = torch.load(path, map_location="cpu")
    return float(checkpoint.get("best_eval_accuracy", 0.0))


stage_c_path = Path(sys.argv[1])
stage_d_path = Path(sys.argv[2])
run_stage_d = sys.argv[3] == "1"

candidates = []
stage_c_score = score(str(stage_c_path))
if stage_c_score is not None:
    candidates.append((stage_c_score, stage_c_path))

if run_stage_d:
    stage_d_score = score(str(stage_d_path))
    if stage_d_score is not None:
        candidates.append((stage_d_score, stage_d_path))

if not candidates:
    raise SystemExit("No candidate checkpoints found.")

candidates.sort(key=lambda item: item[0], reverse=True)
best_score, best_path = candidates[0]
print(best_path)
print(f">>> Selected checkpoint by validation accuracy: {best_path} (best_eval_accuracy={best_score:.4f})", file=sys.stderr)
PY
}

run_stage_c "$@"
run_stage_d "$@"

BEST_PT="$(select_best_checkpoint)"

echo ">>> Official test with results.py"
exec "$PYTHON_BIN" "$ROOT_DIR/results.py" \
  --checkpoint "$BEST_PT" \
  --data-dir "$DATA_DIR" \
  --batch-size "$RESULTS_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE"
