#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$DEFAULT_PYTHON_BIN" ]]; then
  DEFAULT_PYTHON_BIN="python3"
fi
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
SAVE_DIR="${SAVE_DIR:-$ROOT_DIR/checkpoints}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"

# Optional overrides for smoke tests (defaults match the 50% plan: long val runs + final train).
HYDRA_AB_EPOCHS="${HYDRA_AB_EPOCHS:-200}"
HYDRA_CDE_EPOCHS="${HYDRA_CDE_EPOCHS:-240}"
RUN_F_EPOCHS="${RUN_F_EPOCHS:-300}"

COMMON_ARGS=(
  --save-dir "$SAVE_DIR"
  --data-dir "$DATA_DIR"
  --device "$DEVICE"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --val-ratio 0.1
  --split-seed 42
  --model-name wrn_hydra
  --depth 28
  --widen-factor 10
  --optimizer sgd
  --lr 0.1
  --weight-decay 5e-4
  --warmup-epochs 5
  --grad-clip 1.0
  --ema-decay 0.999
)

run_train() {
  local run_name="$1"
  shift
  "$PYTHON_BIN" "$ROOT_DIR/train.py" "${COMMON_ARGS[@]}" --run-name "$run_name" "$@"
}

case "$MODE" in
  run-a)
    run_train hydra_run_a \
      --epochs "$HYDRA_AB_EPOCHS" \
      --eval-split val \
      --aug basic \
      --random-erasing-p 0.0 \
      --mix-mode none \
      --label-smoothing 0.05 \
      --attention none \
      --downsample-mode stride
    ;;
  run-b)
    run_train hydra_run_b \
      --epochs "$HYDRA_AB_EPOCHS" \
      --eval-split val \
      --aug randaugment \
      --randaugment-n 2 \
      --randaugment-m 9 \
      --random-erasing-p 0.1 \
      --mix-mode none \
      --label-smoothing 0.05 \
      --attention none \
      --downsample-mode stride
    ;;
  run-c)
    run_train hydra_run_c \
      --epochs "$HYDRA_CDE_EPOCHS" \
      --eval-split val \
      --aug randaugment \
      --randaugment-n 2 \
      --randaugment-m 9 \
      --random-erasing-p 0.1 \
      --mix-mode both \
      --mixup-alpha 0.2 \
      --cutmix-alpha 1.0 \
      --mix-prob 1.0 \
      --label-smoothing 0.0 \
      --attention none \
      --downsample-mode stride
    ;;
  run-d)
    run_train hydra_run_d \
      --epochs "$HYDRA_CDE_EPOCHS" \
      --eval-split val \
      --aug randaugment \
      --randaugment-n 2 \
      --randaugment-m 9 \
      --random-erasing-p 0.1 \
      --mix-mode both \
      --mixup-alpha 0.2 \
      --cutmix-alpha 1.0 \
      --mix-prob 1.0 \
      --label-smoothing 0.0 \
      --attention eca \
      --downsample-mode antialias
    ;;
  run-e)
    run_train hydra_run_e \
      --epochs "$HYDRA_CDE_EPOCHS" \
      --eval-split val \
      --aug randaugment \
      --randaugment-n 2 \
      --randaugment-m 9 \
      --random-erasing-p 0.1 \
      --mix-mode both \
      --mixup-alpha 0.2 \
      --cutmix-alpha 1.0 \
      --mix-prob 1.0 \
      --label-smoothing 0.0 \
      --attention se \
      --downsample-mode antialias
    ;;
  run-f)
    BEST_RUN="${BEST_RUN:?Set BEST_RUN to the best of hydra_run_c, hydra_run_d, or hydra_run_e}"
    BEST_ARGS_JSON="$SAVE_DIR/$BEST_RUN/args.json"
    if [[ ! -f "$BEST_ARGS_JSON" ]]; then
      echo "Missing args.json for $BEST_RUN at $BEST_ARGS_JSON" >&2
      exit 1
    fi

    mapfile -t FINAL_ARGS < <("$PYTHON_BIN" - "$BEST_ARGS_JSON" <<'PY'
import json
import sys

keys = [
    ("model-name", "model_name"),
    ("depth", "depth"),
    ("widen-factor", "widen_factor"),
    ("attention", "attention"),
    ("downsample-mode", "downsample_mode"),
    ("aug", "aug"),
    ("randaugment-n", "randaugment_n"),
    ("randaugment-m", "randaugment_m"),
    ("random-erasing-p", "random_erasing_p"),
    ("optimizer", "optimizer"),
    ("lr", "lr"),
    ("weight-decay", "weight_decay"),
    ("warmup-epochs", "warmup_epochs"),
    ("grad-clip", "grad_clip"),
    ("ema-decay", "ema_decay"),
    ("label-smoothing", "label_smoothing"),
    ("mix-mode", "mix_mode"),
    ("mixup-alpha", "mixup_alpha"),
    ("cutmix-alpha", "cutmix_alpha"),
    ("mix-prob", "mix_prob"),
]

payload = json.load(open(sys.argv[1]))
for flag, key in keys:
    print(f"--{flag}")
    print(str(payload[key]))
PY
)

    "$PYTHON_BIN" "$ROOT_DIR/train.py" \
      --save-dir "$SAVE_DIR" \
      --data-dir "$DATA_DIR" \
      --device "$DEVICE" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --run-name hydra_run_f_final \
      --epochs "$RUN_F_EPOCHS" \
      --final-train-full \
      --eval-split test \
      "${FINAL_ARGS[@]}"
    ;;
  select-best)
    "$PYTHON_BIN" "$ROOT_DIR/scripts/select_best_run.py" \
      "$SAVE_DIR/hydra_run_c" \
      "$SAVE_DIR/hydra_run_d" \
      "$SAVE_DIR/hydra_run_e"
    ;;
  all-ablations)
    "$0" run-a
    "$0" run-b
    "$0" run-c
    "$0" run-d
    "$0" run-e
    ;;
  *)
    cat <<'EOF'
Usage:
  scripts/hydra_ladder.sh run-a
  scripts/hydra_ladder.sh run-b
  scripts/hydra_ladder.sh run-c
  scripts/hydra_ladder.sh run-d
  scripts/hydra_ladder.sh run-e
  BEST_RUN=hydra_run_d scripts/hydra_ladder.sh run-f
  scripts/hydra_ladder.sh select-best
  scripts/hydra_ladder.sh all-ablations
Environment (optional):
  HYDRA_AB_EPOCHS   default 200 (run-a, run-b)
  HYDRA_CDE_EPOCHS  default 240 (run-c, run-d, run-e)
  RUN_F_EPOCHS      default 300 (run-f)
EOF
    exit 1
    ;;
esac
