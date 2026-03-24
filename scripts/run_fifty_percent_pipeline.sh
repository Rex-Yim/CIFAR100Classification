#!/usr/bin/env bash
# Full sequence for the ~50% test-accuracy plan: run-c, run-d, run-e, select-best, run-f, results.py
# Defaults: HYDRA_CDE_EPOCHS=240, RUN_F_EPOCHS=300. Override for smoke tests, e.g.:
#   HYDRA_CDE_EPOCHS=1 RUN_F_EPOCHS=1 bash scripts/run_fifty_percent_pipeline.sh
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
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export BATCH_SIZE="${BATCH_SIZE:-128}"

if [[ -z "${DEVICE:-}" ]]; then
  export DEVICE="$("$PYTHON_BIN" - <<'PY'
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
)"
fi

echo "DEVICE=$DEVICE DATA_DIR=$DATA_DIR SAVE_DIR=$SAVE_DIR"
echo "HYDRA_CDE_EPOCHS=${HYDRA_CDE_EPOCHS:-240} RUN_F_EPOCHS=${RUN_F_EPOCHS:-300}"

echo ">>> hydra_ladder.sh run-c"
bash "$ROOT_DIR/scripts/hydra_ladder.sh" run-c

echo ">>> hydra_ladder.sh run-d"
bash "$ROOT_DIR/scripts/hydra_ladder.sh" run-d

echo ">>> hydra_ladder.sh run-e"
bash "$ROOT_DIR/scripts/hydra_ladder.sh" run-e

echo ">>> hydra_ladder.sh select-best"
bash "$ROOT_DIR/scripts/hydra_ladder.sh" select-best

export BEST_RUN="$("$PYTHON_BIN" - <<PY
import json
import subprocess
import sys
from pathlib import Path

root = Path("$ROOT_DIR")
save = Path("$SAVE_DIR")
r = subprocess.run(
    [
        sys.executable,
        str(root / "scripts" / "select_best_run.py"),
        str(save / "hydra_run_c"),
        str(save / "hydra_run_d"),
        str(save / "hydra_run_e"),
    ],
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
echo "BEST_RUN=$BEST_RUN"

echo ">>> hydra_ladder.sh run-f"
bash "$ROOT_DIR/scripts/hydra_ladder.sh" run-f

echo ">>> results.py (official test)"
exec "$PYTHON_BIN" "$ROOT_DIR/results.py" \
  --checkpoint "$SAVE_DIR/hydra_run_f_final/best.pt" \
  --data-dir "$DATA_DIR" \
  --batch-size "${RESULTS_BATCH_SIZE:-256}" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE"
