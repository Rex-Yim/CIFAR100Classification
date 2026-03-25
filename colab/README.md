# Colab workflow

Train on **Google Colab** with checkpoints on **Drive**. Local training does not need this folder — use `scripts/local_runner.sh` or `python train.py`.

## Files

- **`CIFAR_Quick.ipynb`** — minimal: setup + **SE-ResNet quick path** via `scripts/quick_se_resnet_results.sh` (baseline → Stage B → official test, ~65% when training completes).
- **`CIFAR_SOTA.ipynb`** — **separate** notebook to **improve** beyond Stage B: assumes `se_resnet_from_34pct/best.pt` on Drive; **Stage C** (max `EPOCHS`, optional early stop via `EARLY_STOP_PATIENCE`) and optional **Stage D** (from Stage C `best.pt`) + official test. Does **not** re-run the full baseline.
- **`prepare_colab_bundle.sh`** — run on your machine to build the tarball Colab unpacks (includes both notebooks).

## Bundle (on your Mac)

```bash
colab/prepare_colab_bundle.sh
```

Creates `artifacts/cifar_hydra_project.tar.gz`. Upload it to **`My Drive/Colab_CIFAR/`** (or My Drive root).

## Colab steps

1. **Runtime → Change runtime type → GPU.**
2. Upload **`CIFAR_Quick.ipynb`** (and the bundle to Drive as above).
3. Run cells in order: **mount Drive** → **unpack** → **pip** → **GPU check** → **env** → **Train + official test**.

That last cell runs `scripts/quick_se_resnet_results.sh` (4-epoch baseline if missing → **30-epoch** Stage B → `results.py`), which is what produces the **~65%** official test run when training finishes.

Data and checkpoints live under **`My Drive/Colab_CIFAR/`** (`data/`, `checkpoints/`).

## Stage C and Stage D (optional)

**Preferred:** open **`CIFAR_SOTA.ipynb`** (Stage C with `EPOCHS` cap + `EARLY_STOP_PATIENCE`, optional Stage D cell, separate `RUN_NAME` per run). Re-run **`colab/prepare_colab_bundle.sh`** after pulling changes so the tarball includes `iterate_from_stagec_stage_d.sh` and updated Stage C defaults.

Or, after **`se_resnet_from_34pct/best.pt`** exists on Drive, run in a **new code cell** in `CIFAR_Quick` (project root is already `os.chdir` from unpack):

```python
import os, subprocess, sys
STAGE_B = SAVE_DIR / 'se_resnet_from_34pct' / 'best.pt'
assert STAGE_B.is_file(), STAGE_B
env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHON_BIN': sys.executable,
       'DATA_DIR': str(DATA_DIR), 'SAVE_DIR': str(SAVE_DIR), 'DEVICE': os.environ['DEVICE'],
       'NUM_WORKERS': os.environ.get('NUM_WORKERS', '2'), 'INIT_CKPT': str(STAGE_B),
       'RUN_NAME': 'se_resnet_stage_c_from65', 'EPOCHS': '150', 'EARLY_STOP_PATIENCE': '12',
       'LR': '0.015', 'WARMUP_EPOCHS': '3'}
subprocess.run(['bash', 'scripts/iterate_from_stageb_stage_c_mixema.sh'], env=env, check=True)
```

Set `EARLY_STOP_PATIENCE` to `'0'` to disable early stopping. Then evaluate with `results.py` on `checkpoints/<RUN_NAME>/best.pt`.

**Stage D** (after Stage C): `bash scripts/iterate_from_stagec_stage_d.sh` with `INIT_CKPT` pointing at Stage C `best.pt` — see `CIFAR_SOTA.ipynb` or `bash scripts/local_runner.sh stage-d`.

**Locally (Mac / Linux):** from repo root, with Stage B at `checkpoints/se_resnet_from_34pct/best.pt`:

```bash
bash scripts/local_runner.sh stage-c
# smoke: EPOCHS=1 bash scripts/local_runner.sh stage-c
```

## Legacy notebook

The older **`CIFAR_Hydra.ipynb`** (WRN / Hydra ladder UI) existed before the minimal quick notebook; retrieve with  
`git show 5a0ecf2:colab/CIFAR_Hydra.ipynb > colab/CIFAR_Hydra.ipynb` if you need it.

## WRN / long Hydra ladder

Those scripts stay in `scripts/` (`hydra_ladder.sh`, `run_fifty_percent_pipeline.sh`) for local use — **not** in the quick notebook.
