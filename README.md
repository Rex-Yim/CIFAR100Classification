# MAEG3080 CIFAR-100 Final Project Starter

This repository now supports three training paths:

- `colab`: the fastest practical GPU path right now
- `wrn_hydra`: the main Google-Cloud-ready WideResNet Hydra path
- `se_resnet`: the older fallback baseline

Core files:

- `train.py`: training loop with validation splits, resume support, and full-train final mode
- `model.py`: `wrn_hydra` and `se_resnet` architectures
- `results.py`: loads a saved checkpoint and evaluates on the official test set
- `utils.py`: dataset loading, stratified val split, augmentations, and mixup/cutmix helpers
- `scripts/hydra_ladder.sh`: the planned Hydra ablation ladder
- `scripts/hydra_m1_lite.sh`: a lower-compute Apple Silicon friendly training preset
- `scripts/select_best_run.py`: picks the best run from saved histories
- `colab/CIFAR_Hydra.ipynb`: uploadable Colab notebook
- `colab/prepare_colab_bundle.sh`: packages the workspace for Colab upload
- `scripts/repro_34pct_4epoch_baseline.sh`: **SE-ResNet** 4-epoch / test-eval baseline (matches historic ~34% local run; Colab notebook can run this first)
- `scripts/iterate_from_34pct.sh`: **finetune after the baseline** — loads `best.pt` with `--init-from` (weights only), fresh LR schedule, RandAugment, default **60** epochs (faster than WRN ladder from scratch)
- `scripts/quick_se_resnet_results.sh`: **one command** — baseline if needed → **30-epoch** finetune (default) → **`results.py`** (fast path for a number to report)
- `scripts/local_runner.sh`: **single entrypoint** for Cursor / terminal (`env`, `install`, `baseline`, `ladder`, `select-best`, `final`, `results`) — same flows as Colab without `google.colab`
- `cloud/setup_gcp_project.sh`: enables APIs and creates the GCS bucket/prefixes
- `cloud/create_gcp_spot_vm.sh`: helper to create the Google Cloud Spot VM
- `cloud/gcp_push_workspace.sh`: uploads this workspace to the VM
- `cloud/gcp_bootstrap.sh`: cloud bootstrap helper
- `cloud/gcp_run_experiment.sh`: cloud training runner with GCS syncing
- `report/report.tex`: LaTeX report template
- `slides/presentation.tex`: Beamer presentation template

## Training paths (pick one)

These are **separate workflows**. They share the same Python entrypoints (`train.py`, `results.py`, `model.py`, `utils.py`) but use different machines and where artifacts live.

| Path | Where it runs | Outputs / state | Start here |
|------|----------------|-----------------|------------|
| **Local** | Your laptop or workstation | `./data/`, `./checkpoints*/`, `./runs/` (all gitignored) | [§1 Install](#1-install-dependencies) → [§1b Local runner](#1b-local-runner-cursor--terminal) or [§2](#2-run-the-current-best-local-style-hydra-configuration) or [§6 M1-lite](#6-m1--low-compute-local-run) |
| **Colab** | Google Colab + optional Drive | Notebook-managed paths; bundle upload | [`colab/README.md`](colab/README.md) and [§7](#7-google-colab-workflow) |
| **Cloud VM** | GCP Spot VM + GCS | Synced via `gsutil` in `cloud/gcp_run_experiment.sh` | [§8](#8-google-cloud-workflow) |

**Colab-only helpers** live under `colab/` (notebook + `prepare_colab_bundle.sh`). **Local convenience** lives under `scripts/` (`hydra_m1_lite.sh`, `hydra_ladder.sh`, etc.). Neither replaces the other: Colab needs the bundle step; local does not.

## 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1b. Local runner (Cursor / terminal)

Use this instead of the Colab notebook when you already have the repo on disk. It sets `DATA_DIR` / `SAVE_DIR` under the project, picks **cuda → mps → cpu** unless you set `DEVICE`, and wraps the same shell scripts the notebook calls.

```bash
source .venv/bin/activate
bash scripts/local_runner.sh help
bash scripts/local_runner.sh env
bash scripts/local_runner.sh install          # optional if deps missing
bash scripts/local_runner.sh baseline       # 4-epoch SE-ResNet sanity check
bash scripts/local_runner.sh ladder run-d   # main WRN-Hydra run (long)
bash scripts/local_runner.sh select-best
bash scripts/local_runner.sh final          # run-f after auto-picking best of c/d/e
bash scripts/local_runner.sh results        # test accuracy on hydra_run_f_final/best.pt
```

## 2. Run the current best local-style Hydra configuration

Example validation-based run:

```bash
python3 train.py \
  --model-name wrn_hydra \
  --depth 28 \
  --widen-factor 10 \
  --attention eca \
  --downsample-mode antialias \
  --aug randaugment \
  --randaugment-n 2 \
  --randaugment-m 9 \
  --random-erasing-p 0.1 \
  --mix-mode both \
  --mixup-alpha 0.2 \
  --cutmix-alpha 1.0 \
  --mix-prob 1.0 \
  --label-smoothing 0.0 \
  --optimizer sgd \
  --lr 0.1 \
  --warmup-epochs 5 \
  --epochs 240 \
  --eval-split val \
  --val-ratio 0.1 \
  --run-name hydra_run_d
```

This creates a run directory like:

```text
./checkpoints/hydra_run_d/
  args.json
  best.pt
  last.pt
  history.json
```

To resume safely:

```bash
python3 train.py --run-name hydra_run_d --resume latest ...
```

## 3. Final full-train run

After selecting the best validation run, retrain on the full `50k` training set and evaluate on the official test set:

```bash
BEST_RUN=hydra_run_d scripts/hydra_ladder.sh run-f
```

## 4. Evaluate the required `results.py`

```bash
python3 results.py --checkpoint ./checkpoints/hydra_run_f_final/best.pt
```

Optional CSV export:

```bash
python3 results.py \
  --checkpoint ./checkpoints/hydra_run_f_final/best.pt \
  --predictions-file ./outputs/test_predictions.csv
```

## 5. Hydra experiment ladder

Run the planned ablations one by one:

```bash
scripts/hydra_ladder.sh run-a
scripts/hydra_ladder.sh run-b
scripts/hydra_ladder.sh run-c
scripts/hydra_ladder.sh run-d
scripts/hydra_ladder.sh run-e
scripts/hydra_ladder.sh select-best
```

Or run the ablation block sequentially:

```bash
scripts/hydra_ladder.sh all-ablations
```

## 6. M1 / low-compute local run

If you want a smarter local run instead of brute-forcing a huge model on CPU, use the M1-lite preset:

```bash
scripts/hydra_m1_lite.sh
```

This preset makes three deliberate tradeoffs:

- uses a smaller `wrn_hydra` backbone (`depth=22`, `widen-factor=4`)
- keeps the high-value recipe pieces (`RandAugment`, `Mixup/CutMix`, `EMA`)
- adds a **hierarchical coarse-label loss** that exploits CIFAR-100's built-in 20 superclasses for extra supervision at low cost

You can override the defaults if needed:

```bash
NUM_THREADS=8 EPOCHS=140 BATCH_SIZE=64 scripts/hydra_m1_lite.sh
```

## 7. Google Colab workflow

This is the fastest route while waiting on Google Cloud GPU quota.

Create the upload bundle locally:

```bash
colab/prepare_colab_bundle.sh
```

This writes:

```text
./artifacts/cifar_hydra_project.tar.gz
```

Then:

1. Open Google Colab and switch the runtime to **GPU**.
2. Upload `colab/CIFAR_Hydra.ipynb` (or create a new Colab notebook and paste cells).
3. Run the notebook cells in order.
4. Put `artifacts/cifar_hydra_project.tar.gz` on Drive under `Colab_CIFAR/` (see notebook).

The notebook stores datasets and checkpoints in Google Drive, so Colab restarts are less painful.

Recommended experiment order on Colab:

```bash
run-c
run-d
run-e
select-best
run-f
```

## 8. Google Cloud workflow

Training should use Google Cloud, not Cloudflare.

Prepare the Google Cloud project and bucket:

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export BUCKET_NAME=YOUR_BUCKET_NAME
export CLOUDSDK_CONFIG=$PWD/.gcloud
cloud/setup_gcp_project.sh
```

Create the Spot VM:

```bash
cloud/create_gcp_spot_vm.sh
```

Push this workspace to the VM:

```bash
cloud/gcp_push_workspace.sh cifar-hydra-l4
```

On the VM:

```bash
cd ~/cifar-hydra
cloud/gcp_bootstrap.sh
cloud/gcp_run_experiment.sh gs://YOUR_BUCKET hydra_run_d \
  --model-name wrn_hydra \
  --depth 28 \
  --widen-factor 10 \
  --attention eca \
  --downsample-mode antialias \
  --aug randaugment \
  --randaugment-n 2 \
  --randaugment-m 9 \
  --random-erasing-p 0.1 \
  --mix-mode both \
  --label-smoothing 0.0 \
  --epochs 240 \
  --eval-split val
```

`gcp_run_experiment.sh` syncs:

- `data/` to `gs://.../datasets`
- run checkpoints to `gs://.../checkpoints/<run-name>`
- logs to `gs://.../logs`

## Notes for the report/presentation

- The main system is now **CIFAR Hydra**: a WideResNet with attention and detail-preserving downsampling.
- Tuning should be reported on the validation split, not the official test set.
- The final reported test result should come from the full-train final run and be reproducible through `results.py`.
- Compile the report with `pdflatex report.tex` in `report/`.
- Compile the slides with `pdflatex presentation.tex` in `slides/`.
