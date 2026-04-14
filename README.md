# MAEG3080 — CIFAR-100 classification (final project)

**Course:** MAEG3080 *Fundamentals of Machine Intelligence* · **Institution:** The Chinese University of Hong Kong (CUHK), Department of Mechanical and Automation Engineering  

**Public repository:** [**github.com/Rex-Yim/CIFAR100Classification**](https://github.com/Rex-Yim/CIFAR100Classification) · [Issues](https://github.com/Rex-Yim/CIFAR100Classification/issues) · [Actions](https://github.com/Rex-Yim/CIFAR100Classification/actions)

PyTorch CIFAR-100 image classification with **WideResNet + Hydra** (`wrn_hydra`) and an **SE-ResNet** baseline (`se_resnet`). Training can run **locally**, in [**Google Colab**](https://colab.research.google.com/), or on a **GCP Spot VM** with GCS sync. Official test accuracy is reported with [`results.py`](results.py).

## Contents

- [Training paths (pick one)](#training-paths-pick-one)
- [1. Install dependencies](#1-install-dependencies)
- [1b. Local runner (Cursor / terminal)](#1b-local-runner-cursor--terminal)
- [2. Run the current best local-style Hydra configuration](#2-run-the-current-best-local-style-hydra-configuration)
- [3. Final full-train run](#3-final-full-train-run)
- [4. Evaluate `results.py`](#4-evaluate-the-required-resultspy)
- [5. Hydra experiment ladder](#5-hydra-experiment-ladder)
- [6. M1 / low-compute local run](#6-m1--low-compute-local-run)
- [7. Google Colab workflow](#7-google-colab-workflow)
- [8. Google Cloud workflow](#8-google-cloud-workflow)
- [Notes for the report / presentation](#notes-for-the-reportpresentation)
- [Repository map](#repository-map)

## Training paths (pick one)

These are **separate workflows**. They share the same Python entrypoints ([`train.py`](train.py), [`results.py`](results.py), [`model.py`](model.py), [`utils.py`](utils.py)) but use different machines and where artifacts live.

| Path | Where it runs | Outputs / state | Start here |
|------|----------------|-----------------|------------|
| **Local** | Your laptop or workstation | `./data/`, `./checkpoints*/`, `./runs/` (gitignored) | [§1 Install](#1-install-dependencies) → [§1b Local runner](#1b-local-runner-cursor--terminal) or [§2](#2-run-the-current-best-local-style-hydra-configuration) or [§6 M1-lite](#6-m1--low-compute-local-run) |
| **Colab** | Google Colab + optional Drive | Notebook-managed paths; bundle upload | [`colab/README.md`](colab/README.md) and [§7](#7-google-colab-workflow) |
| **Cloud VM** | GCP Spot VM + GCS | Synced via `gsutil` in [`cloud/gcp_run_experiment.sh`](cloud/gcp_run_experiment.sh) | [§8](#8-google-cloud-workflow) |

**Colab helpers:** [`colab/`](colab/) (notebook + [`colab/prepare_colab_bundle.sh`](colab/prepare_colab_bundle.sh)). **Local automation:** [`scripts/`](scripts/) (e.g. [`scripts/hydra_m1_lite.sh`](scripts/hydra_m1_lite.sh), [`scripts/hydra_ladder.sh`](scripts/hydra_ladder.sh)). Colab needs the bundle step; a local checkout does not.

## Repository map

| Path | Description |
|------|-------------|
| [`train.py`](train.py) | Training loop (validation splits, resume, full-train final mode) |
| [`model.py`](model.py) | `wrn_hydra` and `se_resnet` architectures |
| [`results.py`](results.py) | Load checkpoint → official **test** set accuracy (course script) |
| [`utils.py`](utils.py) | Data loading, stratified val split, augmentations, Mixup/CutMix |
| [`scripts/local_runner.sh`](scripts/local_runner.sh) | Single entrypoint for local/Cursor: `env`, `install`, `baseline`, `ladder`, … |
| [`scripts/hydra_ladder.sh`](scripts/hydra_ladder.sh) | WRN-Hydra ablation ladder + `run-f` |
| [`scripts/hydra_m1_lite.sh`](scripts/hydra_m1_lite.sh) | Apple Silicon / low-compute preset |
| [`scripts/quick_se_resnet_results.sh`](scripts/quick_se_resnet_results.sh) | SE-ResNet: baseline → finetune → `results.py` (fast reporting path) |
| [`scripts/colab_improve_se_resnet_results.sh`](scripts/colab_improve_se_resnet_results.sh) | Stage C/D improvement from ~65% checkpoint → `results.py` |
| [`colab/CIFAR_Quick.ipynb`](colab/CIFAR_Quick.ipynb) | Uploadable Colab notebook (SE-ResNet quick path) |
| [`report/report.tex`](report/report.tex) | LaTeX final report source |
| [`slides/presentation.tex`](slides/presentation.tex) | Beamer slides source |
| [`report/README.md`](report/README.md) | How to build `artifacts/report.pdf` |

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
2. Upload [`colab/CIFAR_Quick.ipynb`](colab/CIFAR_Quick.ipynb).
3. Run the notebook cells in order.
4. Put `artifacts/cifar_hydra_project.tar.gz` on Drive under `Colab_CIFAR/` (see notebook).

The notebook stores datasets and checkpoints in Google Drive, so Colab restarts are less painful.

The quick notebook runs **[`scripts/quick_se_resnet_results.sh`](scripts/quick_se_resnet_results.sh)** (baseline if needed → finetune → `results.py`). To push beyond the ~65% Stage B checkpoint on Colab, use **[`scripts/colab_improve_se_resnet_results.sh`](scripts/colab_improve_se_resnet_results.sh)**. Long **WRN / Hydra ladder** runs use [`scripts/hydra_ladder.sh`](scripts/hydra_ladder.sh) on your machine or a VM, not the Colab quick notebook.

## 8. Google Cloud workflow

Use **Google Cloud** for the GPU VM + GCS workflow below (not a generic file CDN).

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

[`cloud/gcp_run_experiment.sh`](cloud/gcp_run_experiment.sh) syncs:

- `data/` to `gs://.../datasets`
- run checkpoints to `gs://.../checkpoints/<run-name>`
- logs to `gs://.../logs`

## Notes for the report/presentation

- The **Colab quick path** uses a **two-stage SE-ResNet** (baseline, then finetune with RandAugment). Final **official test** accuracy should come from [`results.py`](results.py) (e.g. **65.17%** with the shipped quick pipeline).
- Report tuning on the **validation** split; reserve the **test** set for the final [`results.py`](results.py) number.
- **Build PDFs** (requires `pdflatex`, e.g. BasicTeX / MacTeX): see [`report/README.md`](report/README.md).

  ```bash
  python scripts/build_pdfs.py
  ```

  Output: `artifacts/report.pdf` and `artifacts/presentation.pdf` (the `artifacts/` directory is gitignored).

---

**Primary link for this project:** [https://github.com/Rex-Yim/CIFAR100Classification](https://github.com/Rex-Yim/CIFAR100Classification)
