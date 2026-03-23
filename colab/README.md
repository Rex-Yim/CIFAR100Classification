# Colab Workflow

This folder is **only** for the Google Colab path (notebook + tarball bundle). For training on your own machine with local `checkpoints/` and `./data`, use the root **README** section *Training paths* and `scripts/hydra_m1_lite.sh` / `python train.py` — you do **not** need Colab or this bundle.

This folder contains a Colab-first training path so you can run **CIFAR Hydra** without waiting for Google Cloud GPU quota.

## Files

- `CIFAR_Hydra_Colab.ipynb`: upload this notebook to Google Colab and run the cells
- `prepare_colab_bundle.sh`: packages the current workspace into one uploadable archive for Colab

## Local step

Create the bundle from this repo:

```bash
colab/prepare_colab_bundle.sh
```

This writes:

```text
artifacts/cifar_hydra_colab_bundle.tar.gz
```

## Baseline repro (~34% test @ 4 epochs)

The notebook includes optional cells that run `scripts/repro_34pct_4epoch_baseline.sh` — the same **small SE-ResNet** recipe as historic `checkpoints_cpu_4ep` (not the WRN Hydra ladder). Run those cells **before** `run-c` / `run-d` if you want Colab to match that baseline.

## Colab steps

1. Open Google Colab.
2. Upload `colab/CIFAR_Hydra_Colab.ipynb`.
3. In Colab, switch runtime to **GPU**.
4. Run the notebook cells in order.
5. When prompted, upload `artifacts/cifar_hydra_colab_bundle.tar.gz`.

The notebook will:

- mount Google Drive
- store data/checkpoints in Drive so they survive restarts
- unpack this project into `/content/cifar-hydra`
- install dependencies
- run Hydra experiments with `scripts/hydra_ladder.sh`
- evaluate the final checkpoint with `results.py`

## Recommended run order in Colab

1. `run-c`
2. `run-d`
3. `run-e`
4. `select-best`
5. `run-f`

That skips the weakest early baselines and gets you to the strongest comparison runs faster.
