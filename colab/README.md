# Colab workflow

Train on **Google Colab** with checkpoints on **Drive**. Local training does not need this folder — use `scripts/local_runner.sh` or `python train.py`.

## Files

- **`CIFAR_Quick.ipynb`** — upload this to Colab (minimal: setup + **SE-ResNet quick path** via `scripts/quick_se_resnet_results.sh`).
- **`prepare_colab_bundle.sh`** — run on your machine to build the tarball Colab unpacks.

## Bundle (on your Mac)

```bash
colab/prepare_colab_bundle.sh
```

Creates `artifacts/cifar_hydra_project.tar.gz`. Upload it to **`My Drive/Colab_CIFAR/`** (or My Drive root).

## Colab steps

1. **Runtime → Change runtime type → GPU.**
2. Upload **`CIFAR_Quick.ipynb`** (and the bundle to Drive as above).
3. Run cells in order: **mount Drive** → **unpack** → **pip** → **GPU check** → **env** → **Train + official test**.

Data and checkpoints live under **`My Drive/Colab_CIFAR/`** (`data/`, `checkpoints/`).

## WRN / long Hydra ladder

Those scripts stay in `scripts/` (`hydra_ladder.sh`, `run_fifty_percent_pipeline.sh`) for local use — **not** in the quick notebook.
