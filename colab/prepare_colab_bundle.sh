#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/artifacts}"
BUNDLE_PATH="${BUNDLE_PATH:-$OUTPUT_DIR/cifar_hydra_project.tar.gz}"

mkdir -p "$OUTPUT_DIR"

tar \
  --exclude=".git" \
  --exclude=".gcloud" \
  --exclude=".venv" \
  --exclude="data" \
  --exclude="checkpoints" \
  --exclude="checkpoints_*" \
  --exclude="runs" \
  --exclude="artifacts" \
  --exclude="__pycache__" \
  --exclude=".DS_Store" \
  -czf "$BUNDLE_PATH" \
  -C "$ROOT_DIR" .

echo "Created Colab bundle at: $BUNDLE_PATH"
