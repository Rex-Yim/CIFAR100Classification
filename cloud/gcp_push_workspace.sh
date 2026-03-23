#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: cloud/gcp_push_workspace.sh <instance-name> [zone] [remote-dir]" >&2
  exit 1
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud is required but was not found." >&2
  exit 1
fi

ROOT_DIR="${ROOT_DIR:-$PWD}"
export CLOUDSDK_CONFIG="${CLOUDSDK_CONFIG:-$ROOT_DIR/.gcloud}"
mkdir -p "$CLOUDSDK_CONFIG"

INSTANCE_NAME="$1"
ZONE="${2:-${ZONE:-$(gcloud config get-value compute/zone 2>/dev/null)}}"
REMOTE_DIR="${3:-~/cifar-hydra}"
ARCHIVE_PATH="${ARCHIVE_PATH:-/tmp/cifar-hydra-workspace.tgz}"

if [[ -z "$ZONE" ]]; then
  echo "Set ZONE or pass it explicitly." >&2
  exit 1
fi

cleanup() {
  rm -f "$ARCHIVE_PATH"
}
trap cleanup EXIT

echo "Packing workspace from $ROOT_DIR..."
tar \
  --exclude=".git" \
  --exclude=".venv" \
  --exclude="data" \
  --exclude="checkpoints" \
  --exclude="checkpoints_*" \
  --exclude="runs" \
  --exclude="__pycache__" \
  --exclude=".DS_Store" \
  -czf "$ARCHIVE_PATH" \
  -C "$ROOT_DIR" .

echo "Copying archive to $INSTANCE_NAME..."
gcloud compute scp \
  --zone "$ZONE" \
  "$ARCHIVE_PATH" \
  "${INSTANCE_NAME}:~/cifar-hydra-workspace.tgz"

echo "Extracting workspace on $INSTANCE_NAME to $REMOTE_DIR..."
gcloud compute ssh "$INSTANCE_NAME" \
  --zone "$ZONE" \
  --command "mkdir -p '$REMOTE_DIR' && tar -xzf ~/cifar-hydra-workspace.tgz -C '$REMOTE_DIR' && rm -f ~/cifar-hydra-workspace.tgz"

echo "Workspace pushed to $INSTANCE_NAME:$REMOTE_DIR"
