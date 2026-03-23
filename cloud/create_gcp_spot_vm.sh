#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$PWD}"
export CLOUDSDK_CONFIG="${CLOUDSDK_CONFIG:-$ROOT_DIR/.gcloud}"
mkdir -p "$CLOUDSDK_CONFIG"

INSTANCE_NAME="${INSTANCE_NAME:-cifar-hydra-l4}"
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"
BOOT_DISK_SIZE_GB="${BOOT_DISK_SIZE_GB:-200}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-7-cu128-ubuntu-2204-nvidia-570}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"

PREFERRED_ZONES=(
  "us-central1-c"
  "us-east1-c"
  "us-west4-b"
)

if [[ -z "$PROJECT_ID" ]]; then
  echo "Set PROJECT_ID or configure gcloud first." >&2
  exit 1
fi

ZONE="${ZONE:-}"
if [[ -n "$ZONE" ]]; then
  CANDIDATE_ZONES=("$ZONE")
else
  CANDIDATE_ZONES=("${PREFERRED_ZONES[@]}")
fi

LAST_ERROR=""
for candidate in "${CANDIDATE_ZONES[@]}"; do
  if ! gcloud compute machine-types describe "$MACHINE_TYPE" --zone "$candidate" >/dev/null 2>&1; then
    echo "Skipping $candidate because $MACHINE_TYPE is not available there."
    continue
  fi

  echo "Creating Spot VM $INSTANCE_NAME in $candidate..."
  if output="$(
    gcloud compute instances create "$INSTANCE_NAME" \
      --project "$PROJECT_ID" \
      --zone "$candidate" \
      --machine-type "$MACHINE_TYPE" \
      --boot-disk-size "${BOOT_DISK_SIZE_GB}GB" \
      --boot-disk-type "pd-balanced" \
      --image-family "$IMAGE_FAMILY" \
      --image-project "$IMAGE_PROJECT" \
      --scopes "https://www.googleapis.com/auth/cloud-platform" \
      --maintenance-policy "TERMINATE" \
      --provisioning-model "SPOT" 2>&1
  )"; then
    echo "$output"
    echo "VM created in $candidate."
    echo "Next: push the repo, ssh in, run cloud/gcp_bootstrap.sh, then cloud/gcp_run_experiment.sh."
    exit 0
  fi

  echo "$output" >&2
  LAST_ERROR="$output"
  echo "Failed in $candidate, trying the next preferred zone if available..." >&2
done

echo "Unable to create $INSTANCE_NAME in any preferred zone." >&2
if [[ -n "$LAST_ERROR" ]]; then
  echo "$LAST_ERROR" >&2
fi
exit 1
