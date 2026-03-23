#!/usr/bin/env bash
set -euo pipefail

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud is required but was not found." >&2
  exit 1
fi

ROOT_DIR="${ROOT_DIR:-$PWD}"
export CLOUDSDK_CONFIG="${CLOUDSDK_CONFIG:-$ROOT_DIR/.gcloud}"
mkdir -p "$CLOUDSDK_CONFIG"

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
BUCKET_NAME="${BUCKET_NAME:-}"
BUCKET_LOCATION="${BUCKET_LOCATION:-us-central1}"
REGION="${REGION:-us-central1}"
ZONE="${ZONE:-us-central1-c}"

if [[ -z "$PROJECT_ID" ]]; then
  echo "Set PROJECT_ID or configure gcloud first." >&2
  exit 1
fi

if [[ -z "$BUCKET_NAME" ]]; then
  echo "Set BUCKET_NAME, for example: BUCKET_NAME=my-cifar-hydra-bucket" >&2
  exit 1
fi

BUCKET_URI="gs://${BUCKET_NAME}"

echo "Using project: $PROJECT_ID"
echo "Using bucket:  $BUCKET_URI"

gcloud config set project "$PROJECT_ID" >/dev/null
gcloud config set compute/region "$REGION" >/dev/null
gcloud config set compute/zone "$ZONE" >/dev/null

echo "Enabling required APIs..."
gcloud services enable \
  compute.googleapis.com \
  storage.googleapis.com

if gcloud storage buckets describe "$BUCKET_URI" >/dev/null 2>&1; then
  echo "Bucket already exists: $BUCKET_URI"
else
  echo "Creating bucket: $BUCKET_URI"
  gcloud storage buckets create "$BUCKET_URI" \
    --project "$PROJECT_ID" \
    --location "$BUCKET_LOCATION" \
    --uniform-bucket-level-access
fi

tmpfile="$(mktemp)"
trap 'rm -f "$tmpfile"' EXIT
printf "placeholder\n" >"$tmpfile"

echo "Creating bucket prefixes..."
gcloud storage cp "$tmpfile" "$BUCKET_URI/datasets/.keep" >/dev/null
gcloud storage cp "$tmpfile" "$BUCKET_URI/checkpoints/.keep" >/dev/null
gcloud storage cp "$tmpfile" "$BUCKET_URI/logs/.keep" >/dev/null
gcloud storage cp "$tmpfile" "$BUCKET_URI/artifacts/.keep" >/dev/null

cat <<EOF
Google Cloud project bootstrap is ready.

Next commands:
  export PROJECT_ID=$PROJECT_ID
  export BUCKET_NAME=$BUCKET_NAME
  export CLOUDSDK_CONFIG=$CLOUDSDK_CONFIG
  cloud/create_gcp_spot_vm.sh
EOF
