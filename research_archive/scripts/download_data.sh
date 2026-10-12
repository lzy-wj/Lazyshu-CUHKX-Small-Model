#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=30
export PYTHONUNBUFFERED=1
mkdir -p outputs/logs
exec flock -n outputs/logs/download.lock \
    conda run --no-capture-output -n "${CONDA_ENV_NAME:-Train}" \
    python scripts/download_data.py "$@"
