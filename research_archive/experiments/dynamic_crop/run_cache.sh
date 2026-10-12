#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p experiments/dynamic_crop/logs
flock outputs/logs/gpu-7.lock \
  conda run --no-capture-output -n Train \
  python experiments/dynamic_crop/prepare_dynamic_crop_cache.py \
    --device 7 --stage "$1" \
  2>&1 | tee "experiments/dynamic_crop/logs/cache_${1}.log"
