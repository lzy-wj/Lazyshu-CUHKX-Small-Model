#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
gpu="${1:-2}"
mkdir -p outputs/logs
exec 9>"outputs/logs/gpu-${gpu}.lock"
flock -n 9 || exit 1
used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
if (( used > 1024 )); then
    printf 'GPU %s is occupied (%s MiB); refusing to interfere.\n' "$gpu" "$used"
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
conda run --no-capture-output -n Train python scripts/diagnose_scratch_encoders.py
