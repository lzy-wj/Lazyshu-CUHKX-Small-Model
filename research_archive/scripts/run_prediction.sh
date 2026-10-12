#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
gpu="$1"
shift
exec 9>"outputs/logs/gpu-${gpu}.lock"
flock -n 9 || exit 1
used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
processes=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$processes" ]] || (( used > 1024 )); then
    printf 'GPU %s is occupied; refusing to interfere.\n' "$gpu"
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=1
script="scripts/predict.py"
if [[ "${1:-}" == "--raw" ]]; then
    shift
    script="scripts/predict_input_raw.py"
fi
conda run --no-capture-output -n Train python "$script" "$@"
