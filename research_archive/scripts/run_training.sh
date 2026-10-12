#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
gpu="$1"
name="$2"
shift 2
mkdir -p outputs/logs
exec 9>"outputs/logs/gpu-${gpu}.lock"
flock -n 9 || exit 1
used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits) || exit 1
processes=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader) || exit 1
if [[ -n "$processes" ]]; then
    printf 'GPU %s has active compute processes; refusing to interfere.\n' "$gpu"
    exit 1
fi
if (( used > 1024 )); then
    printf 'GPU %s is occupied (%s MiB); refusing to interfere.\n' "$gpu" "$used"
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCH_HOME="$PWD/.tools/torch"
printf 'Starting %s on GPU %s at %s\n' "$name" "$gpu" "$(date -Is)" | tee "outputs/logs/${name}.log"
conda run --no-capture-output -n Train python scripts/train.py --name "$name" "$@" 2>&1 | tee -a "outputs/logs/${name}.log"
status=${PIPESTATUS[0]}
printf '%s\n' "$status" > "outputs/logs/${name}.exit"
printf 'Finished at %s, exit=%s\n' "$(date -Is)" "$status" | tee -a "outputs/logs/${name}.log"
exit "$status"
