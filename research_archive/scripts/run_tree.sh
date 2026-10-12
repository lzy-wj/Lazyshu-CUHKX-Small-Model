#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
gpu="$1"
name="$2"
shift 2
mkdir -p outputs/logs
exec 9>"outputs/logs/gpu-${gpu}.lock"
flock -n 9 || exit 1
used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
if (( used > 1024 )); then
    printf 'GPU %s is occupied (%s MiB); refusing to interfere.\n' "$gpu" "$used"
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
printf 'Starting %s on GPU %s at %s\n' "$name" "$gpu" "$(date -Is)" | tee "outputs/logs/${name}.log"
conda run --no-capture-output -n Train python scripts/train_tree.py --name "$name" "$@" 2>&1 | tee -a "outputs/logs/${name}.log"
status=${PIPESTATUS[0]}
printf '%s\n' "$status" > "outputs/logs/${name}.exit"
printf 'Finished at %s, exit=%s\n' "$(date -Is)" "$status" | tee -a "outputs/logs/${name}.log"
exit "$status"
