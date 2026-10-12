#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
gpu="$1"
source="$2"
name="$3"
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
export TORCH_HOME="$PWD/.tools/torch"
printf 'Starting %s from %s at %s\n' "$name" "$source" "$(date -Is)" | tee "outputs/logs/${name}.log"
conda run --no-capture-output -n Train python scripts/train_from_recipe.py --source "$source" --name "$name" 2>&1 | tee -a "outputs/logs/${name}.log"
status=${PIPESTATUS[0]}
printf '%s\n' "$status" > "outputs/logs/${name}.exit"
printf 'Finished at %s, exit=%s\n' "$(date -Is)" "$status" | tee -a "outputs/logs/${name}.log"
exit "$status"
