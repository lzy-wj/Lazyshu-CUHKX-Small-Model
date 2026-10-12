#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
gpu="${1:-7}"
sources=(
    outputs/checkpoints/scratch_v1_ir_f0/best.pt
    outputs/checkpoints/scratch_v1_thermal_f0/best.pt
    outputs/checkpoints/scratch_v1_depth_f0/best.pt
    outputs/checkpoints/scratch_v1_skeleton_f0/best.pt
    outputs/checkpoints/scratch_v1_imu_f0/best.pt
    outputs/checkpoints/scratch_v1_radar_f0/best.pt
)
common=(--strict-scratch --canonical-pose --epochs 1 --batch-size 8 --frames 4 --image-size 96 --workers 2 --limit-batches 2 --swanlab-mode disabled)
bash scripts/run_training.sh "$gpu" encoder_smoke_wide "${common[@]}" --modalities scene_depth --visual-multiplier 2
bash scripts/run_training.sh "$gpu" encoder_smoke_staged_concat "${common[@]}" --modalities scene_ir,thermal,scene_depth,skeleton,imu,radar --fusion concat --encoder-sources "${sources[@]}"
bash scripts/run_training.sh "$gpu" encoder_smoke_staged_transformer "${common[@]}" --modalities scene_ir,thermal,scene_depth,skeleton,imu,radar --fusion transformer --encoder-sources "${sources[@]}"
exec 9>"outputs/logs/gpu-${gpu}.lock"
flock -n 9 || exit 1
used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
if (( used > 1024 )); then exit 1; fi
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=1
conda run --no-capture-output -n Train python scripts/predict.py \
    --checkpoints outputs/checkpoints/encoder_smoke_wide/best.pt outputs/checkpoints/encoder_smoke_staged_transformer/best.pt \
    --name encoder_pipeline_smoke_do_not_submit --batch-size 8 --workers 2
