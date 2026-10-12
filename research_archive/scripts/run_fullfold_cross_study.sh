#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

common=(--strict-scratch --canonical-pose --modalities skeleton,imu,thermal --fusion cross_transformer --epochs 60
        --batch-size 24 --frames 16 --image-size 160 --lr 0.001 --workers 4 --eval-every 2 --patience 60
        --warmup-epochs 5 --thermal-view motion_crop --imu-representation world --swanlab-mode online)

run_fold() {
    gpu=$1
    seed=$2
    fold=$3
    if [[ "$fold" = 0 ]]; then
        sources=(outputs/checkpoints/scratch_v1_skeleton_f0/best.pt outputs/checkpoints/input_v1_imu_world_f0/best.pt
                 outputs/checkpoints/input_v1_thermal_motion_f0/best.pt)
    else
        sources=(outputs/checkpoints/input_v1_skeleton_f1/best.pt outputs/checkpoints/input_v1_imu_world_f1/best.pt
                 outputs/checkpoints/input_v1_thermal_motion_f1/best.pt)
    fi
    bash scripts/run_training.sh "$gpu" "fullfold_cross_s${seed}_f${fold}" "${common[@]}" --seed "$seed" --fold "$fold" \
        --encoder-sources "${sources[@]}"
}

run_fold 2 42 0 &
first=$!
run_fold 3 42 1 &
second=$!
run_fold 7 43 0 &
third=$!
wait "$first" "$second" "$third"
run_fold 2 43 1
