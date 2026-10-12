#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

while [[ ! -f outputs/logs/staged_spatial_queue.exit || ! -f outputs/checkpoints/grouped_legacy_thermal_f1/completion.json ]]; do
    sleep 5
done
test "$(cat outputs/logs/staged_spatial_queue.exit)" = 0
test "$(cat outputs/logs/grouped_legacy_thermal.exit)" = 0

common=(--strict-scratch --canonical-pose --modalities skeleton,imu,thermal --seed 42 --epochs 60 --batch-size 24
        --frames 16 --image-size 160 --lr 0.001 --workers 4 --eval-every 2 --patience 60 --warmup-epochs 5
        --thermal-view motion_crop --imu-representation world --swanlab-mode online)

bash scripts/run_training.sh 2 staged_legacy_cross_f0 "${common[@]}" --fold 0 --fusion cross_transformer \
    --diagnostic-split configs/spatial_grouped_f0.json --encoder-sources \
    outputs/checkpoints/staged_spatial_v1_skeleton_f0/best.pt \
    outputs/checkpoints/staged_spatial_v1_imu_f0/best.pt \
    outputs/checkpoints/grouped_legacy_thermal_f0/best.pt &
first=$!
bash scripts/run_training.sh 3 staged_legacy_cross_f1 "${common[@]}" --fold 1 --fusion cross_transformer \
    --diagnostic-split configs/spatial_grouped_f1.json --encoder-sources \
    outputs/checkpoints/staged_spatial_v1_skeleton_f1/best.pt \
    outputs/checkpoints/staged_spatial_v1_imu_f1/best.pt \
    outputs/checkpoints/grouped_legacy_thermal_f1/best.pt &
second=$!
bash scripts/run_training.sh 7 staged_legacy_self_f0 "${common[@]}" --fold 0 --fusion transformer \
    --diagnostic-split configs/spatial_grouped_f0.json --encoder-sources \
    outputs/checkpoints/staged_spatial_v1_skeleton_f0/best.pt \
    outputs/checkpoints/staged_spatial_v1_imu_f0/best.pt \
    outputs/checkpoints/grouped_legacy_thermal_f0/best.pt &
third=$!
wait "$first" "$second" "$third"
bash scripts/run_training.sh 2 staged_legacy_self_f1 "${common[@]}" --fold 1 --fusion transformer \
    --diagnostic-split configs/spatial_grouped_f1.json --encoder-sources \
    outputs/checkpoints/staged_spatial_v1_skeleton_f1/best.pt \
    outputs/checkpoints/staged_spatial_v1_imu_f1/best.pt \
    outputs/checkpoints/grouped_legacy_thermal_f1/best.pt
