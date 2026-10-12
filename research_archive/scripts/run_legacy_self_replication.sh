#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

while [[ ! -f outputs/logs/legacy_staged_replication.exit ]]; do
    sleep 5
done
test "$(cat outputs/logs/legacy_staged_replication.exit)" = 0

common=(--strict-scratch --canonical-pose --modalities skeleton,imu,thermal --fusion transformer --seed 43
        --epochs 60 --batch-size 24 --frames 16 --image-size 160 --lr 0.001 --workers 4 --eval-every 2
        --patience 60 --warmup-epochs 5 --thermal-view motion_crop --imu-representation world --swanlab-mode online)
bash scripts/run_training.sh 2 rep_s43_legacy_self_f0 "${common[@]}" --fold 0 \
    --diagnostic-split configs/spatial_grouped_f0.json --encoder-sources \
    outputs/checkpoints/rep_s43_skeleton_f0/best.pt outputs/checkpoints/rep_s43_imu_f0/best.pt \
    outputs/checkpoints/rep_s43_legacy_thermal_f0/best.pt &
first=$!
bash scripts/run_training.sh 3 rep_s43_legacy_self_f1 "${common[@]}" --fold 1 \
    --diagnostic-split configs/spatial_grouped_f1.json --encoder-sources \
    outputs/checkpoints/rep_s43_skeleton_f1/best.pt outputs/checkpoints/rep_s43_imu_f1/best.pt \
    outputs/checkpoints/rep_s43_legacy_thermal_f1/best.pt &
second=$!
wait "$first" "$second"
