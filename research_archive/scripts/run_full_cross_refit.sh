#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

common=(--strict-scratch --canonical-pose --modalities skeleton,imu,thermal --fusion cross_transformer --fold -1
        --batch-size 24 --frames 16 --image-size 160 --lr 0.001 --workers 4 --eval-every 2 --patience 120
        --schedule-epochs 47 --warmup-epochs 3.888888888888889 --thermal-view motion_crop --imu-representation world
        --swanlab-mode online --encoder-sources outputs/checkpoints/full_input_v1_skeleton/best.pt
        outputs/checkpoints/full_input_v1_imu_world/best.pt outputs/checkpoints/full_input_v1_thermal_motion/best.pt)

bash scripts/run_training.sh 2 full_input_v1_cross42 "${common[@]}" --seed 42 --epochs 9 &
first=$!
bash scripts/run_training.sh 3 full_input_v1_cross43 "${common[@]}" --seed 43 --epochs 7 &
second=$!
wait "$first" "$second"
