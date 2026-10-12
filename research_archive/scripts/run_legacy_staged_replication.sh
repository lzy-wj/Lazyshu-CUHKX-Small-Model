#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

while [[ ! -f outputs/logs/legacy_staged_followup.exit ]]; do
    sleep 5
done
test "$(cat outputs/logs/legacy_staged_followup.exit)" = 0

source_common=(--strict-scratch --canonical-pose --seed 43 --epochs 60 --batch-size 24 --frames 16 --image-size 160
               --lr 0.001 --workers 4 --eval-every 2 --patience 60 --warmup-epochs 5 --swanlab-mode online)

(
  bash scripts/run_training.sh 2 rep_s43_skeleton_f0 "${source_common[@]}" --modalities skeleton --fusion concat --fold 0 \
      --diagnostic-split configs/spatial_grouped_f0.json
  bash scripts/run_training.sh 2 rep_s43_imu_f1 "${source_common[@]}" --modalities imu --fusion concat --fold 1 \
      --imu-representation world --diagnostic-split configs/spatial_grouped_f1.json
) &
first=$!
(
  bash scripts/run_training.sh 3 rep_s43_imu_f0 "${source_common[@]}" --modalities imu --fusion concat --fold 0 \
      --imu-representation world --diagnostic-split configs/spatial_grouped_f0.json
  bash scripts/run_training.sh 3 rep_s43_legacy_thermal_f1 "${source_common[@]}" --modalities thermal --fusion concat --fold 1 \
      --thermal-view motion_crop --diagnostic-split configs/spatial_grouped_f1.json
) &
second=$!
(
  bash scripts/run_training.sh 7 rep_s43_legacy_thermal_f0 "${source_common[@]}" --modalities thermal --fusion concat --fold 0 \
      --thermal-view motion_crop --diagnostic-split configs/spatial_grouped_f0.json
  bash scripts/run_training.sh 7 rep_s43_skeleton_f1 "${source_common[@]}" --modalities skeleton --fusion concat --fold 1 \
      --diagnostic-split configs/spatial_grouped_f1.json
) &
third=$!
wait "$first" "$second" "$third"

target_common=(--strict-scratch --canonical-pose --modalities skeleton,imu,thermal --fusion cross_transformer --seed 43
               --epochs 60 --batch-size 24 --frames 16 --image-size 160 --lr 0.001 --workers 4 --eval-every 2
               --patience 60 --warmup-epochs 5 --thermal-view motion_crop --imu-representation world --swanlab-mode online)
bash scripts/run_training.sh 2 rep_s43_legacy_cross_f0 "${target_common[@]}" --fold 0 \
    --diagnostic-split configs/spatial_grouped_f0.json --encoder-sources \
    outputs/checkpoints/rep_s43_skeleton_f0/best.pt outputs/checkpoints/rep_s43_imu_f0/best.pt \
    outputs/checkpoints/rep_s43_legacy_thermal_f0/best.pt &
fourth=$!
bash scripts/run_training.sh 3 rep_s43_legacy_cross_f1 "${target_common[@]}" --fold 1 \
    --diagnostic-split configs/spatial_grouped_f1.json --encoder-sources \
    outputs/checkpoints/rep_s43_skeleton_f1/best.pt outputs/checkpoints/rep_s43_imu_f1/best.pt \
    outputs/checkpoints/rep_s43_legacy_thermal_f1/best.pt &
fifth=$!
wait "$fourth" "$fifth"
