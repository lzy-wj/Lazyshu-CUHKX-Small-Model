#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

common=(--strict-scratch --canonical-pose --modalities skeleton --fusion concat
        --scratch-skeleton-encoder dstformer --skeleton-dimension 192 --skeleton-depth 3 --skeleton-heads 6
        --epochs 80 --batch-size 64 --lr 0.0007 --workers 4 --eval-every 2 --patience 24
        --warmup-epochs 5 --swanlab-mode online)

bash scripts/run_training.sh 2 dstformer_s42_f0 "${common[@]}" --seed 42 --fold 0 \
    --diagnostic-split configs/same_user_diagnostic_split.json &
first=$!
bash scripts/run_training.sh 3 dstformer_s42_f1 "${common[@]}" --seed 42 --fold 1 \
    --diagnostic-split configs/same_user_diagnostic_split_f1.json &
second=$!
bash scripts/run_training.sh 7 dstformer_aug_s42_f0 "${common[@]}" --seed 42 --fold 0 --strong-augment \
    --diagnostic-split configs/same_user_diagnostic_split.json &
third=$!

wait "$first" "$second" "$third"
