#!/usr/bin/env bash
set -euo pipefail

for scope in environment user; do
  for reset in keep reset; do
    for fold in 0 1; do
      arguments=(
        --checkpoint "outputs/checkpoints/v7_x3dm_f${fold}/best.pt"
        --fold "$fold"
        --output "outputs/study_x3dm_adabn/${scope}_${reset}_fold${fold}.npz"
        --batch-size 16
        --workers 6
        --adabn-scope "$scope"
      )
      if [[ "$reset" == reset ]]; then
        arguments+=(--adabn-reset)
      fi
      conda run --no-capture-output -n Train python scripts/evaluate_legacy_checkpoint.py "${arguments[@]}"
    done
  done
done
