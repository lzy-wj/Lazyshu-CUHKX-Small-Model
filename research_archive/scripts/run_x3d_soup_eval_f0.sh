#!/usr/bin/env bash
set -euo pipefail

common=(--fold 0 --batch-size 8 --workers 4)
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n Train \
  python scripts/evaluate_legacy_checkpoint.py \
  --checkpoint outputs/checkpoints/v3_x3d_f0/best.pt "${common[@]}" --tta-hflip \
  --output outputs/study_r2plus1d/x3d_s_f0_hflip.npz
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n Train \
  python scripts/evaluate_legacy_checkpoint.py \
  --checkpoint outputs/checkpoints/v7_x3dm_f0/best.pt "${common[@]}" \
  --output outputs/study_r2plus1d/x3d_m_f0.npz
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n Train \
  python scripts/evaluate_legacy_checkpoint.py \
  --checkpoint outputs/checkpoints/v7_x3dm_f0/best.pt "${common[@]}" --tta-hflip \
  --output outputs/study_r2plus1d/x3d_m_f0_hflip.npz
for weight in 025 050 075; do
  checkpoint="outputs/checkpoints/x3d_sm_soup_f0_w${weight}/best_int8.pt"
  CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n Train \
    python scripts/evaluate_legacy_checkpoint.py --checkpoint "$checkpoint" "${common[@]}" \
    --output "outputs/study_r2plus1d/x3d_sm_soup_f0_w${weight}.npz"
  CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n Train \
    python scripts/evaluate_legacy_checkpoint.py --checkpoint "$checkpoint" "${common[@]}" --tta-hflip \
    --output "outputs/study_r2plus1d/x3d_sm_soup_f0_w${weight}_hflip.npz"
done
