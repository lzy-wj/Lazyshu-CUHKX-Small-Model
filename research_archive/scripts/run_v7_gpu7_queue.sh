#!/usr/bin/env bash
set -euo pipefail

while [[ ! -f outputs/logs/r2_depth_ir_32f_staged_full.exit ]]; do
  sleep 20
done
[[ "$(cat outputs/logs/r2_depth_ir_32f_staged_full.exit)" == "0" ]]

exec 9>outputs/logs/gpu-7.lock
flock 9

for fold in 0 1; do
  checkpoint="outputs/checkpoints/r2_staged_soup_f${fold}_w075/best_int8.pt"
  CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n Train \
    python scripts/evaluate_r2plus1d_checkpoint.py \
    --checkpoint "$checkpoint" --fold "$fold" \
    --cache-dir data/processed/r2plus1d_yolo_32_128 \
    --output "outputs/study_r2plus1d/r2_staged_soup_f${fold}_w075_no_tta.npz"
  CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n Train \
    python scripts/evaluate_r2plus1d_checkpoint.py \
    --checkpoint "$checkpoint" --fold "$fold" \
    --cache-dir data/processed/r2plus1d_yolo_16_128 --frames 16 --tta-hflip \
    --output "outputs/study_r2plus1d/r2_staged_soup_f${fold}_w075_16f_hflip.npz"
  for scale in 0.875 0.75; do
    tag="${scale/./}"
    CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n Train \
      python scripts/evaluate_r2plus1d_checkpoint.py \
      --checkpoint "$checkpoint" --fold "$fold" \
      --cache-dir data/processed/r2plus1d_yolo_32_128 --tta-hflip --tta-crop-scale "$scale" \
      --output "outputs/study_r2plus1d/r2_staged_soup_f${fold}_w075_crop${tag}.npz"
  done
done

conda run --no-capture-output -n Train python scripts/interpolate_r2plus1d_checkpoints.py \
  --left outputs/checkpoints/r2_staged_multimodal_full/best.pt \
  --right outputs/checkpoints/r2_depth_ir_32f_staged_full/best.pt --right-weight 0.75 \
  --output outputs/checkpoints/r2_staged_soup_full_w075/best_int8.pt

checkpoint=outputs/checkpoints/r2_staged_soup_full_w075/best_int8.pt
CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n Train \
  python scripts/predict_r2plus1d.py --checkpoints "$checkpoint" \
  --cache-dir data/processed/r2plus1d_yolo_32_128 --tta-hflip \
  --name r2_staged_soup_full_32_hflip
CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n Train \
  python scripts/predict_r2plus1d.py --checkpoints "$checkpoint" \
  --cache-dir data/processed/r2plus1d_yolo_16_128 --frames 16 --tta-hflip \
  --name r2_staged_soup_full_16_hflip
for scale in 0.875 0.75; do
  tag="${scale/./}"
  CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n Train \
    python scripts/predict_r2plus1d.py --checkpoints "$checkpoint" \
    --cache-dir data/processed/r2plus1d_yolo_32_128 --tta-hflip --tta-crop-scale "$scale" \
    --name "r2_staged_soup_full_crop${tag}"
done

CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 \
  conda run --no-capture-output -n Train python scripts/train_r2plus1d.py \
  --name r2_shared_visual_multimodal_f0 --fold 0 --mode shared_multimodal \
  --initialization kinetics --cache-dir data/processed/r2plus1d_yolo_all_visual_16_128 \
  --frames 16 --image-size 128 --epochs 30 --batch-size 8 --workers 4 \
  --lr 0.0005 --backbone-lr-factor 0.1 --weight-decay 0.0001 --warmup-epochs 2 \
  --ema-decay 0.99 --dropout 0.3 --auxiliary-weight 0.15 --label-smoothing 0.03 \
  --class-balance-power 0.25 --seed 42 --patience 10 --swanlab-mode cloud
