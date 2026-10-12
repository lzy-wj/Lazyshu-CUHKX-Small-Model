#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

experiment=experiments/dynamic_crop
cache="$experiment/cache/ablation_dynamic_crop_32_128"
mkdir -p "$experiment/logs"

flock outputs/logs/gpu-7.lock bash -lc "
  set -euo pipefail
  export CUDA_VISIBLE_DEVICES=7
  conda run --no-capture-output -n Train python $experiment/train_scoped.py \
    --name ablation_dynamic_crop_visual_f0 \
    --fold 0 --mode visual --initialization kinetics \
    --weights .tools/checkpoints/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth \
    --cache-dir $cache --frames 32 --image-size 128 \
    --epochs 30 --batch-size 8 --workers 4 \
    --lr 0.0005 --backbone-lr-factor 0.1 --weight-decay 0.0001 \
    --warmup-epochs 2 --ema-decay 0.99 --dropout 0.3 \
    --auxiliary-weight 0.15 --label-smoothing 0.0 \
    --seed 42 --patience 12 --eval-every 1 --swanlab-mode cloud \
    2>&1 | tee $experiment/logs/train_visual_f0.log

  conda run --no-capture-output -n Train python $experiment/train_scoped.py \
    --name ablation_dynamic_crop_staged_f0 \
    --fold 0 --mode multimodal --initialization scratch \
    --weights .tools/checkpoints/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth \
    --init-visual $experiment/checkpoints/ablation_dynamic_crop_visual_f0/best.pt \
    --cache-dir $cache --frames 32 --image-size 128 \
    --epochs 20 --batch-size 8 --workers 4 \
    --lr 0.0003 --backbone-lr-factor 0.1 --weight-decay 0.0001 \
    --warmup-epochs 2 --ema-decay 0.99 --dropout 0.3 \
    --auxiliary-weight 0.15 --label-smoothing 0.03 \
    --seed 42 --patience 8 --eval-every 1 --swanlab-mode cloud \
    2>&1 | tee $experiment/logs/train_staged_f0.log
" 
