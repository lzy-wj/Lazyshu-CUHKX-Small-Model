#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
WEIGHTS="$ROOT/.tools/checkpoints/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth"

launch() {
  local session="$1"
  local gpu="$2"
  local experiment="$3"
  local name="$4"
  local cache="$5"
  local image_size="$6"
  local seed="$7"
  shift 7

  local output="$ROOT/experiments/$experiment/outputs"
  local log="$output/logs/$name.log"
  local lock="$ROOT/outputs/logs/gpu-$gpu.lock"
  mkdir -p "$output/logs"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Session already exists: $session" >&2
    return 1
  fi

  printf -v command '%q ' \
    flock "$lock" env CUDA_VISIBLE_DEVICES="$gpu" PATH="$HOME/miniconda3/envs/Train/bin:$PATH" \
    "$PYTHON" -B "$ROOT/scripts/train_r2plus1d.py" \
    --name "$name" --fold 0 \
    --mode shared_multimodal --initialization kinetics --weights "$WEIGHTS" \
    --cache-dir "$ROOT/$cache" --frames 32 --image-size "$image_size" \
    --epochs 20 --batch-size 8 --workers 4 \
    --lr 0.0003 --backbone-lr-factor 0.1 --weight-decay 0.0001 --warmup-epochs 2 \
    --ema-decay 0.99 --dropout 0.3 --auxiliary-weight 0.15 --label-smoothing 0.03 \
    --class-balance-power 0 --seed "$seed" --patience 20 --eval-every 1 \
    --mixup-alpha 0.3 --visual-strategy joint_film \
    --output-root "$output" --swanlab-mode cloud "$@"
  command+="2>&1 | tee $(printf '%q' "$log")"
  tmux new-session -d -s "$session" "cd $(printf '%q' "$ROOT") && bash -lc $(printf '%q' "$command")"
  echo "$session -> GPU $gpu -> $name"
}

launch v63-film-maskdistill 4 v63_film_maskdistill v63_film_maskdistill_f0 \
  data/processed/r2plus1d_yolo_all_visual_32_128 128 42 \
  --masked-modality-distillation-weight 0.5 --allow-method-combination
launch v65-film-pervisual 7 v65_film_pervisual_aux v65_film_pervisual_aux_f0 \
  data/processed/r2plus1d_yolo_all_visual_32_128 128 42 \
  --per-visual-auxiliary
