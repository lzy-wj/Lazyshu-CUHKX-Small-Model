#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
WEIGHTS="$ROOT/.tools/checkpoints/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth"
SLUGS=(
  v63_film_maskdistill
  v65_film_pervisual_aux
)

name_for() {
  case "$1" in
    v63_film_maskdistill) echo v63_film_maskdistill_f1 ;;
    v65_film_pervisual_aux) echo v65_film_pervisual_aux_f1 ;;
  esac
}

gpu_for() {
  case "$1" in
    v63_film_maskdistill) echo 4 ;;
    v65_film_pervisual_aux) echo 6 ;;
  esac
}

extra_for() {
  case "$1" in
    v63_film_maskdistill) echo '--masked-modality-distillation-weight 0.5 --allow-method-combination' ;;
    v64_film_invariant_skeleton) echo '--skeleton-representation invariant' ;;
    v65_film_pervisual_aux) echo '--per-visual-auxiliary' ;;
    *) echo '' ;;
  esac
}

for slug in "${SLUGS[@]}"; do
  name="${slug}_f0"
  completion="$ROOT/experiments/$slug/outputs/checkpoints/$name/completion.json"
  until [[ -f "$completion" ]]; do sleep 60; done
  "$PYTHON" -B "$HERE/evaluate.py" --slug "$slug" --fold 0 \
    2>&1 | tee "$ROOT/experiments/$slug/outputs/logs/fold0_evaluate.log"
done

"$PYTHON" -B "$HERE/select_candidates.py" 2>&1 | tee "$HERE/selection.log"

while IFS= read -r slug; do
  gpu="$(gpu_for "$slug")"
  name="$(name_for "$slug")"
  output="$ROOT/experiments/$slug/outputs"
  log="$output/logs/$name.log"
  session="${slug}-f1"
  completion="$output/checkpoints/$name/completion.json"
  if [[ -f "$completion" ]] || tmux has-session -t "$session" 2>/dev/null; then
    echo "Fold 1 already active or complete: $slug"
    continue
  fi
  lock="$ROOT/outputs/logs/gpu-$gpu.lock"
  cache=data/processed/r2plus1d_yolo_all_visual_32_128
  image_size=128
  seed=42
  read -r -a extra <<< "$(extra_for "$slug")"
  printf -v command '%q ' \
    flock "$lock" env CUDA_VISIBLE_DEVICES="$gpu" PATH="$HOME/miniconda3/envs/Train/bin:$PATH" \
    "$PYTHON" -B "$ROOT/scripts/train_r2plus1d.py" \
    --name "$name" --fold 1 \
    --mode shared_multimodal --initialization kinetics --weights "$WEIGHTS" \
    --cache-dir "$ROOT/$cache" --frames 32 --image-size "$image_size" \
    --epochs 20 --batch-size 8 --workers 4 \
    --lr 0.0003 --backbone-lr-factor 0.1 --weight-decay 0.0001 --warmup-epochs 2 \
    --ema-decay 0.99 --dropout 0.3 --auxiliary-weight 0.15 --label-smoothing 0.03 \
    --class-balance-power 0 --seed "$seed" --patience 20 --eval-every 1 \
    --mixup-alpha 0.3 --visual-strategy joint_film \
    --output-root "$output" --swanlab-mode cloud "${extra[@]}"
  command+="2>&1 | tee $(printf '%q' "$log"); "
  command+="$(printf '%q' "$PYTHON") -B $(printf '%q' "$HERE/evaluate.py") --slug $(printf '%q' "$slug") --fold 1 "
  command+="2>&1 | tee $(printf '%q' "$output/logs/fold1_evaluate.log")"
  tmux new-session -d -s "$session" "cd $(printf '%q' "$ROOT") && bash -lc $(printf '%q' "$command")"
done < <(jq -r '.selected[]' "$HERE/selection.json")
