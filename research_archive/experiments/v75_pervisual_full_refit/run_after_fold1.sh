#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
SOURCE="$ROOT/experiments/v65_film_pervisual_aux/outputs"
COMPLETION="$SOURCE/checkpoints/v65_film_pervisual_aux_f1/completion.json"
OUTPUT="$ROOT/experiments/v75_pervisual_full_refit/outputs"
WEIGHTS="$ROOT/.tools/checkpoints/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth"
mkdir -p "$OUTPUT/logs"

until [[ -f "$COMPLETION" ]]; do sleep 60; done
CUDA_VISIBLE_DEVICES='' "$PYTHON" -B "$ROOT/experiments/nightly_20260910/evaluate.py" \
  --slug v65_film_pervisual_aux --fold 1 \
  2>&1 | tee "$SOURCE/logs/fold1_evaluate.log"

read -r CORRECT EPOCHS <<< "$(CUDA_VISIBLE_DEVICES='' "$PYTHON" - <<'PY'
import json
from pathlib import Path
root = Path.cwd()
metrics = []
for fold in (0, 1):
    folder = root / f'experiments/v65_film_pervisual_aux/outputs/checkpoints/v65_film_pervisual_aux_f{fold}'
    completion = json.loads((folder / 'completion.json').read_text())
    rows = 652 if fold == 0 else 705
    metrics.append((round(completion['best_accuracy'] * rows), completion['best_epoch']))
print(sum(value[0] for value in metrics), round(sum(value[1] for value in metrics) / 2))
PY
)"

if (( CORRECT <= 1010 )); then
  printf 'Rejected: V65 raw combined %s does not exceed V55 raw combined 1010\n' "$CORRECT" \
    | tee "$OUTPUT/REJECTED.txt"
  exit 0
fi

flock "$ROOT/outputs/logs/gpu-6.lock" env CUDA_VISIBLE_DEVICES=6 \
  "$PYTHON" -B "$ROOT/scripts/train_r2plus1d.py" \
  --name v75_pervisual_full_e${EPOCHS} --fold -1 \
  --mode shared_multimodal --initialization kinetics --weights "$WEIGHTS" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" \
  --frames 32 --image-size 128 --epochs "$EPOCHS" --batch-size 8 --workers 4 \
  --lr 0.0003 --backbone-lr-factor 0.1 --weight-decay 0.0001 --warmup-epochs 2 \
  --ema-decay 0.99 --dropout 0.3 --auxiliary-weight 0.15 --label-smoothing 0.03 \
  --class-balance-power 0 --seed 42 --patience 20 --eval-every 1 \
  --mixup-alpha 0.3 --visual-strategy joint_film --per-visual-auxiliary \
  --output-root "$OUTPUT" --swanlab-mode cloud \
  2>&1 | tee "$OUTPUT/logs/v75_pervisual_full_e${EPOCHS}.log"
