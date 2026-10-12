#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
GPU="${1:?Usage: $0 GPU}"
CHECKPOINT="$ROOT/experiments/scene_context_v13/outputs/checkpoints/v23_shared5_joint_progressive160_maskdistill_f1/best_int8.pt"
PSEUDO="$ROOT/experiments/scene_context_v13/outputs/predictions/v13_pairaware_validation_f1.npz"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_160"
PROGRESSIVE="$HERE/outputs/progressive_f1.npz"
mkdir -p "$HERE/outputs" "$HERE/logs"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B \
  "$ROOT/experiments/scene_context_v13/adapt_pseudo_target.py" \
  --checkpoint "$CHECKPOINT" --pseudo-labels "$PSEUDO" --fold 1 \
  --scope heads --epochs 2 --lr 0.0002 --confidence-threshold 0.6 \
  --retention-weight 0.25 --batch-size 16 --workers 4 --seed 42 \
  --cache-dir "$CACHE" --frames 32 --tta-hflip --tta-crop-scale 0.75 \
  --output "$PROGRESSIVE" 2>&1 | tee "$HERE/logs/progressive_f1.log"

"$PYTHON" -B "$ROOT/experiments/v29_crop_system_fusion/blend_archives.py" \
  --adapted "$ROOT/experiments/v24_pseudo_validation/outputs/v23_warm60_pseudo_adapt_f0.npz" \
  --crop "$ROOT/experiments/v42_student_initialization_sweep/outputs/progressive_f0.npz" \
  --adapted-weight 0.9 --output "$HERE/outputs/v43_mix_f0.npz"
"$PYTHON" -B "$ROOT/experiments/v29_crop_system_fusion/blend_archives.py" \
  --adapted "$ROOT/experiments/v24_pseudo_validation/outputs/v23_warm60_pseudo_adapt_f1.npz" \
  --crop "$PROGRESSIVE" --adapted-weight 0.9 --output "$HERE/outputs/v43_mix_f1.npz"
for fold in 0 1; do
  "$PYTHON" -B "$ROOT/experiments/v30_hard_pseudo_sweep/score_archive.py" \
    --archive "$HERE/outputs/v43_mix_f${fold}.npz" --fold "$fold" \
    --output "$HERE/outputs/v43_mix_f${fold}_score.json"
done

"$PYTHON" - "$HERE/outputs/v43_mix_f0_score.json" "$HERE/outputs/v43_mix_f1_score.json" \
  "$HERE/outputs/combined.json" <<'PY'
import json
import sys
from pathlib import Path

fold0 = json.loads(Path(sys.argv[1]).read_text())
fold1 = json.loads(Path(sys.argv[2]).read_text())
correct = fold0["direct_pairaware_correct"] + fold1["direct_pairaware_correct"]
report = {
    "mixture": {"warm60_hard_adapted": 0.9, "progressive160_hard_adapted": 0.1},
    "fold0": fold0,
    "fold1": fold1,
    "combined": {"correct": correct, "rows": 1357, "accuracy": correct / 1357},
    "reference": {"correct": 1168, "rows": 1357},
    "promote": correct > 1168,
}
Path(sys.argv[3]).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
