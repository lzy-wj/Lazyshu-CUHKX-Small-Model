#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
GPU="${1:?Usage: $0 GPU}"
V24="$ROOT/experiments/v24_full_refit/outputs"
PROGRESSIVE_CHECKPOINT="$V24/checkpoints/v24_full_progressive160_e7/best.pt"
PSEUDO="$ROOT/experiments/scene_context_v13/outputs/predictions/v14_test_pseudo_labels.npz"
PROGRESSIVE="$V24/predictions/v44_progressive_exact_adapt.npz"
PROGRESSIVE_ADAPTER="$V24/final/v44_progressive_exact_head_delta.pt"
INPUT="$V24/predictions/v44_progressive_compound_exact.npz"
DECODED="$V24/predictions/v44_progressive_compound_exact_pair_aware.npz"
SUBMISSION="$V24/submissions/v44_progressive_compound_exact_do_not_submit.csv"
REPORT="$V24/reports/v44_progressive_compound_exact_pair_aware.json"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_160"
mkdir -p "$HERE/logs"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B \
  "$ROOT/experiments/scene_context_v13/adapt_pseudo_test.py" \
  --checkpoint "$PROGRESSIVE_CHECKPOINT" --pseudo-labels "$PSEUDO" --scope heads \
  --epochs 2 --lr 0.0002 --confidence-threshold 0.6 --retention-weight 0.25 \
  --batch-size 16 --workers 4 --seed 42 --cache-dir "$CACHE" --frames 32 \
  --tta-hflip --tta-crop-scale 0.75 --output "$PROGRESSIVE" \
  --adapter-output "$PROGRESSIVE_ADAPTER" 2>&1 | tee "$HERE/logs/exact_test.log"
"$PYTHON" -B "$HERE/build_test.py" \
  --main "$V24/predictions/v32_domain_adaptation_gate.npz" \
  --progressive "$PROGRESSIVE" \
  --x3dl "$ROOT/outputs/predictions/v9_x3dl_soup_full_cropblend50.npz" \
  --skeleton-auxiliary "$V24/predictions/v37_auxiliary_adabn.npz" --output "$INPUT"
"$PYTHON" -B "$ROOT/experiments/scene_context_v13/build_pair_aware_submission.py" \
  --input "$INPUT" --output "$DECODED" --submission "$SUBMISSION" --report "$REPORT" \
  --template-strength 1.8 --gap-penalty 1.0 --maximum-edits 6 --alignment-beam-size 40

"$PYTHON" - "$V24/submissions/v33_domain_gate_calibrated_pair_aware.csv" "$SUBMISSION" <<'PY'
import sys
import pandas as pd

left = pd.read_csv(sys.argv[1]).set_index("path")["prediction"]
right = pd.read_csv(sys.argv[2]).set_index("path")["prediction"]
if not left.index.equals(right.index):
    raise RuntimeError("V33 and V44 paths are not aligned")
print({"changed_vs_v33": int((left != right).sum())})
PY
