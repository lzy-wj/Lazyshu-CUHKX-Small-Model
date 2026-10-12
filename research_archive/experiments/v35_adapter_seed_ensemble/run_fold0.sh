#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
GPU="${1:?Usage: $0 GPU}"
CHECKPOINT="$ROOT/experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f0.pt"
PSEUDO="$ROOT/experiments/scene_context_v13/outputs/predictions/v13_pairaware_validation_f0.npz"
SEED42="$ROOT/experiments/v24_pseudo_validation/outputs/v23_warm60_pseudo_adapt_f0.npz"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"
OUTPUT="$HERE/outputs"
LOGS="$HERE/logs"
mkdir -p "$OUTPUT" "$LOGS"

for seed in 43 44; do
  archive="$OUTPUT/seed${seed}_f0.npz"
  if [[ ! -f "$archive" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B \
      "$ROOT/experiments/scene_context_v13/adapt_pseudo_target.py" \
      --checkpoint "$CHECKPOINT" --pseudo-labels "$PSEUDO" --fold 0 \
      --scope heads --epochs 2 --lr 0.0002 --confidence-threshold 0.6 \
      --retention-weight 0.25 --batch-size 16 --workers 4 --seed "$seed" \
      --cache-dir "$CACHE" --frames 32 --tta-hflip --tta-crop-scale 0.75 \
      --output "$archive" 2>&1 | tee "$LOGS/seed${seed}_f0.log"
  fi
done

declare -A MEMBERS
MEMBERS[seed42_43]="$SEED42 $OUTPUT/seed43_f0.npz"
MEMBERS[seed42_44]="$SEED42 $OUTPUT/seed44_f0.npz"
MEMBERS[seed43_44]="$OUTPUT/seed43_f0.npz $OUTPUT/seed44_f0.npz"
MEMBERS[seed42_43_44]="$SEED42 $OUTPUT/seed43_f0.npz $OUTPUT/seed44_f0.npz"

"$PYTHON" -B "$ROOT/experiments/v30_hard_pseudo_sweep/score_archive.py" \
  --archive "$SEED42" --fold 0 --output "$OUTPUT/seed42_f0_score.json"
for name in seed43 seed44; do
  "$PYTHON" -B "$ROOT/experiments/v30_hard_pseudo_sweep/score_archive.py" \
    --archive "$OUTPUT/${name}_f0.npz" --fold 0 --output "$OUTPUT/${name}_f0_score.json"
done
for name in seed42_43 seed42_44 seed43_44 seed42_43_44; do
  read -r -a inputs <<< "${MEMBERS[$name]}"
  "$PYTHON" -B "$HERE/build_ensemble.py" --inputs "${inputs[@]}" \
    --output "$OUTPUT/${name}_f0.npz"
  "$PYTHON" -B "$ROOT/experiments/v30_hard_pseudo_sweep/score_archive.py" \
    --archive "$OUTPUT/${name}_f0.npz" --fold 0 --output "$OUTPUT/${name}_f0_score.json"
done

"$PYTHON" - "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
members = {
    "seed42": [42],
    "seed43": [43],
    "seed44": [44],
    "seed42_43": [42, 43],
    "seed42_44": [42, 44],
    "seed43_44": [43, 44],
    "seed42_43_44": [42, 43, 44],
}
rows = []
for name, seeds in members.items():
    item = json.loads((root / f"{name}_f0_score.json").read_text())
    item.update({"name": name, "seeds": seeds})
    rows.append(item)
rows.sort(
    key=lambda item: (
        item["direct_pairaware_correct"],
        item["raw_correct"],
        -len(item["seeds"]),
    ),
    reverse=True,
)
summary = {
    "selection": "Highest direct pair-aware, then raw, then fewer adapters.",
    "reference": {"raw_correct": 493, "direct_pairaware_correct": 550, "seeds": [42]},
    "selected": rows[0],
    "ranking": rows,
}
(root / "fold0_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
