#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
V26="$ROOT/experiments/v26_system_distillation"
V28="$ROOT/experiments/v28_tta_crop"
mkdir -p "$HERE/outputs"

for fold in 0 1; do
  "$PYTHON" -B "$HERE/blend_archives.py" \
    --adapted "$V26/outputs/predictions/v26_system_distill_f${fold}.npz" \
    --crop "$V28/outputs/v28_crop0875_f${fold}.npz" \
    --adapted-weight 0.4 --output "$HERE/outputs/v29_fused_f${fold}.npz"
done

"$PYTHON" -B "$V26/evaluate_direct_pairaware.py" \
  --fold0 "$HERE/outputs/v29_fused_f0.npz" \
  --fold1 "$HERE/outputs/v29_fused_f1.npz" \
  --output "$HERE/outputs/direct_pairaware.json"

"$PYTHON" - "$HERE/outputs/direct_pairaware.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text())
report["reference_correct"] = 1170
report["promote"] = report["combined"]["correct"] > report["reference_correct"]
report["fusion"] = {"v26_system_distill": 0.4, "v28_crop0875": 0.6}
path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
if not report["promote"]:
    raise SystemExit("V29 did not improve over V26")
PY
