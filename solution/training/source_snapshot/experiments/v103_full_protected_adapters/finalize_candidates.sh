#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
DEST="$HERE/candidates"
mkdir -p "$DEST"
cd "$ROOT"
"$PYTHON" -u -B "$HERE/build_candidates.py" \
  --v89 "$ROOT/experiments/v89_pseudo_multiscale/outputs/v89_equal_multiscale_raw.npz" \
  --intermediate "$HERE/predictions/v103_full_protected_intermediate.npz" \
  --dual "$HERE/predictions/v103_full_protected_dual.npz" \
  --output-dir "$DEST"
for name in v103_v89_intermediate_equal v103_v89_dual_equal v103_v89_adapter_equal; do
  "$PYTHON" -u -B "$ROOT/experiments/scene_context_v13/build_pair_aware_submission.py" \
    --input "$DEST/${name}_raw.npz" \
    --output "$DEST/${name}_pair.npz" \
    --submission "$DEST/${name}_do_not_submit.csv" \
    --report "$DEST/${name}_pair.json"
done
"$PYTHON" - "$ROOT/experiments/v89_pseudo_multiscale/outputs/v89_equal_multiscale_pair_aware.npz" "$DEST" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np
reference = np.load(sys.argv[1])["probabilities"].argmax(axis=1)
destination = Path(sys.argv[2])
report = {}
for path in sorted(destination.glob("v103_*_pair.npz")):
    values = np.load(path)["probabilities"].argmax(axis=1)
    report[path.stem] = {"changes_vs_v89_pair": int((values != reference).sum())}
(destination / "candidate_audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
