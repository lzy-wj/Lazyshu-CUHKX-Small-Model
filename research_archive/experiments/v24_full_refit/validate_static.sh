#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"
PYTHON="${PYTHON:-${HOME}/miniconda3/envs/Train/bin/python}"
cd "$ROOT"

for script in "$DIR"/*.sh; do
  bash -n "$script"
done
PYTHONPYCACHEPREFIX="/tmp/v24_full_refit_pycache" \
  "$PYTHON" -m py_compile "$DIR/predict_test.py" "$DIR/audit_release.py"
"$PYTHON" - <<'PY'
import json
from pathlib import Path

from src.data import get_records

root = Path("experiments/v24_full_refit")
for name in ("asset_budget.json", "expected_artifacts.json", "protocol.json"):
    json.loads((root / name).read_text())
protocol = json.loads((root / "protocol.json").read_text())
assert protocol["fit_scope"]["fold"] == -1
assert protocol["fit_scope"]["train_clips"] == 3036
assert abs(sum(source["weight"] for source in protocol["sources"]) - 1.0) < 1e-12
assert [source["epochs"] for source in protocol["sources"]] == [13, 16, 7, 16, 9]
records = get_records()
assert sum(record["split"] == "train" for record in records) == 3036
for cache in (
    Path("data/processed/r2plus1d_yolo_all_visual_32_128/manifest.json"),
    Path("data/processed/r2plus1d_yolo_all_visual_32_160/manifest.json"),
):
    assert json.loads(cache.read_text())["complete"] is True
assert Path("outputs/final_legal_multimodal_v9/models/yolo11n_compact.pt").is_file()
print("JSON manifests: valid")
PY

grep -q -- '--fold -1' "$DIR/run_full_no_radar.sh"
grep -q -- '--fold -1' "$DIR/run_full_warm_maskdistill.sh"
grep -q -- '0.082416 0.079504 0.148508 0.089572 0.600000' "$DIR/run_finalize.sh"
grep -q -- 'build_pair_aware_submission.py' "$DIR/run_finalize.sh"
for runner in "$DIR"/run_full_*.sh; do
  if grep -q -- '--use-radar' "$runner"; then
    echo "Unexpected radar enablement in $runner" >&2
    exit 1
  fi
  grep -q -- '--output-root "$OUTPUT_ROOT"' "$runner"
done
if grep -RInE 'kaggle competitions submit|api\.competition_submit' "$DIR" \
  --include='*.sh' --include='*.py' --exclude='validate_static.sh'; then
  echo "Unexpected Kaggle submission command found." >&2
  exit 1
fi

echo "V24 full-refit static validation passed; no GPU work was launched."
