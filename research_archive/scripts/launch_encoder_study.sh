#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
conda run --no-capture-output -n Train python -c '
import json
from pathlib import Path
root=Path(".")
for filename in ["outputs/study_encoders/audit.json", "outputs/study_encoders/diagnosis/report.json"]:
    report=json.loads((root/filename).read_text())
    assert report["status"] in ["passed", "complete"], filename
for name in ["encoder_smoke_wide", "encoder_smoke_staged_concat", "encoder_smoke_staged_transformer"]:
    report=json.loads((root/"outputs/checkpoints"/name/"completion.json").read_text())
    assert report["status"] == "complete", name
    assert (root/"outputs/logs"/(name+".exit")).read_text().strip() == "0", name
manifest=json.loads((root/"outputs/predictions/encoder_pipeline_smoke_do_not_submit_manifest.json").read_text())
assert manifest["rows"] == 405 and manifest["test_accuracy"] is None
print("CPU audit, clean-checkpoint diagnosis, CUDA smoke and prediction checks passed")
'
if tmux has-session -t encoder_study 2>/dev/null; then
    printf 'encoder_study tmux session already exists; refusing a duplicate scheduler.\n'
    exit 1
fi
tmux new-session -d -s encoder_study "bash -lc 'cd ${LAZYSHU_PROJECT_ROOT}; set -o pipefail; conda run --no-capture-output -n Train python scripts/run_scratch_study.py --config configs/encoder_study.json --output outputs/study_encoders 2>&1 | tee outputs/logs/encoder_study.log; status=\${PIPESTATUS[0]}; printf \"%s\\n\" \"\$status\" > outputs/logs/encoder_study.exit; exit \"\$status\"'"
tmux list-sessions
