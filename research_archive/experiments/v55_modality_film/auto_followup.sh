#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
FOLD0_COMPLETION="$HERE/outputs/checkpoints/v55_modality_film_f0/completion.json"
FOLD1_COMPLETION="$HERE/outputs/checkpoints/v55_modality_film_f1/completion.json"

until [[ -f "$FOLD0_COMPLETION" ]]; do sleep 60; done
"$PYTHON" -B "$HERE/evaluate.py" --fold 0 2>&1 | tee "$HERE/outputs/logs/v55_evaluate_f0.log"
if [[ "$(jq -r .advance_fold1 "$HERE/outputs/fold0_evaluation.json")" != true ]]; then
  echo "V55_STOP=fold0_gate"
  exit 0
fi

echo "V55_FOLD1_STARTED_AT=$(date --iso-8601=seconds)"
bash "$HERE/run_fold1.sh"
until [[ -f "$FOLD1_COMPLETION" ]]; do sleep 30; done
"$PYTHON" -B "$HERE/evaluate.py" --fold 1 2>&1 | tee "$HERE/outputs/logs/v55_evaluate_f1.log"
