#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-6}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
FOLD0="$HERE/outputs/v80_spatial_masked_encoder_f0/completion.json"
NAME="v80_spatial_masked_encoder_f1"
LOG="$HERE/outputs/logs/${NAME}.log"

[[ "$GPU" =~ ^[5-7]$ ]] || { echo "V80 may use only currently unreserved rear GPUs 5-7" >&2; exit 2; }
[[ -f "$FOLD0" ]] || { echo "Fold0 completion is required" >&2; exit 2; }
read -r BEST_EPOCH CORRECT NLL <<<"$($PYTHON - "$FOLD0" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
print(x['best_epoch'], x['metrics']['correct'], x['metrics']['nll'])
PY
)"
$PYTHON - "$CORRECT" "$NLL" <<'PY'
import sys
correct=int(sys.argv[1]); nll=float(sys.argv[2])
if correct < 490 or nll > 1.1791210174560547:
    raise SystemExit(f'Fold0 gate failed: correct={correct}, nll={nll}')
PY
UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v gpu="$GPU" '$1==gpu {print $2}')"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fqx "$UUID"; then
  echo "GPU $GPU is busy" >&2
  exit 3
fi
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
exec "$PYTHON" -u "$HERE/train.py" \
  --name "$NAME" --fold 1 \
  --init-checkpoint "$ROOT/experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_fp32_f1.pt" \
  --dimension 512 --depth 4 --heads 8 --feedforward-multiplier 3 \
  --residual-scale 0.25 --reconstruction-mask-probability 0.35 \
  --reconstruction-weight 0.2 --retention-weight 0.5 --retention-temperature 2.0 \
  --epochs "$BEST_EPOCH" --batch-size 8 --workers 4 --lr 0.00015 --weight-decay 0.0005 \
  --warmup-epochs 1 --ema-decay 0.99 --label-smoothing 0.03 --mixup-alpha 0.3 \
  --seed 42 --output-root "$HERE/outputs" --swanlab-mode cloud 2>&1 | tee "$LOG"
