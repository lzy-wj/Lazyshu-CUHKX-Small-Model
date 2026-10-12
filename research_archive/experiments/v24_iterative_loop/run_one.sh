#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?GPU index required}"
FOLD="${2:?fold required}"
K="${3:?K required}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
NAME="v24_iterative_k${K}_f${FOLD}"
RUN_DIR="$HERE/runs/$NAME"
mkdir -p "$RUN_DIR"

if [[ "$FOLD" == "1" ]]; then
  [[ -f "$HERE/selection.json" ]] || { echo "Fold1 requires selection.json" >&2; exit 2; }
  SELECTED="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_iterations"])' "$HERE/selection.json")"
  [[ "$K" == "$SELECTED" ]] || { echo "Fold1 K=$K rejected; Fold0 selected K=$SELECTED" >&2; exit 2; }
fi

if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v gpu="$GPU" '$1==gpu {print $2}')"; then
  echo "GPU $GPU already has a compute process; refusing launch" >&2
  exit 3
fi

HOUR="$(TZ=Asia/Shanghai date +%H)"
if (( 10#$HOUR >= 8 && 10#$HOUR < 18 )); then
  BUSY="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u | sed '/^$/d' | wc -l)"
  (( 8 - BUSY >= 5 )) || { echo "Daytime launch would leave fewer than four free GPUs" >&2; exit 3; }
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export PATH="$HOME/miniconda3/envs/Train/bin:$PATH"
INIT="$ROOT/experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_fp32_f${FOLD}.pt"
COMMAND=(python -B "$HERE/train_iterative.py"
  --name "$NAME" --fold "$FOLD" --iterations "$K" --init-checkpoint "$INIT"
  --epochs 10 --batch-size 8 --workers 4 --lr 0.00015 --backbone-lr-factor 0.1
  --weight-decay 0.0001 --warmup-epochs 1 --ema-decay 0.99
  --auxiliary-weight 0.15 --deep-supervision-decay 0.6 --feedback-temperature 2.0
  --label-smoothing 0.03 --mixup-alpha 0.3 --seed 42 --patience 10
  --output-root "$HERE/outputs" --swanlab-mode cloud)

printf '%q ' "${COMMAND[@]}" > "$RUN_DIR/command.txt"
printf '\n' >> "$RUN_DIR/command.txt"
cat > "$RUN_DIR/launch.json" <<EOF
{"name":"$NAME","gpu":$GPU,"fold":$FOLD,"iterations":$K,"started_at":"$(TZ=Asia/Shanghai date --iso-8601=seconds)","log":"$RUN_DIR/train.log"}
EOF
echo "$$" > "$RUN_DIR/launcher.pid"
exec "${COMMAND[@]}" > >(tee "$RUN_DIR/train.log") 2>&1
