#!/usr/bin/env bash
set -euo pipefail

GPU_K1="${1:?GPU for K1 required}"
GPU_K2="${2:?GPU for K2 required}"
GPU_K3="${3:?GPU for K3 required}"
HERE="$(cd "$(dirname "$0")" && pwd)"
[[ "$GPU_K1" != "$GPU_K2" && "$GPU_K1" != "$GPU_K3" && "$GPU_K2" != "$GPU_K3" ]] || {
  echo "Stage 1 requires three distinct idle GPUs" >&2
  exit 2
}
HOUR="$(TZ=Asia/Shanghai date +%H)"
if (( 10#$HOUR >= 8 && 10#$HOUR < 18 )); then
  echo "Parallel three-GPU Stage 1 is night-only; use run_one.sh sequentially during daytime" >&2
  exit 3
fi

pids=()
for item in "1:$GPU_K1" "2:$GPU_K2" "3:$GPU_K3"; do
  K="${item%%:*}"
  GPU="${item##*:}"
  "$HERE/run_one.sh" "$GPU" 0 "$K" > "$HERE/stage1_k${K}.launcher.log" 2>&1 &
  pids+=("$!")
done
printf '%s\n' "${pids[@]}" > "$HERE/stage1.pids"
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
(( status == 0 )) || { echo "At least one Fold0 run failed" >&2; exit 1; }
python "$HERE/select_fold0_k.py"
