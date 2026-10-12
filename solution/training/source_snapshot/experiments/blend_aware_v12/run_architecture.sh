#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?usage: run_architecture.sh METHOD GPU [FOLD]}"
GPU="${2:?usage: run_architecture.sh METHOD GPU [FOLD]}"
FOLD="${3:-0}"
EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$EXPERIMENT_ROOT/../.." && pwd)"
CONDA="$HOME/miniconda3/bin/conda"
NAME="v12_${METHOD}_f${FOLD}"
LOG_DIR="$EXPERIMENT_ROOT/outputs/logs"
mkdir -p "$LOG_DIR"
SEED=42

case "$METHOD" in
  joint_radar)
    MODE="multimodal"
    CACHE="$REPOSITORY_ROOT/data/processed/r2plus1d_yolo_32_128"
    INITIALIZATION="scratch"
    EXTRA=(--init-visual "$REPOSITORY_ROOT/outputs/checkpoints/r2_depth_ir_32f_f${FOLD}/best.pt" --use-radar)
    ;;
  shared5)
    MODE="shared_multimodal"
    CACHE="$REPOSITORY_ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"
    INITIALIZATION="kinetics"
    EXTRA=()
    ;;
  shared6)
    MODE="shared_multimodal"
    CACHE="$REPOSITORY_ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"
    INITIALIZATION="kinetics"
    EXTRA=(--use-radar)
    ;;
  photometric)
    MODE="multimodal"
    CACHE="$REPOSITORY_ROOT/data/processed/r2plus1d_yolo_32_128"
    INITIALIZATION="scratch"
    EXTRA=(--init-visual "$REPOSITORY_ROOT/outputs/checkpoints/r2_depth_ir_32f_f${FOLD}/best.pt" --photometric-jitter 0.15)
    ;;
  mixup)
    MODE="multimodal"
    CACHE="$REPOSITORY_ROOT/data/processed/r2plus1d_yolo_32_128"
    INITIALIZATION="scratch"
    EXTRA=(--init-visual "$REPOSITORY_ROOT/outputs/checkpoints/r2_depth_ir_32f_f${FOLD}/best.pt" --mixup-alpha 0.3)
    ;;
  mixup_s43)
    MODE="multimodal"
    CACHE="$REPOSITORY_ROOT/data/processed/r2plus1d_yolo_32_128"
    INITIALIZATION="scratch"
    SEED=43
    EXTRA=(--init-visual "$REPOSITORY_ROOT/outputs/checkpoints/r2_depth_ir_32f_f${FOLD}/best.pt" --mixup-alpha 0.3)
    ;;
  centered_mixup)
    MODE="multimodal"
    CACHE="$REPOSITORY_ROOT/data/processed/r2plus1d_yolo_32_128"
    INITIALIZATION="scratch"
    EXTRA=(
      --init-visual "$REPOSITORY_ROOT/outputs/checkpoints/r2_depth_ir_32f_f${FOLD}/best.pt"
      --init-skeleton "$REPOSITORY_ROOT/outputs/checkpoints/r2_centered_pretrain_f${FOLD}/best.pt"
      --skeleton-representation centered
      --canonical-pose
      --mixup-alpha 0.3
    )
    ;;
  centered_control)
    MODE="multimodal"
    CACHE="$REPOSITORY_ROOT/data/processed/r2plus1d_yolo_32_128"
    INITIALIZATION="scratch"
    EXTRA=(
      --init-visual "$REPOSITORY_ROOT/outputs/checkpoints/r2_depth_ir_32f_f${FOLD}/best.pt"
      --init-skeleton "$REPOSITORY_ROOT/outputs/checkpoints/r2_centered_pretrain_f${FOLD}/best.pt"
      --skeleton-representation centered
      --canonical-pose
    )
    ;;
  mixstyle)
    MODE="multimodal"
    CACHE="$REPOSITORY_ROOT/data/processed/r2plus1d_yolo_32_128"
    INITIALIZATION="scratch"
    EXTRA=(
      --init-visual "$REPOSITORY_ROOT/outputs/checkpoints/r2_depth_ir_32f_f${FOLD}/best.pt"
      --mixstyle-probability 0.5
      --mixstyle-alpha 0.1
    )
    ;;
  deep_mixup)
    MODE="multimodal"
    CACHE="$REPOSITORY_ROOT/data/processed/r2plus1d_yolo_32_128"
    INITIALIZATION="scratch"
    EXTRA=(
      --init-visual "$REPOSITORY_ROOT/outputs/checkpoints/r2_depth_ir_32f_f${FOLD}/best.pt"
      --mixup-alpha 0.3
      --fusion-depth 6
    )
    ;;
  temporal_mixup)
    MODE="multimodal"
    CACHE="$REPOSITORY_ROOT/data/processed/r2plus1d_yolo_32_128"
    INITIALIZATION="scratch"
    EXTRA=(
      --init-visual "$REPOSITORY_ROOT/outputs/checkpoints/r2_depth_ir_32f_f${FOLD}/best.pt"
      --mixup-alpha 0.3
      --use-temporal-embedding
    )
    ;;
  noflip_mixup)
    MODE="multimodal"
    CACHE="$REPOSITORY_ROOT/data/processed/r2plus1d_yolo_32_128"
    INITIALIZATION="scratch"
    EXTRA=(
      --init-visual "$REPOSITORY_ROOT/outputs/checkpoints/r2_depth_ir_32f_f${FOLD}/best.pt"
      --mixup-alpha 0.3
      --horizontal-flip-probability 0
    )
    ;;
  warm_mixup)
    MODE="multimodal"
    CACHE="$REPOSITORY_ROOT/data/processed/r2plus1d_yolo_32_128"
    INITIALIZATION="scratch"
    EXTRA=(
      --init-model "$REPOSITORY_ROOT/outputs/checkpoints/r2_depth_ir_32f_staged_f${FOLD}/best.pt"
      --mixup-alpha 0.3
      --lr 0.0001
      --warmup-epochs 1
    )
    ;;
  strong_aux_mixup)
    MODE="multimodal"
    CACHE="$REPOSITORY_ROOT/data/processed/r2plus1d_yolo_32_128"
    INITIALIZATION="scratch"
    EXTRA=(
      --init-visual "$REPOSITORY_ROOT/outputs/checkpoints/r2_depth_ir_32f_f${FOLD}/best.pt"
      --mixup-alpha 0.3
      --auxiliary-weight 0.6
    )
    ;;
  moddrop_mixup)
    MODE="multimodal"
    CACHE="$REPOSITORY_ROOT/data/processed/r2plus1d_yolo_32_128"
    INITIALIZATION="scratch"
    EXTRA=(
      --init-visual "$REPOSITORY_ROOT/outputs/checkpoints/r2_depth_ir_32f_f${FOLD}/best.pt"
      --mixup-alpha 0.3
      --structured-modality-dropout 0.15
      --allow-method-combination
    )
    ;;
  gated_moddrop_mixup)
    MODE="multimodal"
    CACHE="$REPOSITORY_ROOT/data/processed/r2plus1d_yolo_32_128"
    INITIALIZATION="scratch"
    EXTRA=(
      --init-visual "$REPOSITORY_ROOT/outputs/checkpoints/r2_depth_ir_32f_f${FOLD}/best.pt"
      --mixup-alpha 0.3
      --structured-modality-dropout 0.15
      --allow-method-combination
      --fusion-output gated
    )
    ;;
  joint7_mixup)
    MODE="multimodal"
    CACHE="$REPOSITORY_ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"
    INITIALIZATION="scratch"
    EXTRA=(
      --init-visual "$REPOSITORY_ROOT/outputs/checkpoints/r2_depth_ir_32f_f${FOLD}/best.pt"
      --mixup-alpha 0.3
    )
    ;;
  *) echo "Unknown method: $METHOD" >&2; exit 2 ;;
esac

set +e
flock "$REPOSITORY_ROOT/outputs/logs/gpu-${GPU}.lock" \
  env CUDA_VISIBLE_DEVICES="$GPU" "$CONDA" run --no-capture-output -n Train \
  python "$REPOSITORY_ROOT/scripts/train_r2plus1d.py" \
  --name "$NAME" \
  --fold "$FOLD" \
  --mode "$MODE" \
  --initialization "$INITIALIZATION" \
  --weights "$REPOSITORY_ROOT/.tools/checkpoints/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth" \
  --cache-dir "$CACHE" \
  --frames 32 \
  --image-size 128 \
  --epochs 20 \
  --batch-size 8 \
  --workers 4 \
  --lr 0.0003 \
  --backbone-lr-factor 0.1 \
  --weight-decay 0.0001 \
  --warmup-epochs 2 \
  --ema-decay 0.99 \
  --dropout 0.3 \
  --auxiliary-weight 0.15 \
  --label-smoothing 0.03 \
  --class-balance-power 0 \
  --seed "$SEED" \
  --patience 20 \
  --eval-every 1 \
  --save-epoch-checkpoints \
  --output-root "$EXPERIMENT_ROOT/outputs" \
  --swanlab-mode cloud \
  "${EXTRA[@]}" 2>&1 | tee "$LOG_DIR/${NAME}.log"
status=${PIPESTATUS[0]}
printf '%s\n' "$status" > "$LOG_DIR/${NAME}.exit"
exit "$status"
