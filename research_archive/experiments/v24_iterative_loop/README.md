# V24 Shared Iterative Fusion

## Frozen question

Can repeated use of one shared Transformer fusion block preserve V23's stronger raw clip probabilities in the sequence system without rerunning any CNN/RNN encoder or increasing parameters with `K`?

The controlled comparison uses the same V23 fold-specific warm-60 soup initialization, split, seed, augmentation, optimizer, EMA, auxiliary loss, and ten-epoch schedule. The only experimental variable is `K in {1,2,3}`.

## Architecture

1. The shared R(2+1)D-34 visual CNN runs once over a joint Depth/Thermal/IR batch.
2. Skeleton and IMU CNN/LSTM encoders each run once.
3. Their projected tokens and padding mask are cached. V23 contains 12 visual, 16 skeleton, and 16 IMU tokens.
4. At step 1, the original learned class token and cached modality tokens pass through the original two-layer Transformer fusion block. This is exactly the non-iterative V23 forward.
5. At later steps, the previous class state receives a projected soft class distribution, then the **same** Transformer object consumes that state and the unchanged cached tokens again.
6. Soft feedback is detached by default, as required by the frozen protocol. Every step receives deep supervision with normalized weights proportional to `0.6^(K-step-1)`.
7. Optional batch-level early exit is disabled in training and in the confirmatory ablation. It can later stop when every sample is confident or stable.

The encoder is never called inside the iteration loop. `K` changes compute, not state-dict size. `K=1` does not consume the feedback modules and is numerically identical to the existing residual-fusion forward in evaluation mode.

## Selection protocol

Stage 1 trains `K=1/2/3` only on Fold 0. `select_fold0_k.py` ranks completed runs by:

1. correct clips, descending;
2. NLL, ascending;
3. missing-visual correct clips, descending.

Stage 2 reads the resulting `selection.json` and permits exactly that K on Fold 1. `run_one.sh` rejects any other Fold-1 K. Fold 1 uses the fixed tenth-epoch EMA and is evaluated only once after training; it never selects an epoch or triggers validation-based early stopping. Promotion additionally requires at least `+3` Fold-0 clips over K=1 with no worse NLL, followed by a positive combined Fold0+Fold1 gain.

## Night launch

Inspect GPUs first, then launch three genuinely idle cards before 08:00 Beijing time:

```bash
nvidia-smi
tmux new-session -d -s v24-loop-f0 \
  'bash experiments/v24_iterative_loop/run_stage1_fold0.sh 0 3 4'
```

After Stage 1 completes and `selection.json` is reviewed, launch the single blind Fold-1 run:

```bash
nvidia-smi
tmux new-session -d -s v24-loop-f1 \
  'bash experiments/v24_iterative_loop/run_stage2_fold1_selected.sh 0'
```

During daytime, the three-GPU stage refuses to launch. Individual Fold-0 controls may be run sequentially with `run_one.sh GPU 0 K` while preserving four free GPUs. Every launch writes command, GPU, start time, launcher PID and logs under `runs/`.

## Assets and latency

`estimate_assets.py` computes exact parameter and quantized tensor counts from the V23 checkpoint. The iterative feedback path adds 8,257 parameters: 7,680 projection weights, 192 projection biases, 384 LayerNorm parameters, and one scalar gate. The count is identical for K=1/2/3.

The cached V23 modality tokens require 16,896 BF16 bytes per sample. Exact INT8 tensor storage is 65,254,089 bytes, leaving 29.75 MB against the 95 MB design target before non-model packaged assets. Each additional two-layer fusion iteration is approximately 34.73 MFLOPs per sample, small relative to the R(2+1)D CNN. Exact GPU latency and peak memory must be measured after training:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/v24_iterative_loop/profile_latency.py \
  --checkpoint experiments/v24_iterative_loop/outputs/v24_iterative_k2_f0/best.pt \
  --fold 0 --output experiments/v24_iterative_loop/latency_k2_f0.json
```

## Main-agent integration

- Promote only the Fold-0-selected K after its one blind Fold-1 run passes the frozen gates.
- Use `IterativeR2Plus1D.encode_once` and the recurrent class-state loop as the exact source patch; do not wrap repeated calls around the existing full model because that would rerun the CNN/RNN encoders.
- Keep detached feedback and deep-supervision decay 0.6 frozen unless a new Fold-0-only ablation is declared.
- Freeze K, epoch schedule, decoder and INT8 recipe before full-data refitting.
- Package only one INT8 iterative checkpoint plus existing allowed assets; no Kaggle submission is performed here.
