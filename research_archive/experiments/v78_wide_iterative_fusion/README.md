# V78 Wide Shared Iterative Fusion

## Question

Does additional multimodal fusion capacity become useful when the same fusion
network is reused for iterative refinement?

The experiment keeps the V23 warm-60 checkpoint, data split, augmentation,
optimizer, EMA, and ten-epoch schedule fixed. Fold 0 compares only `K=1` and
`K=2` with the same 768-dimensional, four-layer Transformer refinement module.

## Architecture

1. The pretrained joint Depth/Thermal/IR R(2+1)D-34 and the Skeleton/IMU
   encoders execute once.
2. The original 192-dimensional two-layer fusion produces the protected base
   prediction.
3. Cached modality tokens and the fused class state are projected to 768
   dimensions.
4. One four-layer Transformer refinement module is shared for `K` passes.
5. Each pass predicts a zero-initialized residual over the protected base
   logits. Later passes receive detached soft-label feedback.
6. Every pass receives deep supervision with decay `0.6`.

The model has about 89 million parameters independent of `K`. INT8 model state
plus the compact YOLO asset must remain below the decimal 100 MB limit.

## Protocol

- Fold 0 runs `K=1` and `K=2` in parallel on physical GPUs 6 and 7 only after
  both GPUs are confirmed idle.
- Larger capacity passes its gate at `>=490/652`, three rows above the frozen
  V23 epoch-0 Fold-0 reference of `487/652`, with NLL no worse than `1.17912`.
- Looping passes its gate only when `K=2` strictly exceeds `K=1` in correct
  rows without worse NLL.
- Fold 1 remains blind until Fold-0 selection is complete.
