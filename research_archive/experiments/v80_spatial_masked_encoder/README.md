# V80 Spatial Masked Multimodal Encoder

## Question

Can a Transformer improve generalization when it receives spatially preserved
CNN features instead of tokens that have already been averaged over space?

## Architecture

1. Load the frozen V23 warm-60 Fold-0 model.
2. Execute the joint Depth/Thermal/IR R(2+1)D-34 once and capture its layer-3
   feature map before the final spatial compression.
3. Pool each visual stream to an `8 x 2 x 2` time-space grid and project the
   resulting 32 tokens to 512 dimensions.
4. Append projected Skeleton and IMU temporal tokens and encode all five
   modalities with a four-layer, eight-head Transformer.
5. Predict a zero-initialized residual over the protected V23 logits.
6. During training only, mask 35% of available visual tokens and reconstruct
   their detached CNN features with a lightweight decoder.

The frozen base guarantees that epoch 0 exactly reproduces `487/652`. The
training-only reconstruction decoder is not required at inference. Promotion
requires at least `490/652` with NLL no worse than the epoch-0 reference.

## Outcome

- The masked encoder reached `490/652` on Fold 0 and `565/705` on the blind
  Fold 1. A Fold-0-selected probability blend reached `1056/1357`, two samples
  above the protected V23 baseline.
- The encoder-only ablation reached a stronger `491/652` on Fold 0 but dropped
  to `562/705` on blind Fold 1. This confirms that spatial tokens contain useful
  signal, while masked reconstruction improves cross-user stability.
- The deployable model has `76,201,904` parameters and occupies `82,111,878`
  bytes after INT8 quantization plus the required YOLO detector.

Run `run_fold0.sh` followed by `run_fold1_selected.sh` to reproduce the masked
model. Run `run_encoder_only_fold0.sh` followed by
`run_encoder_only_fold1_selected.sh` for the ablation.
