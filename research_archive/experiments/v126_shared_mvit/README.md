# V126 Shared MViT-V2 Multimodal Encoder

V126 replaces the shared R(2+1)D visual CNN with a shared MViT-V2-S video
Transformer initialized on Kinetics-400. Depth, Thermal, and IR are encoded in
one shared visual batch, then fused with Skeleton and IMU tokens by a two-layer
Transformer. The full model has 36.2M parameters and occupies 72.4MB in FP16,
leaving room for preprocessing assets under the 100MB limit.

The primary run uses all five informative modalities. A simultaneous visual-only
ablation measures whether structured sensors add cross-person generalization.

