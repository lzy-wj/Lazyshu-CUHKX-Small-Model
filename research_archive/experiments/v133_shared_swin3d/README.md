# V133 Shared Swin3D Multimodal Encoder

V133 is the local-window counterpart to V126. One Kinetics-pretrained Swin3D-T
encoder is shared across Depth, Thermal, and IR, then its modality tokens are
fused with Skeleton and IMU by the same two-layer Transformer and expert head.
This isolates the video-transformer inductive bias while preserving the
successful full multimodal design.

