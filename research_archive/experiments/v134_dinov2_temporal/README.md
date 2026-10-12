# V134 DINOv2 Temporal Multimodal Model

V134 tests a genuinely different visual prior for the remaining fine-grained
hand-object errors. A single official DINOv2 ViT-S/14 encoder is shared across
RGB, Thermal, and IR frames. Frame-level class and patch-mean features are
encoded by a temporal Transformer, then fused with Skeleton and IMU tokens by
a second Transformer.

The first-stage experiment freezes DINOv2 and caches eight uniformly sampled
frames per visual stream. Two training objectives are compared on both subject
held-out folds: classification only, and classification plus cross-user
supervised contrastive learning. No test labels are used. A full deployable
model would store one FP16 22.06M-parameter DINOv2 encoder plus the selected
temporal head, comfortably below 100MB.
