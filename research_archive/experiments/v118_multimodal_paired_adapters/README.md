# V118 Multimodal Paired Adapters

V118 extends the successful V113 person-specific visual adaptation to all
informative streams. Zero-initialized temporal residual adapters are added to
the frozen Skeleton and IMU token encoders, alongside the existing three visual
adapters. This tests whether person-specific kinematics transfer across rooms
better than appearance alone. All source/target supervision and optimization
settings remain identical to V113.
