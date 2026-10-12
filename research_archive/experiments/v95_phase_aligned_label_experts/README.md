# V95 Phase-Aligned Label Experts

V95 is a large-teacher experiment combining two orthogonal inductive biases:

1. Depth, Thermal, IR, Skeleton, and IMU features are resampled into 16 shared
   action phases and fused within each phase.
2. Four independent temporal Transformer experts specialize in fixed semantic
   label groups while retaining an `Other` objective.

The V23 multimodal encoder is frozen. Expert residual heads are zero initialized,
so the initial network is exactly function preserving. The semantic partition
is fixed from action names before validation and no blend weights are searched.
