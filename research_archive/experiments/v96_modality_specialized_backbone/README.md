# V96 Modality-Specialized Visual Backbone

V96 tests whether trainable encoder capacity, rather than frozen expert-head
capacity, improves cross-subject generalization. Depth, Thermal, and IR share
the stem and low residual stages but receive independently trainable high-level
R(2+1)D residual stages copied from the V23 backbone.

Two preregistered capacities are compared: independent layer4 (`split4`) and
independent layer3+layer4 (`split3`). Both are exactly function preserving at
initialization. Skeleton, IMU, fusion, and classifier parameters stay frozen;
per-visual auxiliary supervision forces each visual branch to remain useful.

Training uses the fixed V23 recipe at 160 pixels for six epochs, BF16 autocast,
MixUp, source-logit retention, and EMA. Both folds use the final epoch. The
evaluation script declares all direct and equal-probability ensemble variants
before results are available; it never searches blend weights.
