# V105 PoseC3D-Style Heatmaps

V105 tests a representation change rather than more fusion capacity. The raw
17-joint 3D skeleton is body-centered, scale-normalized, and yaw-canonicalized,
then rendered into `XZ`, `YZ`, and `XY` Gaussian heatmaps for five fixed body
parts. A compact depthwise 3D CNN learns trajectories from these heatmaps.

The branch is trained independently on both user-disjoint folds. Its only
promotion path is a Fold-0-selected, Fold-1-blind probability blend with V89,
followed by the frozen pair decoder. No test prediction or Kaggle submission is
created unless the two-fold gate passes.
