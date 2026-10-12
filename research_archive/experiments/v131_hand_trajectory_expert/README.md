# V131 Hand-Trajectory Expert

V131 targets the remaining fine-grained hand-object confusions without relying
on an unreliable 3D-to-image projection. It extracts body-normalized 3D
trajectories for the spine, head, shoulders, elbows, and wrists directly from
Skeleton. A temporal CNN captures short motion, a bidirectional GRU captures
action progression, and a one-layer Transformer plus attention pooling models
long-range hand interactions. The branch is trained independently on both
subject-disjoint folds before any fusion is considered.

