# V85 Skeleton Scale Randomization

V85 targets the measured cross-subject body-scale shift without choosing an
arbitrary jitter magnitude. For every training clip, it samples the torso
scale of another clip from the same training fold and transfers that scale to
the 3D body around its root trajectory. Bone and velocity streams are then
recomputed exactly.

This breaks the shortcut between body size, subject identity, and action
frequency while preserving action and root motion. It uses no test statistic
and changes only training-time Skeleton augmentation on the V23 warm recipe.

Promotion requires a pair-aware correct-count gain and a test-domain-weighted
gain in both independent folds.
