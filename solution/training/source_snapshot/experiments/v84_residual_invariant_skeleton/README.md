# V84 Residual-Invariant Skeleton Encoder

V84 tests one representation change on the V23 warm masked-distillation
recipe. The existing raw-coordinate Skeleton TCN remains intact. A second TCN
encodes a body-centered, torso-scale-normalized pose/bone/velocity view, and a
linear residual projection combines both views.

The projection is initialized to `[I, 0]`, so the extended model is exactly
the source model before optimization. This differs from the rejected scratch
DSTformer: it adds a small invariant residual without discarding absolute
motion or replacing the trained encoder.

The input audit found a real cross-subject scale shift: test median torso
length is `0.46` train standard deviations smaller, and median full-body span
is `0.52` standard deviations smaller. The exact report is
`outputs/skeleton_shift_audit.json`.

Fold 0 and Fold 1 are independent. Promotion requires a pair-aware correct
count gain and a test-domain-weighted gain in both folds.
