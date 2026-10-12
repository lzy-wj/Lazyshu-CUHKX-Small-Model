# V83 Synchronized Multimodal Flip

The legacy augmentation flips Depth, Thermal, and IR while leaving Skeleton
and IMU unchanged. V83 keeps the V23 warm masked-distillation recipe fixed but
mirrors every modality consistently:

- swap left/right skeleton joints and negate the horizontal coordinate;
- swap left/right arm and leg IMU devices;
- reflect acceleration as a polar vector and gyroscope as an axial vector.

Fold 0 and Fold 1 are trained independently. Promotion requires improvement on
the test-domain-weighted metric in both folds before any full-data refit.
