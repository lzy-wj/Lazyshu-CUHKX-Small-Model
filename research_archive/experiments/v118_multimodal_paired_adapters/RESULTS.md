# Results

- V114 visual-only: 1091 raw and 1180 pair-aware of 1357.
- V118 visual + Skeleton + IMU: exactly 1091 raw and 1180 pair-aware.
- Fold0 selects the same 0.6 adapter weight; target-weighted scores are
  identical on both folds.
- Decision: reject. The frozen structured encoders are already stable enough;
  the useful person-specific correction is concentrated in visual appearance.
