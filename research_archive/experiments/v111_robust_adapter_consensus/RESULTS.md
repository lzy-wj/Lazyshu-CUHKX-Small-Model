# Results

Fold 0 target-domain-weighted pair accuracy selects the original arithmetic
mean over geometric pooling, coordinate-wise median pooling, and strict
majority acceptance. Consequently the frozen Fold-1 evaluation and combined
result are exactly V104: 1,175/1,357 and 89.2466% target-weighted pair accuracy.

This negative result is still informative: V104's gain comes from calibrated
soft diversity across all four domain-1 components, not from a removable
adapter outlier. The arithmetic fusion remains fixed.

The complete comparison is in `outputs/evaluation.json`.
