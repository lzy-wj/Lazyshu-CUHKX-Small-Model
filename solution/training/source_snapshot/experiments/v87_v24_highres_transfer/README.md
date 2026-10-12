# V87 V24 High-Resolution Transfer

V87 transfers the validated `128 -> 160` inference change to the full-data V24
checkpoint without changing weights, TTA, decoder, or model assets. It first
generates local-only test predictions and measures disagreement with the public
V24 submission. No Kaggle submission is allowed from this directory until the
V85 scale experiment and V86 consistency audit finish.

The local justification is `experiments/v86_highres_spatial_encoder/outputs/v23_pure_highres_evaluation.json`:
the same V23 parameter soup improves from `1054` to `1057` raw and from `1153`
to `1160` after the frozen pair-aware decoder, with pair-weighted gains in both
folds.
