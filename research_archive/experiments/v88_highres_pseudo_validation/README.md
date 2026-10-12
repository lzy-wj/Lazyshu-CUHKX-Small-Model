# V88 High-Resolution Pseudo-Target Validation

This experiment tests one structural change against V24: increase the visual
input from 128 to 160 pixels during the same two-epoch target-fold pseudo-label
adaptation and evaluation. The checkpoint, pseudo labels, trainable scope,
optimizer, augmentation, TTA, and pair decoder remain unchanged.

The experiment is promoted only if direct pair-aware accuracy beats the V24
`1168/1357` reference and test-domain-weighted pair accuracy improves on both
held-out folds.
