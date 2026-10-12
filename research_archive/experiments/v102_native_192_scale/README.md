# V102 Native 192-Pixel Scale

V102 is the final scale ablation. Native 160 pixels improves both target folds,
while native 224 pixels clearly regresses. The unique midpoint, 192 pixels, is
tested with original-frame rebuilding and frozen V23 weights. If it does not
beat 160 pixels on both target-weighted folds, the resolution branch closes.
