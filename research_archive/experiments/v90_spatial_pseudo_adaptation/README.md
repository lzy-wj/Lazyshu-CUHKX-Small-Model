# V90 Spatial-Token Pseudo Adaptation

V90 applies the frozen V24 two-epoch pseudo-target protocol to the 160-pixel V86
spatial-token encoder. Only the V86 residual classification head is trainable;
the R(2+1)D backbone, temporal sensor encoders, spatial Transformer, pseudo
labels, augmentations, TTA, and pair decoder are fixed.

The direct V90 prediction tests whether spatially preserved high-resolution
features adapt better than globally pooled features. A single predeclared equal
average with V24 tests architecture-view marginalization without searching a
blend coefficient.
