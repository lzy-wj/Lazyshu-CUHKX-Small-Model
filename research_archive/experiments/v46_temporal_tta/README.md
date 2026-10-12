# V46 Synchronized Temporal TTA

## Question

Can synchronized temporal crops improve the V32 hard-pseudo system without
changing training, model parameters, spatial preprocessing, or the decoder?

The adapted model sees the same temporal interval for video, skeleton, and IMU.
Each temporal view retains the frozen four-view spatial TTA: original, horizontal
flip, center crop, and center-crop flip.

## Protocol

1. Reproduce the V24 two-epoch heads-only pseudo adaptation independently for
   Fold 0 and Fold 1.
2. Save predictions for the full clip, early/late 87.5% windows, centered 87.5%
   window, and early/late 75% windows.
3. Select the temporal-view family and its blend weight on Fold 0 only.
4. Apply the selected setting once to blind Fold 1.
5. Keep the V32 domain gate and pair-aware decoder frozen.

This changes inference compute only. The serialized model remains the existing
V24 INT8 checkpoint plus its small head delta.

