# V158 Group-DRO + Balanced Softmax

V150 improves every Fold-0 user and gains six raw rows by minimizing worst-user
risk. V152 independently gains five raw rows through canonical long-tail logit
adjustment. V158 combines these orthogonal corrections: Group-DRO aggregates
per-user Balanced-Softmax losses, while architecture, augmentation, and V23
warm initialization remain fixed.

The exact confirmed settings are reused without a new sweep. Fold 1 remains
gated by final V127-aware Fold-0 performance.
