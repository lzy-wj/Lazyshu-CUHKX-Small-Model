# V49 Last-Fusion-Layer Pseudo Adaptation

## Question

Can V24's pair-aware pseudo labels adapt the final Transformer fusion layer,
instead of only its classification and normalization heads, without losing
cross-subject generalization?

## Frozen Protocol

- Same V23 fold checkpoint and V13 pair-aware pseudo labels as V24.
- Same two epochs, confidence threshold `0.6`, retention weight `0.25`, BF16,
  augmentation, AdaBN, and four-view spatial TTA.
- Only the scope changes from `heads` to `fusion_last`; learning rate is reduced
  from `2e-4` to the previously audited `1e-4`.
- V32 domain weights and pair-aware decoder remain frozen.
- Fold 1 is run once only if Fold 0 gains at least two decoded rows without a
  raw regression.

