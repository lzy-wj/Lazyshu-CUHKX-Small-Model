# V50 Per-User Pseudo Adaptation

## Question

Does adapting an independent small classification head for each unseen target
user avoid gradient conflict between subjects and environments?

## Protocol

- Reuse V24's two-epoch, heads-only, pair-aware pseudo-label recipe unchanged.
- Train one adapter independently for every validation user from the same frozen
  fold checkpoint.
- Reset AdaBN and infer only that user's clips before merging the fold.
- Keep V32 domain weights and the pair-aware decoder frozen.
- Fold 0 must gain at least two decoded rows without raw regression before one
  blind Fold-1 run.

Four deployment deltas remain below 1 MB in total and do not alter the base
CNN/RNN/Transformer architecture.

