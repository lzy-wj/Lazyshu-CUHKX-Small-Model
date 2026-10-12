# V47 Environment-Specific AdaBN

## Question

Does estimating visual BatchNorm statistics separately for the two acquisition
environments improve the V32 hard-pseudo system?

## Protocol

- The V24 heads-only pseudo adaptation is reproduced unchanged.
- For each environment, BatchNorm statistics are reset and estimated from the
  fold's labeled-source inputs plus unlabeled target inputs in that environment.
- No labels are consumed by AdaBN.
- Spatial TTA, V32 domain weights, and the pair-aware decoder remain frozen.
- Fold 0 is screened first. Fold 1 is run once only if Fold 0 gains at least two
  decoded rows without losing raw accuracy.

The method adds no serialized parameters and remains within the V24 asset budget.

