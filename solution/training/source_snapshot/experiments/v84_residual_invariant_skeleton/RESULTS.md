# V84 Results

## Decision

Reject. The measured train/test body-scale shift is real, but adding a second
deterministically normalized Skeleton view does not improve generalization.

| Variant | Raw correct | Mean test-domain weighted raw | Pair-aware correct | Mean test-domain weighted pair |
| --- | ---: | ---: | ---: | ---: |
| V23 control | 1044 / 1357 | 76.56% | 1154 / 1357 | 86.81% |
| V84 residual-invariant | 1028 / 1357 | 75.55% | 1126 / 1357 | 83.72% |
| 50/50 probability ensemble | 1033 / 1357 | 75.96% | 1150 / 1357 | 86.61% |

The extension is only `376,512` parameters and starts exactly at the source
model through an `[I, 0]` projection, but it regresses both test-weighted folds.
The full frozen report is `outputs/evaluation.json`; the input-distribution
evidence remains in `outputs/skeleton_shift_audit.json` and motivates V85's
training-time de-correlation rather than another deterministic branch.
