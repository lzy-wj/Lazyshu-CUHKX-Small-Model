# V85 Results

## Decision

Reject. Training-fold body-scale randomization improves Fold 0 but regresses
the test-weighted Fold 1 and the sequence-level result.

| Variant | Raw correct | Mean test-domain weighted raw | Pair-aware correct | Mean test-domain weighted pair |
| --- | ---: | ---: | ---: | ---: |
| V23 control | 1044 / 1357 | 76.56% | 1154 / 1357 | 86.81% |
| V85 scale randomization | 1039 / 1357 | 76.16% | 1138 / 1357 | 85.35% |
| 50/50 probability ensemble | 1037 / 1357 | 76.04% | 1142 / 1357 | 85.51% |

Fold 0 raw accuracy rises from `479` to `481`, but Fold 1 falls from `565` to
`558`. The first two-epoch pilot is excluded because its scale draw advanced
the global Python RNG and changed later video augmentations. The clean run uses
an isolated worker RNG and is the only result in `outputs/evaluation.json`.
