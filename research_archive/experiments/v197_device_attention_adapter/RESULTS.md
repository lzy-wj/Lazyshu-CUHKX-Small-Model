# Results

The device-aware residual is stopped after seven epochs because both seeds show
the same Fold0-to-Fold1 reversal. Under strict Fold-0 epoch selection:

| Seed | Selected epoch | Fold 0 | Fold 1 | Combined vs V23 |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 4 | 481/652 | 560/705 | -3 |
| 43 | 6 | 480/652 | 560/705 | -4 |

Separate device tokens increase Fold-0 flexibility but do not generalize to the
second held-out user set. The architecture is rejected and no test candidate is
generated.
