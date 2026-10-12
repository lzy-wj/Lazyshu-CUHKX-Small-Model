# Results

## Standalone

| Fold | Raw | Paired decoder |
| --- | ---: | ---: |
| 0 | 405/652 (62.12%) | 481/652 (73.77%) |
| 1 | 486/705 (68.94%) | 556/705 (78.87%) |
| Combined | 891/1,357 (65.66%) | 1,037/1,357 (76.42%) |

## Fusion selection

Routes `unmatched`, `paired`, and `both` were tested at weights 0.02, 0.05,
and 0.10. Selection used Fold 0 only, then transferred once to Fold 1. All
nonzero settings reduced Fold-0 accuracy, so the selected model is the V127
control with zero Swin3D weight.

Swin3D is rejected for the current data scale. Its local-window video prior
does not recover enough cross-person action invariance to complement V127.
