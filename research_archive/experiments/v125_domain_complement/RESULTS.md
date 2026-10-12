# Results

| System | Fold 0 | Fold 1 | Combined | Target-weighted mean |
|---|---:|---:|---:|---:|
| V14 | 559 | 627 | 1186/1357 | 88.83% |
| V114 | 552 | 628 | 1180/1357 | 89.80% |
| V125 | **563** | **633** | **1196/1357** | **90.08%** |

The supervision-availability route improves total accuracy on both folds and
adds 16 correct rows over V114 without fitting a confidence threshold or blend
weight. It is the new local sequence-system baseline. Fold 0 target-weighted
accuracy is 0.13 points below V114, while Fold 1 gains 0.68 points; the mean
target-weighted result improves by 0.27 points.

