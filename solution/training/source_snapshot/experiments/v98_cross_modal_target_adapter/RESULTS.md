# V98 Results

| Variant | Raw | Pair-aware | Target-domain weighted pair |
|---|---:|---:|---:|
| V89 local best | 1077/1357 | **1170/1357** | **88.81%** |
| Main adapter | 1084/1357 | 1155/1357 | 87.90% |
| Cross-modal adapter | 1090/1357 | 1154/1357 | 87.77% |
| Cross-modal / V89 equal average | **1092/1357** | 1168/1357 | 88.60% |

Cross-modal fused-to-stream distillation is a genuine clip-classification gain:
it improves raw accuracy on both folds and raises the fixed V89 ensemble by 15
rows. However, the frozen sequence decoder does not preserve those gains. A
Fold0-only minimum-margin calibration, fixed geometric ensembles, adapter
consensus, and a confidence rescue all fail blind Fold1 or remain below V89.

The adapters are not promoted directly. V99 changes only the stale hard
pseudo-label teacher: it regenerates labels from the current V89 sequence
system, whose pair-aware OOF is nine rows stronger than the old V13 teacher.
