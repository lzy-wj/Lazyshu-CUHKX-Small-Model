# V92 Results

| Model | Raw | Pair-aware | Target-domain weighted pair |
|---|---:|---:|---:|
| V89 scale average | 1077/1357 | **1170/1357** | **88.81%** |
| V92 consensus 160px | **1081/1357** | 1165/1357 | 88.22% |
| V24 + V92 equal average | **1081/1357** | 1167/1357 | 88.41% |

Cross-scale agreement substantially improves the measured accuracy of retained
pseudo targets, but this does not improve the final sequence-decoded metric.
Removing disagreement rows appears to discard useful adaptation coverage. V92
is rejected; no full-data refit or test candidate is produced.
