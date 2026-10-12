# V90 Results

| Model | Raw | Pair-aware | Target-domain weighted pair |
|---|---:|---:|---:|
| V89 scale average | **1077/1357** | **1170/1357** | **88.81%** |
| V90 spatial 160px | 1060/1357 | 1168/1357 | 88.33% |
| V24 + V90 equal average | 1071/1357 | 1167/1357 | 88.54% |

Pseudo adapting the spatial residual head recovers the V24 pair count but loses
target-domain-weighted accuracy relative to V89. Its fixed architecture average
also regresses. The spatial branch is rejected and will not be refitted or
submitted.
