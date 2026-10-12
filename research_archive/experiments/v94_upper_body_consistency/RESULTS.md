# V94 Results

| Model | Raw | Pair-aware | Target-domain weighted pair |
|---|---:|---:|---:|
| V89 scale average | 1077/1357 | **1170/1357** | **88.81%** |
| V94 upper consistency 160px | 1065/1357 | 1157/1357 | 87.82% |
| V24 + V94 scale average | **1079/1357** | 1164/1357 | 88.49% |

Explicit full/upper consistency slightly improves the blended raw classifier,
but degrades sequence coherence and both-fold target-domain evidence relative to
V89. V94 and the upper-body crop direction are rejected. No full-data refit or
test prediction is produced.
