# V102 Results

| Frozen V23 scale | Raw | Pair-aware | Target-domain weighted pair |
|---|---:|---:|---:|
| 160px | **1057/1357** | 1160/1357 | 87.32% |
| 192px | 1043/1357 | **1162/1357** | **87.38%** |
| 160/192 equal | 1048/1357 | 1158/1357 | 87.48% |

Native 192 pixels gains two total pair-decoded rows but loses target-weighted
accuracy on Fold0 (`86.49% -> 86.31%`) while improving Fold1. It fails the
predeclared both-fold gate. Together with the clear 224-pixel regression, this
closes the resolution branch at 160 pixels; no 192/224 pseudo adaptation is
run.
