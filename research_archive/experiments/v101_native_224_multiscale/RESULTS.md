# V101 Results

| Frozen V23 scale | Raw | Pair-aware | Target-domain weighted pair |
|---|---:|---:|---:|
| 128px | 1054/1357 | 1153/1357 | 86.34% |
| 160px | **1057/1357** | **1160/1357** | **87.32%** |
| 224px | 982/1357 | 1120/1357 | 85.41% |
| 128/160/224 equal | 1058/1357 | 1164/1357 | 87.66% |

Native 224-pixel inference is outside the frozen model's learned scale regime
and regresses both folds substantially. It fails the preregistered gate, so no
224-pixel pseudo adaptation is run. V102 performs the unique 160/224 midpoint
test before closing the scale branch.
