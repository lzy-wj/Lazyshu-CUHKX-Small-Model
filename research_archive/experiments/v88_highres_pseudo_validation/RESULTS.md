# V88 Results

| Model | Raw | Pair-aware | Target-domain weighted pair |
|---|---:|---:|---:|
| V24 pseudo-adapted 128px | 1070/1357 | **1168/1357** | 88.22% |
| V88 pseudo-adapted 160px | **1082/1357** | 1167/1357 | **88.81%** |

V88 gains 12 raw classifications and improves target-domain-weighted pair
accuracy on both folds (`87.63% -> 88.10%` and `88.81% -> 89.51%`). Its total
pair-aware count is one row lower because Fold 0 loses rows in domain 0, which
has zero weight under the measured 405-row test-domain composition.

The preregistered direct promotion gate is not met. V88 is rejected as a
standalone replacement but retained as independently positive scale evidence
for the fixed multi-scale test in V89.
