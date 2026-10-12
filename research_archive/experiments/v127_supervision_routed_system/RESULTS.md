# Results

| System | Fold 0 | Fold 1 | Combined | Target-weighted mean |
|---|---:|---:|---:|---:|
| V71 | **570** | 631 | 1201/1357 | 89.81% |
| V114 | 552 | 628 | 1180/1357 | 89.80% |
| V127 | 569 | **636** | **1205/1357** | **90.36%** |

The fixed supervision route gains four rows over the historical local-count
leader V71 and 25 over V114. Mean target-domain-weighted accuracy gains 0.55
points over both components. Fold 0 gives back one V71 row while Fold 1 gains
five, so deployment remains protected against V24 public regressions.

The union of all required V71 and V114 deployment assets is 83,900,083 bytes,
leaving 16,099,917 bytes below the competition limit.
