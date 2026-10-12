# V99 Results

| Variant | Raw | Pair-aware | Target-domain weighted pair |
|---|---:|---:|---:|
| V89 local best | 1077/1357 | **1170/1357** | **88.81%** |
| V98 old-teacher cross | **1090/1357** | 1154/1357 | 87.77% |
| Refreshed main | 1087/1357 | 1168/1357 | 88.56% |
| Refreshed cross | 1083/1357 | 1168/1357 | 88.58% |
| Refreshed main/cross equal | 1088/1357 | 1169/1357 | 88.71% |

Refreshing hard labels from the current V89 sequence teacher recovers 14-15
pair-aware rows relative to V98, confirming that stale pseudo labels caused
most of the sequence mismatch. It simultaneously gives back part of V98's raw
gain and still stops one row below V89. Fixed equal ensembles do not cross the
promotion gate, so no V99 model is submitted.
