# V97 Results

| Variant | Raw | Pair-aware | Target-domain weighted pair |
|---|---:|---:|---:|
| V23 high-resolution source | 1057/1357 | 1160/1357 | 87.32% |
| V89 local best | **1077/1357** | 1170/1357 | 88.81% |
| Intermediate protected adapter | 1057/1357 | **1173/1357** | 87.91% |
| Dual protected adapter | 1057/1357 | 1169/1357 | 88.18% |
| Dual adapter / V89 equal average | 1074/1357 | 1170/1357 | **89.08%** |

Freezing the source network and limiting learning to 0.21M-0.62M
zero-initialized CNN adapter parameters eliminates V96's broad regression. The
intermediate model gains three total pair-aware rows over V89, but error
attribution shows that gain is concentrated in domain 0, which is absent from
the measured test composition. The dual/V89 average raises target-weighted
accuracy on both folds (`88.10% -> 88.39%`, `89.51% -> 89.78%`) but ties the
total pair count. It is retained as evidence, not yet promoted as a new best.
