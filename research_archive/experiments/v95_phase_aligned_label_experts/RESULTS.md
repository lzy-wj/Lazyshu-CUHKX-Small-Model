# V95 Results

| Model | Parameters | Raw | Pair-aware | Target-domain weighted pair |
|---|---:|---:|---:|---:|
| V86 spatial 160px | 76.2M | 1052/1357 | **1165/1357** | **87.79%** |
| V95 phase experts, d384 | 79.1M | **1054/1357** | 1162/1357 | 87.52% |
| V95 phase experts, d768 | 121.0M | 1053/1357 | 1162/1357 | 87.52% |

Both semantic-expert teachers fail the preregistered promotion rule. Doubling
the expert width adds 41.9M parameters without changing pair-aware or weighted
accuracy, directly showing that capacity is not the active bottleneck in this
frozen-encoder design.

The experts are not distilled because a weaker teacher would only transfer its
errors. V89 remains the strongest credible candidate at `1170/1357` and 88.81%
target-domain-weighted pair accuracy.
