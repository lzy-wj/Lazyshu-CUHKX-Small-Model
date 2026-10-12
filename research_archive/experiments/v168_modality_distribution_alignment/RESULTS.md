# Results

The Fold-0 signal does not transfer to the blind fold under the same route and
weight:

| Variant | Fold 0 delta vs V166 | Fold 1 delta vs V166 |
|---|---:|---:|
| minimum 5 modalities | +6 | -6 |
| full 6 modalities | +9 | -6 |

For full6, the fixed 0.25 blend makes 11 gains and 2 losses on Fold 0, but only
3 gains and 9 losses on Fold 1. Restricting to unmatched users avoids most
changes but gives `0/-1` rather than an improvement. Decision: reject; the
Fold-0 gain is paired-user-specific and not a stable generalization result.
