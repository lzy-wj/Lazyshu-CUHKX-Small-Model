# V34 Results

Every threshold from `0.4` through `2.0` ties at `102/141` on the Fold 0
target domain when the low-confidence adapted weight is `0.25` or `0.5`.
The simplicity tie-break selects threshold `2.0` and weight `0.5`, exactly
recovering V32.

Blind Fold 1 remains `623/705`; combined accuracy remains `1174/1357`.
Confidence-dependent gating adds no value and is rejected in favor of V32.
