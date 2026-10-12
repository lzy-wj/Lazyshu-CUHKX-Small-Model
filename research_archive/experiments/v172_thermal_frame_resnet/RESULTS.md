# Results

Strict cross-subject clip accuracy for the official-style 224px frame ResNet34:

| Modality | Fold 0 | Fold 1 |
|---|---:|---:|
| Thermal | 247/626 | 309/682 |
| IR | 235/636 | 251/642 |
| Depth_Color | 243/635 | 252/641 |

The official paper's high frame-level scores do not transfer to the stricter
paired-identity-isolated folds. A Fold-0-selected low-weight Thermal blend
changes one raw prediction correctly but is neutral on Fold 1; after the frozen
sequence decoder it regresses V166 by `-6/-3`. Decision: reject.
