# Results

Clip-level temporal-segment supervision improves strict Fold 0 slightly but is
not stable across folds:

| Modality | Fold 0 | Fold 1 |
|---|---:|---:|
| Thermal TSN | 249/626 | 283/682 |
| IR TSN | 244/636 | 254/642 |

Thermal loses 26 Fold-1 rows against the frame-supervised model. IR gains only
small standalone counts and remains far below V166. Decision: reject.
