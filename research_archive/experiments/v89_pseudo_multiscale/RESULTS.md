# V89 Results

| Model | Raw | Pair-aware | Target-domain weighted pair |
|---|---:|---:|---:|
| V24 pseudo-adapted 128px | 1070/1357 | 1168/1357 | 88.22% |
| V88 pseudo-adapted 160px | **1082/1357** | 1167/1357 | **88.81%** |
| V89 equal multi-scale | 1077/1357 | **1170/1357** | **88.81%** |

The fixed equal average is the first candidate in this branch to exceed the V24
pair-aware reference while also improving target-domain-weighted pair accuracy
on both folds. It is promoted as a genuine scale-marginalization gain; no blend
weight was searched.

The local-only 405-row test candidate changes 9 labels from submitted V24 and 3
from V87. Deployable assets are 71,276,146 bytes: one quantized base model, the
existing YOLO cropper, and two small head deltas. No Kaggle submission has been
made.
