# V93 Results

| Model | Raw | Pair-aware | Target-domain weighted pair |
|---|---:|---:|---:|
| V89 scale average | **1077/1357** | **1170/1357** | **88.81%** |
| V93 upper-body 160px | 1066/1357 | 1161/1357 | 88.00% |
| V24 + V93 scale average | 1076/1357 | 1166/1357 | 88.52% |

An unseen upper-body crop improves Fold 1 but damages Fold 0. Inference-only
cropping is rejected. The follow-up must train explicit full/upper-view
consistency rather than assuming the frozen representation is crop invariant.
