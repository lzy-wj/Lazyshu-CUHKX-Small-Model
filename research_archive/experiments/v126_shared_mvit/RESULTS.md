# Results

| MViT variant | Raw | Pair-aware |
|---|---:|---:|
| V126 full multimodal | 988/1357 | 1121/1357 |
| V126 visual-only, stopped at epoch 8 | 925/1357 | 1104/1357 |
| V128 stronger visual adaptation | **1012/1357** | **1123/1357** |
| Domain-routed MViT hybrid | **1020/1357** | 1118/1357 |

Structured Skeleton/IMU inputs improve both folds over visual-only MViT. The
stronger backbone adaptation adds 24 raw rows over V126, confirming substantial
Kinetics-to-false-color domain mismatch. However, every Fold-0-selected soft
fusion with V127 underperforms the control; the formal selection is zero MViT
weight and remains 1205/1357. MViT is retained only as independent test-time
teacher evidence, not as a promoted deployment component.

