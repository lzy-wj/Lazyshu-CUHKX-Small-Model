# Results

| Objective | Raw | Pair-aware | V127 error oracle |
| --- | ---: | ---: | ---: |
| CE + mixup | 694/1,357 | 857/1,357 | 15/152 |
| Cross-user contrastive | **716/1,357** | **867/1,357** | 10/152 |

Cross-user contrastive learning improves the frozen DINOv2 representation by
22 raw and 10 pair-decoded rows, confirming that person invariance remains
useful. However, the model is far below V127 and agrees with very few of its
residual errors. This evidence does not justify an expensive end-to-end DINOv2
fine-tune or deployment fusion.

The experiment closes generic frame-level image SSL as the next large gain.
V135 instead adapts the proven video encoder inside fixed hard-label families.
