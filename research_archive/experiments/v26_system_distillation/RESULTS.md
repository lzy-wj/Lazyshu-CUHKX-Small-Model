# V26 Results

| Metric | V24 hard pseudo | V26 system distill |
|---|---:|---:|
| Fold 0 direct pair-aware | 550/652 | 550/652 |
| Fold 1 direct pair-aware | 618/705 | 620/705 |
| Combined | 1168/1357 | **1170/1357** |

Kaggle public: `0.90049` (submission `56140482`), below the V24 hard-pseudo
candidate at `0.91044`. The multi-teacher system correction is therefore not
promoted despite its two-row local gain.

The system teacher scored `1195/1357`. Distilling it into one student's heads
retained a two-row improvement over direct hard pseudo-labeling while requiring
only one deployable model and a small head adapter. The candidate passes its
predeclared promotion gate. No additional hyperparameters were searched.
