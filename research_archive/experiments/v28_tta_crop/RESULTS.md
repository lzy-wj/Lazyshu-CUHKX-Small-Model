# V28 Results

| Metric | V23 crop 0.75 | V28 crop 0.875 |
|---|---:|---:|
| Fold 0 raw | 487/652 | **491/652** |
| Fold 0 direct pair-aware | 540/652 | **544/652** |
| Fold 1 direct pair-aware | 613/705 | **624/705** |
| Combined direct pair-aware | 1153/1357 | **1168/1357** |

The wider center crop improves both folds without changing model parameters.
It matches the V24 hard-pseudo result but remains below V26 system distillation
at `1170/1357`, so V28 is retained as a complementary inference view rather
than promoted by itself.
