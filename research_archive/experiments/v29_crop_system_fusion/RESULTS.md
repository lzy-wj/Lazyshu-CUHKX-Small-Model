# V29 Results

| Metric | V26 system distill | V29 fused |
|---|---:|---:|
| Fold 0 direct pair-aware | 550/652 | 550/652 |
| Fold 1 direct pair-aware | 620/705 | **627/705** |
| Combined | 1170/1357 | **1177/1357** |

The local direct pair-aware accuracy is `86.7354%`, a seven-row gain over V26.
The neighboring weights `0.375` and `0.40` tie, which reduces concern that the
result is caused by a narrow validation optimum.

Kaggle public: `0.88557` (submission `56140451`). The wider unadapted crop view
does not transfer to the test domain and is rejected. V24 hard pseudo remains
the public best at `0.91044`.
