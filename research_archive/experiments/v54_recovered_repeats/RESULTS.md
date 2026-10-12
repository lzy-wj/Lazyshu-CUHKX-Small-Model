# V54 Results

| Metric | V52 | V54 |
|---|---:|---:|
| Fold 0 selection | 568/652 | **574/652** |
| Fold 1 blind | 625/705 | **626/705** |
| Combined | 1193/1357 | **1200/1357** |

Repeat recovery uses only acquisition timestamps and model-probability similarity.
The settings selected on Fold 0 are `maximum_gap=150`,
`minimum_similarity=0.8`, and `power=1.0`.

The local gain is real, but the final pair-aware test decoder cancels all six
pre-decoder top-1 changes. The resulting 405-row CSV is byte-identical to V52,
so V54 is not an informative Kaggle submission and must not consume a submission
slot.
