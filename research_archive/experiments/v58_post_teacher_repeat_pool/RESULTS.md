# V58 Results

V58 applies timestamp/probability-only repeat pooling after V57 has reproduced
the sequence-decoded teacher. Fold 0 selects `maximum_gap=150`,
`minimum_similarity=0.8`, and `power=1.0`.

| Metric | V57 | V58 |
|---|---:|---:|
| Fold 0 direct | 567/652 | 568/652 |
| Fold 1 blind | 628/705 | 625/705 |
| Combined | **1195/1357** | 1193/1357 |

The one-row Fold-0 gain reverses by three rows on Fold 1. V58 is rejected and
does not produce a test candidate or Kaggle submission.
