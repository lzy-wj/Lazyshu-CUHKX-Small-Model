# V41 Results

The mixed INT6 checkpoint is `41,207,221` bytes.

| Metric | V24 INT8 | V41 mixed INT6 |
|---|---:|---:|
| Fold 0 direct pair-aware | 550/652 | 550/652 |
| Fold 1 direct pair-aware | 618/705 | 617/705 |
| Combined | **1168/1357** | 1167/1357 |

INT6 almost preserves the public-best student but misses the two-fold retention
criterion by one row. The compressed dual-backbone route is rejected rather
than spending deployment complexity on a locally weaker base.
