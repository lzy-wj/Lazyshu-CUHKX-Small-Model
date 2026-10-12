# V32 Results

| Metric | V24 hard pseudo | V32 domain gate |
|---|---:|---:|
| Fold 0 direct pair-aware | 550/652 | **551/652** |
| Fold 1 direct pair-aware | 618/705 | **623/705** |
| Combined | 1168/1357 | **1174/1357** |

Fold 0 selects adapted weight `0.5` only for unmatched environment-1 rows and
weight `1.0` for the other three domains. The rule transfers unchanged to Fold
1 and gains five more rows there. This candidate needs no additional model
asset beyond the existing V24 checkpoint and head adapter.
