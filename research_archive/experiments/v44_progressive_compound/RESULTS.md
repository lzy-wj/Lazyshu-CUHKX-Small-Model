# V44 Results

The Fold-0-selected compound combines V32 domain gating, a 10% progressive
student correction, low-confidence X3D-L and skeleton auxiliaries, then the
fixed calibrated decoder.

| Metric | V24 hard pseudo | V44 |
|---|---:|---:|
| Fold 0 pair-aware | 550/652 | **552/652** |
| Fold 1 blind | 618/705 | **628/705** |
| Combined | 1168/1357 | **1180/1357** |

Exact test inference changes only `SM_test_0334` from class `22` to class `21`
relative to V33. Shipping the full progressive checkpoint would exceed the
100 MB budget; a low-rank adapter is technically possible but adds deployment
complexity for one changed test row. The exact CSV remains a research artifact
and is not a submission candidate.

