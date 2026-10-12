# V39 Results

| Metric | V24 hard pseudo | V39 |
|---|---:|---:|
| Fold 0 raw | 493/652 | **497/652** |
| Fold 1 raw | 577/705 | **582/705** |
| Fold 0 direct pair-aware | 550/652 | **552/652** |
| Fold 1 direct pair-aware | 618/705 | **627/705** |
| Combined direct pair-aware | 1168/1357 | **1179/1357** |

V39 gains eleven rows (`+0.81` percentage points) locally. The gain is larger
on the blind Fold 1 (`+9`) than on Fold 0 (`+2`), supporting transfer rather
than a narrow Fold 0 optimum.

On the 405-row test set, V39 produces exactly the same final labels and SHA256
as V33 after decoding. It therefore adds no submission information while
requiring an extra X3D-L asset, so V33 remains the preferred deployable form.
