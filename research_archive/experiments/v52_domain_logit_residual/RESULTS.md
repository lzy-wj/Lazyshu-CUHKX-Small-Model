# V52 Results

| Metric | V51 | V52 |
|---|---:|---:|
| Fold 0 pair-aware | 563/652 | **568/652** |
| Fold 1 blind | 624/705 | **625/705** |
| Combined | 1187/1357 | **1193/1357** |

Domain-conditioned dual-input residuals retain all Fold-1 gain and add six
correct rows overall. They reproduce `1331/1357` V26 teacher decisions and end
only two correct rows below the `1195/1357` system-teacher audit.

The test bundle has 9,720 residual parameters because the test set contains
three of the four domains. Total model assets are `71,131,752` bytes.

