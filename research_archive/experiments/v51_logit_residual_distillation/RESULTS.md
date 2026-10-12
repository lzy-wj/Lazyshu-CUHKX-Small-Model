# V51 Results

Fold 0 selects retention `0.25` by decoded accuracy with a conservative
stronger-retention tie break.

| Metric | V32 | V51 |
|---|---:|---:|
| Fold 0 pair-aware | 551/652 | **563/652** |
| Fold 1 transductive blind audit | 623/705 | **624/705** |
| Combined | 1174/1357 | **1187/1357** |

The 1,640-parameter residual preserves substantially more of the frozen V26
system teacher than modifying the original model heads. A Fold-0-trained
calibrator does not transfer without target refitting, so this is explicitly a
transductive pseudo-label method rather than target-independent calibration.

