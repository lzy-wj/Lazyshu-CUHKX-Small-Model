# V80 Results

All Fold 1 recipes were fixed using Fold 0 and executed once without
intermediate Fold 1 validation.

| Model | Fold 0 | Fold 1 | Combined | Delta vs V23 |
| --- | ---: | ---: | ---: | ---: |
| V23 protected baseline | 487/652 | 567/705 | 1054/1357 | 0 |
| Spatial Transformer, encoder only | 491/652 | 562/705 | 1053/1357 | -1 |
| Spatial Transformer + masked reconstruction | 490/652 | 565/705 | 1055/1357 | +1 |
| Fold-0-selected V23/masked blend | 490/652 | 566/705 | 1056/1357 | +2 |

The encoder-only result shows that retaining the visual `8 x 2 x 2` grid is
useful on Fold 0, but its gain does not transfer to unseen users. Masking 35%
of visual tokens and reconstructing detached CNN features sacrifices one Fold
0 sample but recovers three Fold 1 samples relative to encoder-only training.
The masked model is therefore the safer V80 candidate, though the small net
gain means spatial fusion is not the main remaining bottleneck.

The inference graph contains `76,201,904` parameters. The INT8 checkpoint plus
YOLO is `82,111,878` bytes, below the 100 MB limit. The reconstruction decoder
contains `525,312` additional training-only parameters and is excluded from the
deployment checkpoint.
