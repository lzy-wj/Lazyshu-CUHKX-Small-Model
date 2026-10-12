# Results

## Outcome

The PoseC3D-style skeleton heatmap branch does not improve the incumbent. Its
standalone accuracy is 37.42% on Fold 0 and 39.01% on Fold 1. A small blend can
regularize Fold 0, but that gain does not transfer to the blind fold.

| Candidate | Fold 0 pair | Fold 1 pair | Combined |
| --- | ---: | ---: | ---: |
| V89 | 548 | 622 | 1,170 |
| V105 selected | 549 | 620 | 1,169 |

The selected blend used global temperature 4 and weight 0.05, chosen only on
Fold 0. The blind Fold-1 regression rejects the branch. The result indicates
that rasterizing the already-available skeleton into low-resolution heatmaps
loses discriminative temporal geometry rather than adding complementary
invariance.

The complete machine-readable audit is in `outputs/evaluation.json`.
