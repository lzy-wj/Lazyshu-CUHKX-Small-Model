# Status

Complete and rejected.

- The three-view grouped-joint heatmap cache contains 3,441 clips at
  `[32, 15, 48, 48]`.
- The compact depthwise 3D CNN has 344,760 parameters.
- Standalone validation is 244/652 on Fold 0 and 275/705 on Fold 1.
- The Fold-0-selected blend reaches 1,169/1,357 after pair decoding, one below
  V89. It gains one row on Fold 0 but loses two on the blind Fold 1.
- No test prediction or deployment asset is promoted.
