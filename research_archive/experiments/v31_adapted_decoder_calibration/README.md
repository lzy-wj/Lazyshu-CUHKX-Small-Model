# V31 Adapted Decoder Calibration

V31 recalibrates the already frozen pair-aware decoder for V24 hard-adapted
probabilities. Fold 0 selects among a compact grid around the V13 settings;
Fold 1 is evaluated exactly once with the selected setting. No model weights,
training data, pseudo labels, or prediction probabilities are changed.

The reference setting is template strength `1.8`, gap penalty `0.8`, maximum
edits `5`, and minimum margin `0.0`, scoring `1168/1357` over two folds.
