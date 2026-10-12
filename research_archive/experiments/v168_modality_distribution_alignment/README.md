# V168 Modality-Distribution Alignment

The test set has at least five modalities in every row and all six modalities
in 394/405 rows. Training has all six modalities in only 2748/3036 rows, with
the sparse rows concentrated by user and class. V168 tests whether this
availability mismatch creates subject-label shortcuts.

All variants warm-start the same V23 fold checkpoint. Fold 0 is exploratory;
only a robust winner advances unchanged to Fold 1.

- `minimum5`: train and select checkpoints on rows with at least five modalities.
- `full6`: train and select checkpoints on rows with all six modalities.
- `selection5`: retain all training rows and only align checkpoint selection.
- `minimum5_nodistill`: minimum-five training without masked-modality distillation.

The warm-start screen is followed by `minimum5_retrain` and `full6_retrain`,
which reproduce the original V23 optimization from the same V18 initializer.
This tests whether a low-learning-rate V23 warm start merely preserves an
already learned availability shortcut.
