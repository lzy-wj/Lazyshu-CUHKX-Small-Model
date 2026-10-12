# V181 Relative IMU Experts

This experiment tests the unused chest-relative IMU representation with the
same randomly initialized CNN+BiLSTM protocol as the existing world-coordinate
IMU baseline.  `relative` changes only the representation; `dual` gives world
and relative streams independent encoders and fuses their temporal tokens with
a two-layer Transformer.

Both subject-held-out folds are trained independently.  Blend route, weight,
and temperature are selected on Fold 0 and transferred unchanged to Fold 1.
Promotion is measured against both V127 and the fixed V166 correction mask.

