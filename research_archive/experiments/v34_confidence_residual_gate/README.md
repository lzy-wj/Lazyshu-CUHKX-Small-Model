# V34 Confidence Residual Gate

V34 refines V32 only within unmatched environment-1 rows. High-confidence V13
pseudo labels retain the fully adapted prediction; lower-confidence rows blend
back toward the unadapted V24 prediction. Fold 0 selects a threshold and low-
confidence adapted weight, then transfers them unchanged to Fold 1.
