# V189 Low-LR Statistics Adapter

This is the learning-rate replication of V187.  It freezes every inherited
parameter and trains the official-IMU statistics residual at `3e-4` instead of
`1e-3`.  Per-epoch validation probabilities are retained so one Fold-0-selected
epoch can be transferred unchanged to Fold 1.

