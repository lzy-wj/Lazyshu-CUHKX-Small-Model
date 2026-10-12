# V191 Low-LR Statistics Logit Adapter

Learning-rate replication of V190 at `3e-4`.  All inherited parameters remain
frozen; only the official-IMU statistics-to-logits residual and its zero gate
are trainable.  Validation probabilities are retained for every epoch.

