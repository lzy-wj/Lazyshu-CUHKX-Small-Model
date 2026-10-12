# V190 Statistics Logit Adapter

V190 freezes the complete inherited V23 network and trains an independent
official-IMU statistics branch as a zero-gated residual on the final 40-class
logits.  Unlike V187, this branch does not need to align its features to the
existing IMU token geometry.  Initial predictions remain exactly unchanged,
and per-epoch validation probabilities are saved for strict epoch transfer.

