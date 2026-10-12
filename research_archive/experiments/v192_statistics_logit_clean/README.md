# V192 Clean Statistics Logit Adapter

V190 showed that mixing samples while the inherited predictor is frozen does
not improve held-out accuracy.  V192 changes only that causal factor: MixUp is
disabled, while masked-modality distillation remains active.  The zero-gated
official-IMU logit residual and all other settings match V190.

