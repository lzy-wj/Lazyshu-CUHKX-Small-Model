# V193 Clean Statistics Token Adapter

V193 is the clean-sample counterpart of the twice-replicated V187/V189 token
adapter.  It freezes all inherited parameters, keeps masked-modality
distillation, and disables MixUp.  This isolates whether sample mixing obscures
the restored official IMU signal.

