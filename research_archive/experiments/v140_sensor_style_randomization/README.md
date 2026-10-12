# V140 Sensor Style Randomization

V140 targets cross-device visual shift directly. Depth, Thermal, and IR are
not natural RGB streams; their transfer curves vary across sensors and
recording environments. During training, each visual stream independently
receives a random gamma transform. A second ablation adds the existing mild
gain/offset jitter. All temporal, spatial, model, and optimization settings
remain unchanged from the warm V23 continuation.

Both variants preserve mixup and masked-modality distillation from the source
recipe. Subject-held-out Fold 0 and Fold 1 run in parallel.
