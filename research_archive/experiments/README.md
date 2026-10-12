# V9 Single-Variable Ablation Protocol

## Frozen Baseline

- Subject folds: fold 0 users `[3, 8, 18, 23]`; fold 1 users `[5, 7, 17, 22]`.
- Primary branch: 32-frame multimodal R(2+1)D using the existing RGB/IR cache, skeleton and world-frame IMU.
- Fixed seed, initialization, optimizer, schedule, batch size, TTA, ensemble weights and repeat decoder.
- Fold-0 reference: R(2+1)D raw `67.7914%`; full V9 blend raw `71.0123%`; decoded `76.8405%`.
- Fold-1 is not used for fold-0 design decisions.

## Acceptance Gate

1. Change exactly one scientific variable.
2. Report standalone raw, available-only, fixed-weight ensemble raw and fixed decoder accuracy.
3. Advance to fold 1 only if fold-0 raw improves by at least `0.5` percentage points without decoded regression, or decoded improves by at least two samples without raw regression.
4. Do not inspect or submit Kaggle test labels during ablation.
5. Store code, logs, checkpoints and reports only inside the assigned experiment directory, except uniquely named generated caches.

## Resources

- Conda environment: `Train`.
- Allowed owned GPUs: 2, 3 and 7, each guarded by `outputs/logs/gpu-<id>.lock`.
- GPUs 0, 1, 4, 5 and 6 contain foreign processes and must not be used.
- Shared raw data and pretrained assets are read-only inputs.
